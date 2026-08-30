# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persist and reuse the memory-profiling result across engine boots.

On startup, vLLM measures how much GPU memory the KV cache can use and
computes the ``--kv-cache-memory`` value that reproduces that allocation.
For a fixed (model, config, hardware, library) combination the result is
deterministic, yet it is re-measured on every boot.

When ``VLLM_ENABLE_STARTUP_PLAN=1``, each worker persists that value under
``{VLLM_CACHE_ROOT}/startup_plan/`` (regenerable derived state, alongside
the torch.compile cache), keyed by a fingerprint of everything the value
depends on, and later boots apply it automatically -- skipping the
memory-profiling measurement and the CUDA-graph memory estimation pass --
if and only if the fingerprint matches and the device has at least as much
free memory as when the plan was recorded. On any mismatch the worker
falls back to full profiling, so a stale plan costs nothing and is never
trusted.

Weighted PCP also uses this startup hook to make the one-shot model profile
cover each rank's causal prefix. The scheduler's global token budget is
preserved; only the profile call temporarily sees the cumulative token budget
visible to the current PCP rank.
"""

import hashlib
import json
import os
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

PLAN_SCHEMA_VERSION = 1


def compute_plan_fingerprint(
    vllm_config: VllmConfig, rank: int, world_size: int
) -> str:
    """Hash everything the profiled KV-cache memory value depends on.

    ``VllmConfig.compute_hash()`` covers the vLLM version and the model,
    cache, parallel, and compilation configs, but deliberately contains no
    device identity (``DeviceConfig.compute_hash`` is empty), so device
    name, total memory, compute capability, and the torch/CUDA build are
    added here. The vLLM version is also pinned as an explicit factor so
    version invalidation holds no matter how ``compute_hash`` evolves.
    Rank is included because per-rank memory use differs under TP/PP.
    Driver-only changes are not part of the key; the free-memory gate at
    apply time bounds the residual risk.
    """
    from vllm import __version__ as vllm_version

    capability = current_platform.get_device_capability()
    factors = {
        "schema": PLAN_SCHEMA_VERSION,
        "vllm": vllm_version,
        "vllm_config": vllm_config.compute_hash(),
        "device_name": current_platform.get_device_name(),
        "device_total_memory": current_platform.get_device_total_memory(),
        "device_capability": str(capability) if capability else "",
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "rank": rank,
        "world_size": world_size,
    }
    digest = hashlib.sha256(json.dumps(factors, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def _plan_path(fingerprint: str) -> str:
    return os.path.join(
        envs.VLLM_CACHE_ROOT, "startup_plan", f"startup_plan_{fingerprint}.json"
    )


def _load_plan(fingerprint: str) -> dict | None:
    path = _plan_path(fingerprint)
    try:
        with open(path) as f:
            plan = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Ignoring unreadable startup plan %s: %s", path, e)
        return None
    if (
        plan.get("schema") != PLAN_SCHEMA_VERSION
        or plan.get("fingerprint") != fingerprint
    ):
        return None
    return plan


def _applicable_kv_cache_memory_bytes(
    plan: dict, current_free_memory: int
) -> int | None:
    kv_bytes = plan.get("kv_cache_memory_bytes")
    baseline = plan.get("free_memory_baseline")
    if not isinstance(kv_bytes, int) or not isinstance(baseline, int):
        return None
    if kv_bytes <= 0:
        return None
    if current_free_memory < baseline:
        logger.info(
            "Startup plan not applied: current free memory (%.2f GiB) is "
            "below the recorded baseline (%.2f GiB); falling back to full "
            "memory profiling.",
            current_free_memory / (1 << 30),
            baseline / (1 << 30),
        )
        return None
    return kv_bytes


def _pcp_causal_profile_tokens(
    per_rank_tokens: tuple[int, ...], pcp_rank: int
) -> int:
    """Return the cumulative causal context visible to one PCP rank.

    PCP partitions a causal prefill into ordered contiguous segments. Rank r
    computes its local segment while attending to every earlier segment, so its
    profile must cover ``sum(lengths[: r + 1])`` rather than only its local
    segment length. For PCP=2 this makes rank1 profile the full global context.
    """
    if not 0 <= pcp_rank < len(per_rank_tokens):
        raise ValueError(
            f"Invalid PCP rank {pcp_rank} for {len(per_rank_tokens)} partitions"
        )
    return max(1, sum(per_rank_tokens[: pcp_rank + 1]))


def _prepare_pcp_profile_run(worker: "Worker") -> None:
    """Make the next profile call cover weighted PCP causal rank execution.

    Memory profiling runs before the PCP manager and KV cache are initialized,
    so the normal batch partitioner cannot model each rank's causal prefix.
    Install the PCP tile/FFN wrappers early, then wrap exactly one
    ``profile_run`` call with the cumulative token budget visible to this rank.
    The global scheduler/model-runner limit is restored even if profiling
    raises.

    This single scalar is a conservative proxy: it covers the rank's causal
    context even though the real Wavefront path may have fewer rank-local query
    rows than context rows.
    """
    parallel_config = worker.vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    if pcp_size <= 1:
        return

    additional_config = worker.vllm_config.additional_config
    if not isinstance(additional_config, dict):
        return

    from vllm.v1.worker.gpu.pcp_tile import configure_pcp_tiling

    tile_size = configure_pcp_tiling(worker.vllm_config)

    # The Wavefront execution planner is selected only when explicit weighted
    # partitioning is configured. Keep legacy PCP profiling unchanged.
    if "pcp_partition_weights" not in additional_config:
        return

    from vllm.distributed.parallel_state import get_pcp_group
    from vllm.v1.worker.gpu.pcp_weighted_partition import (
        parse_pcp_partition_weights,
        weighted_partition_lengths,
    )

    runner = worker.model_runner
    global_num_tokens = runner.max_num_tokens
    pcp_rank = get_pcp_group().rank_in_group
    weights = parse_pcp_partition_weights(additional_config, pcp_size)
    per_rank_tokens = weighted_partition_lengths(global_num_tokens, weights)
    causal_profile_tokens = _pcp_causal_profile_tokens(per_rank_tokens, pcp_rank)

    original_profile_run = runner.profile_run

    def profile_run_once() -> None:
        previous_max_num_tokens = runner.max_num_tokens
        runner.max_num_tokens = causal_profile_tokens
        try:
            original_profile_run()
        finally:
            runner.max_num_tokens = previous_max_num_tokens
            runner.profile_run = original_profile_run

    runner.profile_run = profile_run_once
    logger.info(
        "Prepared PCP-aware startup profile: rank=%d global_tokens=%d "
        "local_tokens=%d causal_profile_tokens=%d per_rank_tokens=%s "
        "tile_size=%d",
        pcp_rank,
        global_num_tokens,
        per_rank_tokens[pcp_rank],
        causal_profile_tokens,
        per_rank_tokens,
        tile_size,
    )


def maybe_apply_startup_plan(worker: "Worker") -> None:
    """If enabled and ``--kv-cache-memory`` was not set explicitly, apply a
    persisted plan by setting ``worker.cache_config.kv_cache_memory_bytes``.
    No-op unless ``VLLM_ENABLE_STARTUP_PLAN=1``.

    The PCP profile wrapper is prepared independently of startup-plan caching,
    because ``determine_available_memory`` always invokes this hook immediately
    before its one-shot model profile.
    """
    _prepare_pcp_profile_run(worker)

    if (
        not envs.VLLM_ENABLE_STARTUP_PLAN
        or worker.cache_config.kv_cache_memory_bytes is not None
    ):
        return
    fingerprint = compute_plan_fingerprint(
        worker.vllm_config, worker.rank, worker.parallel_config.world_size
    )
    plan = _load_plan(fingerprint)
    if plan is None:
        return
    current_free_memory = worker.init_snapshot.free_memory
    kv_bytes = _applicable_kv_cache_memory_bytes(plan, current_free_memory)
    if kv_bytes is None:
        return
    logger.info(
        "Applying persisted startup plan (fingerprint %s): "
        "kv_cache_memory_bytes=%d (%.2f GiB), recorded free-memory "
        "baseline %.2f GiB, current %.2f GiB. Memory profiling will "
        "be skipped.",
        fingerprint,
        kv_bytes,
        kv_bytes / (1 << 30),
        plan["free_memory_baseline"] / (1 << 30),
        current_free_memory / (1 << 30),
    )
    worker.cache_config.kv_cache_memory_bytes = kv_bytes


def maybe_save_startup_plan(worker: "Worker", kv_cache_memory_bytes: int) -> None:
    """Atomically persist this boot's profiling result for future boots.
    No-op unless ``VLLM_ENABLE_STARTUP_PLAN=1``; failures are logged,
    never raised."""
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return
    fingerprint = compute_plan_fingerprint(
        worker.vllm_config, worker.rank, worker.parallel_config.world_size
    )
    path = _plan_path(fingerprint)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "schema": PLAN_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "kv_cache_memory_bytes": int(kv_cache_memory_bytes),
            "free_memory_baseline": int(worker.init_snapshot.free_memory),
        }
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
        logger.info("Saved startup plan to %s", path)
    except OSError as e:
        logger.warning("Failed to save startup plan to %s: %s", path, e)
