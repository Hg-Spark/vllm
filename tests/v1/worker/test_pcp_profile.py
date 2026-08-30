# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm.distributed.parallel_state as parallel_state
from vllm.v1.worker import startup_plan
from vllm.v1.worker.gpu import pcp_microbatch


def _make_worker(*, rank: int, weights: list[float], max_num_tokens: int = 10):
    seen_profile_tokens: list[int] = []
    runner = SimpleNamespace(max_num_tokens=max_num_tokens)

    def profile_run() -> None:
        seen_profile_tokens.append(runner.max_num_tokens)

    runner.profile_run = profile_run
    parallel_config = SimpleNamespace(prefill_context_parallel_size=len(weights))
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        additional_config={
            "pcp_partition_weights": weights,
            "pcp_microbatch_size": 4,
        },
    )
    worker = SimpleNamespace(vllm_config=vllm_config, model_runner=runner)
    return worker, runner, profile_run, seen_profile_tokens, rank


@pytest.mark.parametrize(
    ("rank", "weights", "expected_tokens"),
    [
        (0, [1, 3], 3),
        (1, [1, 3], 7),
        (0, [1, 1], 5),
        (1, [1, 1], 5),
    ],
)
def test_prepare_pcp_profile_run_uses_rank_local_tokens(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    weights: list[float],
    expected_tokens: int,
) -> None:
    worker, runner, original_profile_run, seen_profile_tokens, _ = _make_worker(
        rank=rank,
        weights=weights,
    )
    configured: list[object] = []
    monkeypatch.setattr(
        pcp_microbatch,
        "configure_pcp_memory_microbatching",
        lambda config: configured.append(config) or 4,
    )
    monkeypatch.setattr(
        parallel_state,
        "get_pcp_group",
        lambda: SimpleNamespace(rank_in_group=rank),
    )

    startup_plan._prepare_pcp_profile_run(worker)
    assert runner.max_num_tokens == 10

    runner.profile_run()

    assert seen_profile_tokens == [expected_tokens]
    assert runner.max_num_tokens == 10
    assert runner.profile_run is original_profile_run
    assert configured == [worker.vllm_config]


def test_prepare_pcp_profile_run_restores_global_limit_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = SimpleNamespace(max_num_tokens=11)

    def profile_run() -> None:
        assert runner.max_num_tokens == 7
        raise RuntimeError("profile failed")

    runner.profile_run = profile_run
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
            additional_config={
                "pcp_partition_weights": [1, 2],
                "pcp_microbatch_size": 4,
            },
        ),
        model_runner=runner,
    )
    monkeypatch.setattr(
        pcp_microbatch,
        "configure_pcp_memory_microbatching",
        lambda config: 4,
    )
    monkeypatch.setattr(
        parallel_state,
        "get_pcp_group",
        lambda: SimpleNamespace(rank_in_group=1),
    )

    startup_plan._prepare_pcp_profile_run(worker)
    with pytest.raises(RuntimeError, match="profile failed"):
        runner.profile_run()

    assert runner.max_num_tokens == 11
    assert runner.profile_run is profile_run


def test_prepare_pcp_profile_run_keeps_legacy_pcp_profile_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = SimpleNamespace(max_num_tokens=10)
    calls: list[int] = []

    def profile_run() -> None:
        calls.append(runner.max_num_tokens)

    runner.profile_run = profile_run
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
            additional_config={"pcp_microbatch_size": 4},
        ),
        model_runner=runner,
    )
    monkeypatch.setattr(
        pcp_microbatch,
        "configure_pcp_memory_microbatching",
        lambda config: 4,
    )

    startup_plan._prepare_pcp_profile_run(worker)
    runner.profile_run()

    assert calls == [10]
    assert runner.profile_run is profile_run
