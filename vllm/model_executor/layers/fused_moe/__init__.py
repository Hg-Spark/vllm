# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Any

from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    activation_without_mul,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.fused_moe.routed_experts import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.triton_utils import HAS_TRITON

_base_fused_moe_parallel_make = FusedMoEParallelConfig.make


def _make_pcp_local_moe_parallel_config(
    tp_size_: int,
    pcp_size_: int,
    dp_size_: int,
    sp_size_: int,
    vllm_parallel_config,
):
    """Replicate MoE weights across PCP for the TP=DP=1, EP-off path."""
    if (
        pcp_size_ > 1
        and tp_size_ == 1
        and dp_size_ == 1
        and not vllm_parallel_config.enable_expert_parallel
    ):
        pcp_size_ = 1
    return _base_fused_moe_parallel_make(
        tp_size_,
        pcp_size_,
        dp_size_,
        sp_size_,
        vllm_parallel_config,
    )


FusedMoEParallelConfig.make = staticmethod(_make_pcp_local_moe_parallel_config)

_config: dict[str, Any] | None = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> dict[str, Any] | None:
    return _config


__all__ = [
    "FusedMoEFactory",
    "FusedMoERouter",
    "FusedMoEConfig",
    "FusedMoEQuantConfig",
    "FusedMoEParallelConfig",
    "FusedMoEMethodBase",
    "MoEActivation",
    "UnquantizedFusedMoEMethod",
    "FusedMoeWeightScaleSupported",
    "FusedMoEExpertsModular",
    "FusedMoEActivationFormat",
    "FusedMoEPrepareAndFinalizeModular",
    "GateLinear",
    "MoERunner",
    "RoutingMethodType",
    "RoutedExperts",
    "SharedExperts",
    "activation_without_mul",
    "apply_moe_activation",
    "fused_moe_make_expert_params_mapping",
    "override_config",
    "get_config",
]

if HAS_TRITON:
    # import to register the custom ops
    from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
        BatchedDeepGemmExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
        CutlassBatchedExpertsFp8,
        CutlassExpertsFp8,
        CutlassExpertsW4A8Fp8,
    )
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
        BatchedTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
        TritonOrDeepGemmExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonExperts,
        TritonWNA16Experts,
    )
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        XPUExperts,
        XPUExpertsFp8,
        XPUExpertsMxFp4,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts,
        get_config_file_name,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk,
    )
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        GroupedTopk,
    )

    __all__ += [
        "AiterExperts",
        "fused_topk",
        "fused_experts",
        "get_config_file_name",
        "GroupedTopk",
        "CutlassExpertsFp8",
        "CutlassBatchedExpertsFp8",
        "CutlassExpertsW4A8Fp8",
        "TritonExperts",
        "TritonWNA16Experts",
        "BatchedTritonExperts",
        "DeepGemmExperts",
        "BatchedDeepGemmExperts",
        "TritonOrDeepGemmExperts",
        "XPUExperts",
        "XPUExpertsFp8",
        "XPUExpertsBlockFp8",
        "XPUExpertsMxFp8",
        "XPUExpertsMxFp4",
    ]
else:
    # Some model classes directly use the custom ops. Add placeholders
    # to avoid import errors.
    def _raise_exception(method: str):
        raise NotImplementedError(f"{method} is not implemented as lack of triton.")

    fused_topk = lambda *args, **kwargs: _raise_exception("fused_topk")
    fused_experts = lambda *args, **kwargs: _raise_exception("fused_experts")
