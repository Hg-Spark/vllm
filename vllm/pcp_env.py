# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early registration for PCP-specific environment flags."""

import os

from vllm import envs


def _pcp_nvtx_enabled() -> bool:
    return os.getenv("VLLM_PCP_NVTX", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# vLLM validates all VLLM_* variables against this registry. Register the PCP
# profiling flag during package initialization so entrypoints and spawned
# workers accept it without changing unrelated environment defaults.
envs.environment_variables.setdefault("VLLM_PCP_NVTX", _pcp_nvtx_enabled)
