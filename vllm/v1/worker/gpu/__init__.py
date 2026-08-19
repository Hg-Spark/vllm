# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.gpu import pcp_manager as _pcp_manager
from vllm.v1.worker.gpu.pcp_runahead_ext import install_pcp_extensions

install_pcp_extensions(_pcp_manager)
