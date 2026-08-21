# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import numpy as np
import torch

from vllm.v1.attention.ops.pcp_nixl import NixlMemoryRegion, PCPNixlPeerTransport


def _peer_with_wrapper() -> tuple[PCPNixlPeerTransport, MagicMock]:
    peer = PCPNixlPeerTransport(
        world_size=2,
        rank=0,
        device=torch.device("cpu"),
    )
    wrapper = MagicMock()
    peer._wrapper = wrapper
    peer._memory_type = "DRAM"
    return peer, wrapper


def test_submit_read_uses_stable_prepared_region_handles() -> None:
    peer, wrapper = _peer_with_wrapper()
    local = NixlMemoryRegion(
        base_addr=1000,
        block_bytes=64,
        num_blocks=8,
        device_id=0,
        local_xfer_handle=17,
    )
    peer._remote_region_handles[(1, 3)] = 29
    peer._remote_num_blocks[(1, 3)] = 8
    wrapper.make_prepped_xfer.return_value = 41

    handle = peer.submit_read(
        local_region=local,
        local_block_ids=(1, 4),
        source_rank=1,
        remote_region_id=3,
        remote_block_ids=(2, 5),
    )

    assert handle == 41
    args = wrapper.make_prepped_xfer.call_args.args
    assert args[:2] == ("READ", 17)
    assert np.array_equal(args[2], np.asarray((1, 4), dtype=np.int64))
    assert args[3] == 29
    assert np.array_equal(args[4], np.asarray((2, 5), dtype=np.int64))
    wrapper.transfer.assert_called_once_with(41)


def test_check_read_releases_only_completed_handle() -> None:
    peer, wrapper = _peer_with_wrapper()
    wrapper.check_xfer_state.side_effect = ["PROC", "DONE"]

    assert peer.check_read(7) == "PROC"
    wrapper.release_xfer_handle.assert_not_called()

    assert peer.check_read(7) == "DONE"
    wrapper.release_xfer_handle.assert_called_once_with(7)
