# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import numpy as np
import torch

from vllm.v1.attention.ops.pcp_nixl import NixlWrite, PCPNixlPeerTransport


def _peer_with_wrapper() -> tuple[PCPNixlPeerTransport, MagicMock]:
    peer = PCPNixlPeerTransport(
        world_size=2,
        rank=0,
        device=torch.device("cpu"),
    )
    wrapper = MagicMock()
    peer._wrapper = wrapper
    peer._memory_type = "DRAM"
    peer._local_xfer_handle = 17
    peer._local_region_offsets = (0, 8)
    peer._local_num_blocks = (8, 8)
    peer._remote_xfer_handles[1] = 29
    peer._remote_region_offsets[1] = (0, 8)
    peer._remote_num_blocks[(1, 0)] = 8
    peer._remote_num_blocks[(1, 1)] = 8
    return peer, wrapper


def test_submit_write_batch_flattens_multiple_layer_regions() -> None:
    peer, wrapper = _peer_with_wrapper()
    wrapper.make_prepped_xfer.return_value = 41

    handle = peer.submit_write_batch(
        destination_rank=1,
        writes=(
            NixlWrite(
                local_region_id=0,
                remote_region_id=0,
                local_block_ids=np.asarray((1, 4), dtype=np.int64),
                remote_block_ids=np.asarray((2, 5), dtype=np.int64),
            ),
            NixlWrite(
                local_region_id=1,
                remote_region_id=1,
                local_block_ids=np.asarray((2, 5), dtype=np.int64),
                remote_block_ids=np.asarray((3, 6), dtype=np.int64),
            ),
        ),
    )

    assert handle == 41
    args = wrapper.make_prepped_xfer.call_args.args
    assert args[:2] == ("WRITE", 17)
    assert np.array_equal(args[2], np.asarray((1, 4, 10, 13), dtype=np.int64))
    assert args[3] == 29
    assert np.array_equal(args[4], np.asarray((2, 5, 11, 14), dtype=np.int64))
    wrapper.transfer.assert_called_once_with(41)


def test_check_transfer_releases_only_completed_handle() -> None:
    peer, wrapper = _peer_with_wrapper()
    wrapper.check_xfer_state.side_effect = ["PROC", "DONE"]

    assert peer.check_transfer(7) == "PROC"
    wrapper.release_xfer_handle.assert_not_called()

    assert peer.check_transfer(7) == "DONE"
    wrapper.release_xfer_handle.assert_called_once_with(7)
