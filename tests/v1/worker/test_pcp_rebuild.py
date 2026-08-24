# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from vllm.model_executor.layers.attention.pcp import _pad_prefill_for_collective
from vllm.v1.worker.gpu.pcp_manager import PCPManager, weighted_partition_lengths


def test_weighted_partition_repairs_zero_length_rank() -> None:
    lengths = weighted_partition_lengths(2, (1000.0, 1.0), alignment=1)
    assert lengths == (1, 1)


def test_tiny_prefill_tail_is_replicated_instead_of_empty_slice() -> None:
    scheduled = np.asarray([1], dtype=np.int32)
    computed = np.asarray([0], dtype=np.int32)
    prefilling = np.asarray([True], dtype=np.bool_)
    query_start = np.asarray([0, 1], dtype=np.int32)

    for rank in (0, 1):
        manager = PCPManager(
            pcp_world_size=2,
            pcp_rank=rank,
            device=torch.device("cpu"),
        )
        segments = manager._get_rank_segments(
            rank,
            scheduled,
            computed,
            prefilling,
            query_start,
        )
        assert len(segments) == 1
        assert segments[0].global_batch_slice == slice(0, 1)
        assert segments[0].rank_local_batch_slice == slice(0, 1)


def test_collective_padding_reuses_stream_local_scratch() -> None:
    first_input = torch.arange(6, dtype=torch.float32).view(3, 2)
    second_input = torch.arange(6, 12, dtype=torch.float32).view(3, 2)

    first = _pad_prefill_for_collective(
        first_input,
        num_decode_tokens=0,
        collective_num_tokens=5,
    )
    first_ptr = first.untyped_storage().data_ptr()
    second = _pad_prefill_for_collective(
        second_input,
        num_decode_tokens=0,
        collective_num_tokens=5,
    )

    assert second.untyped_storage().data_ptr() == first_ptr
    torch.testing.assert_close(second[:3], second_input)
    torch.testing.assert_close(second[3:], torch.zeros(2, 2))
