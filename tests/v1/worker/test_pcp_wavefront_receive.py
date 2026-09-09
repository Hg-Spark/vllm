# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.model_executor.layers.attention.pcp_wavefront_runtime as wavefront


class _Work:
    def __init__(self) -> None:
        self.wait_count = 0

    def wait(self) -> None:
        self.wait_count += 1


def test_layer_receive_post_is_separate_from_wait(monkeypatch) -> None:
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=1,
        ranks=[10, 11],
        device_group=object(),
    )
    works: list[_Work] = []

    monkeypatch.setattr(wavefront, "get_pcp_group", lambda: group)

    def fake_batch_isend_irecv(ops):
        batch_works = [_Work() for _ in ops]
        works.extend(batch_works)
        return batch_works

    monkeypatch.setattr(wavefront.dist, "batch_isend_irecv", fake_batch_isend_irecv)

    pending = wavefront.post_layer_receive_into(
        (torch.empty(3, 4), torch.empty(3, 2))
    )
    assert [work.wait_count for work in works] == [0, 0]

    wavefront.wait_layer_receive(pending)
    assert [work.wait_count for work in works] == [1, 1]
