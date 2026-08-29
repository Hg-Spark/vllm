# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.model_executor.layers.attention.pcp_runahead as runahead


class _Work:
    def __init__(self) -> None:
        self.wait_count = 0

    def wait(self) -> None:
        self.wait_count += 1


def test_second_layer_waits_for_previous_send_credit(monkeypatch) -> None:
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        ranks=[10, 11],
        device_group=object(),
    )
    works: list[_Work] = []

    monkeypatch.setattr(runahead, "get_pcp_group", lambda: group)

    def fake_isend(tensor, dst, group):
        del tensor, dst, group
        work = _Work()
        works.append(work)
        return work

    monkeypatch.setattr(runahead.dist, "isend", fake_isend)
    runahead.flush_pending_sends()

    runahead.post_layer_send((torch.zeros(2), torch.zeros(2)))
    assert [work.wait_count for work in works] == [0, 0]

    runahead.post_layer_send((torch.ones(2), torch.ones(2)))
    assert [work.wait_count for work in works[:2]] == [1, 1]
    assert [work.wait_count for work in works[2:]] == [0, 0]

    runahead.flush_pending_sends()
    assert [work.wait_count for work in works[2:]] == [1, 1]


def test_consumer_waits_for_full_layer_receive(monkeypatch) -> None:
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=1,
        ranks=[10, 11],
        device_group=object(),
    )
    works: list[_Work] = []

    monkeypatch.setattr(runahead, "get_pcp_group", lambda: group)

    def fake_irecv(tensor, src, group):
        del tensor, src, group
        work = _Work()
        works.append(work)
        return work

    monkeypatch.setattr(runahead.dist, "irecv", fake_irecv)

    received = runahead.recv_layer_like((torch.empty(3, 4), torch.empty(3, 2)))

    assert received[0].shape == (3, 4)
    assert received[1].shape == (3, 2)
    assert [work.wait_count for work in works] == [1, 1]
