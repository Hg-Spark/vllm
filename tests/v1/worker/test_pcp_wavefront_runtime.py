# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.attention.pcp_wavefront_runtime as wavefront


class _Work:
    def __init__(self) -> None:
        self.wait_count = 0

    def wait(self) -> None:
        self.wait_count += 1


def test_max_outstanding_layers_from_env(monkeypatch) -> None:
    monkeypatch.setenv(wavefront._MAX_OUTSTANDING_LAYERS_ENV, "3")
    assert wavefront._read_max_outstanding_layers() == 3


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_max_outstanding_layers_rejects_invalid_values(monkeypatch, value) -> None:
    monkeypatch.setenv(wavefront._MAX_OUTSTANDING_LAYERS_ENV, value)
    with pytest.raises(RuntimeError):
        wavefront._read_max_outstanding_layers()


def test_second_layer_waits_for_previous_send_credit(monkeypatch) -> None:
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        ranks=[10, 11],
        device_group=object(),
    )
    works: list[_Work] = []
    batch_sizes: list[int] = []

    monkeypatch.setattr(wavefront, "get_pcp_group", lambda: group)
    monkeypatch.setattr(wavefront, "_MAX_OUTSTANDING_LAYERS", 1)

    def fake_batch_isend_irecv(ops):
        batch_sizes.append(len(ops))
        batch_works = [_Work() for _ in ops]
        works.extend(batch_works)
        return batch_works

    monkeypatch.setattr(
        wavefront.dist,
        "batch_isend_irecv",
        fake_batch_isend_irecv,
    )
    wavefront.flush_pending_sends()

    wavefront.post_layer_transfer((torch.zeros(2), torch.zeros(2)))
    assert batch_sizes == [2]
    assert [work.wait_count for work in works] == [0, 0]

    wavefront.post_layer_transfer((torch.ones(2), torch.ones(2)))
    assert batch_sizes == [2, 2]
    assert [work.wait_count for work in works[:2]] == [1, 1]
    assert [work.wait_count for work in works[2:]] == [0, 0]

    wavefront.flush_pending_sends()
    assert [work.wait_count for work in works[2:]] == [1, 1]


def test_two_outstanding_layers_waits_on_third_handoff(monkeypatch) -> None:
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        ranks=[10, 11],
        device_group=object(),
    )
    works: list[_Work] = []

    monkeypatch.setattr(wavefront, "get_pcp_group", lambda: group)
    monkeypatch.setattr(wavefront, "_MAX_OUTSTANDING_LAYERS", 2)

    def fake_batch_isend_irecv(ops):
        batch_works = [_Work() for _ in ops]
        works.extend(batch_works)
        return batch_works

    monkeypatch.setattr(
        wavefront.dist,
        "batch_isend_irecv",
        fake_batch_isend_irecv,
    )
    wavefront.flush_pending_sends()

    payload = (torch.zeros(2), torch.zeros(2))
    wavefront.post_layer_transfer(payload)
    wavefront.post_layer_transfer(payload)
    assert [work.wait_count for work in works] == [0, 0, 0, 0]

    wavefront.post_layer_transfer(payload)
    assert [work.wait_count for work in works[:2]] == [1, 1]
    assert [work.wait_count for work in works[2:]] == [0, 0, 0, 0]

    wavefront.flush_pending_sends()


def test_consumer_waits_for_full_layer_receive(monkeypatch) -> None:
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=1,
        ranks=[10, 11],
        device_group=object(),
    )
    works: list[_Work] = []
    batch_sizes: list[int] = []

    monkeypatch.setattr(wavefront, "get_pcp_group", lambda: group)

    def fake_batch_isend_irecv(ops):
        batch_sizes.append(len(ops))
        batch_works = [_Work() for _ in ops]
        works.extend(batch_works)
        return batch_works

    monkeypatch.setattr(
        wavefront.dist,
        "batch_isend_irecv",
        fake_batch_isend_irecv,
    )

    received = wavefront.recv_layer_payload(
        (torch.empty(3, 4), torch.empty(3, 2))
    )

    assert batch_sizes == [2]
    assert received[0].shape == (3, 4)
    assert received[1].shape == (3, 2)
    assert [work.wait_count for work in works] == [1, 1]
