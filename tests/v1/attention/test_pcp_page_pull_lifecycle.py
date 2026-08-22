# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import torch

from vllm.v1.attention.ops.pcp_page_pull import PCPPagePlan, PCPPagePullTransport


class _ObservedLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.entered = threading.Event()

    def acquire(self) -> None:
        self._lock.acquire()

    def release(self) -> None:
        self._lock.release()

    def __enter__(self):
        self.entered.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._lock.release()


def _plan() -> PCPPagePlan:
    return PCPPagePlan(
        segment_to_rank=(0, 1),
        blocks_by_segment=((0,), (1,)),
        block_size=16,
    )


def test_configure_step_switches_state_under_progress_lock() -> None:
    transport = PCPPagePullTransport(
        world_size=2,
        rank=0,
        device=torch.device("cpu"),
    )
    transport._pending_ready.append(object())  # type: ignore[arg-type]
    transport._ready_waiting.append((9, 9))
    transport._inflight[(9, 9)] = object()  # type: ignore[assignment]
    transport._done_pairs.add((9, 9))

    lock = _ObservedLock()
    lock.acquire()
    transport._progress_lock = lock  # type: ignore[assignment]
    errors: list[Exception] = []

    def configure() -> None:
        try:
            transport.configure_step(epoch=7, plan=_plan())
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=configure)
    thread.start()
    assert lock.entered.wait(1.0)

    assert transport._epoch == 0
    assert transport._plan is None
    assert transport._ready_waiting
    assert transport._inflight
    assert transport._done_pairs

    lock.release()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert not errors
    assert transport._epoch == 7
    assert transport._plan is not None
    assert not transport._step_finished
    assert not transport._pending_ready
    assert not transport._ready_waiting
    assert not transport._inflight
    assert not transport._done_pairs


def test_finish_step_clears_plan_under_progress_lock() -> None:
    transport = PCPPagePullTransport(
        world_size=2,
        rank=0,
        device=torch.device("cpu"),
    )
    transport.configure_step(epoch=1, plan=_plan())

    lock = _ObservedLock()
    lock.acquire()
    transport._progress_lock = lock  # type: ignore[assignment]
    errors: list[Exception] = []

    def finish() -> None:
        try:
            transport.finish_step()
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=finish)
    thread.start()
    assert lock.entered.wait(1.0)

    assert transport._plan is not None
    assert not transport._step_finished

    lock.release()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert not errors
    assert transport._plan is None
    assert transport._step_finished


def test_close_stops_persistent_progress_thread() -> None:
    transport = PCPPagePullTransport(
        world_size=1,
        rank=0,
        device=torch.device("cpu"),
    )
    transport._layer_names = ("layer",)
    transport._layer_memory = [object()]  # type: ignore[list-item]
    transport.configure_step(
        epoch=1,
        plan=PCPPagePlan(
            segment_to_rank=(0,),
            blocks_by_segment=((0,),),
            block_size=16,
        ),
    )
    transport.close()
    assert transport._progress_thread is None
