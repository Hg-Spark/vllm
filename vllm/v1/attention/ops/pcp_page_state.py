# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent PCP-local KV page ownership and validity state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_UNKNOWN_OWNER = -1
_INVALID_BLOCK = -1


@dataclass
class _RequestPageState:
    request_id: str
    owners: np.ndarray
    local_valid_blocks: np.ndarray
    computed_tokens: int = 0


class PCPPageStateTracker:
    """Track logical-page ownership and rank-local valid replicas.

    vLLM remains authoritative for physical block allocation. This tracker only
    records which PCP rank produced each logical page and whether this process'
    current physical block contains a valid copy of that page.
    """

    def __init__(
        self,
        *,
        rank: int,
        block_size: int,
        max_model_len: int,
    ) -> None:
        if block_size <= 0:
            raise ValueError("PCP page-state block size must be positive")
        if max_model_len <= 0:
            raise ValueError("PCP page-state max model length must be positive")
        self.rank = rank
        self.block_size = block_size
        self.max_pages = (max_model_len + block_size - 1) // block_size
        self._states: dict[int, _RequestPageState] = {}

    def _new_state(self, request_id: str) -> _RequestPageState:
        return _RequestPageState(
            request_id=request_id,
            owners=np.full(self.max_pages, _UNKNOWN_OWNER, dtype=np.int16),
            local_valid_blocks=np.full(
                self.max_pages, _INVALID_BLOCK, dtype=np.int32
            ),
        )

    def prepare_request(
        self,
        req_state_idx: int,
        request_id: str,
        computed_tokens: int,
    ) -> _RequestPageState:
        if not 0 <= computed_tokens <= self.max_pages * self.block_size:
            raise ValueError(f"invalid computed token count: {computed_tokens}")
        state = self._states.get(req_state_idx)
        if (
            state is None
            or state.request_id != request_id
            or computed_tokens < state.computed_tokens
        ):
            state = self._new_state(request_id)
            self._states[req_state_idx] = state
        return state

    def has_known_prefix(self, state: _RequestPageState, num_tokens: int) -> bool:
        if num_tokens <= 0:
            return True
        num_pages = (num_tokens + self.block_size - 1) // self.block_size
        return bool((state.owners[:num_pages] >= 0).all())

    def owner(self, state: _RequestPageState, page_idx: int) -> int:
        return int(state.owners[page_idx])

    def assign_owner(
        self,
        state: _RequestPageState,
        page_idx: int,
        owner_rank: int,
    ) -> None:
        current = int(state.owners[page_idx])
        if current not in (_UNKNOWN_OWNER, owner_rank):
            raise RuntimeError(
                "PCP logical page owner changed unexpectedly: "
                f"page={page_idx}, old={current}, new={owner_rank}"
            )
        state.owners[page_idx] = owner_rank

    def local_is_valid(
        self,
        state: _RequestPageState,
        page_idx: int,
        physical_block_id: int,
    ) -> bool:
        return int(state.local_valid_blocks[page_idx]) == physical_block_id

    def mark_local_valid(
        self,
        state: _RequestPageState,
        page_idx: int,
        physical_block_id: int,
    ) -> None:
        state.local_valid_blocks[page_idx] = physical_block_id

    def invalidate_local(self, state: _RequestPageState, page_idx: int) -> None:
        state.local_valid_blocks[page_idx] = _INVALID_BLOCK

    def invalidate_mutable_tail(
        self,
        state: _RequestPageState,
        computed_tokens: int,
    ) -> None:
        if computed_tokens <= 0 or computed_tokens % self.block_size == 0:
            return
        page_idx = computed_tokens // self.block_size
        owner_rank = self.owner(state, page_idx)
        if owner_rank >= 0 and owner_rank != self.rank:
            # The owner will append new tokens into the same physical page in
            # this chunk, so any old non-owner replica becomes stale.
            self.invalidate_local(state, page_idx)

    def advance(self, state: _RequestPageState, computed_tokens: int) -> None:
        if computed_tokens < state.computed_tokens:
            raise RuntimeError(
                "PCP page-state computed tokens moved backwards unexpectedly"
            )
        state.computed_tokens = computed_tokens


__all__ = ["PCPPageStateTracker"]
