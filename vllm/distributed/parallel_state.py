# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""vLLM distributed state.
It takes over the control of the distributed environment from PyTorch.
The typical workflow is:

- call `init_distributed_environment` to initialize the distributed environment.
- call `initialize_model_parallel` or `ensure_model_parallel_initialized` to
 initialize the model parallel groups.

- any code dealing with the distributed stuff

- call `destroy_model_parallel` to destroy the model parallel groups.
- call `destroy_distributed_environment` to destroy the distributed environment.

If you only need to use the distributed environment without model/pipeline
 parallelism, you can skip the model parallel initialization and destruction
 steps.
"""

import contextlib
import gc
import pickle
import weakref
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Protocol
from unittest.mock import patch

import torch
import torch.distributed
import torch.distributed._functional_collectives as funcol
import torch.distributed._symmetric_memory
from torch.distributed import Backend, ProcessGroup, Store

import vllm.envs as envs
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.utils import (
    StatelessProcessGroup,
    get_cached_tcp_store_client,
)
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.network_utils import get_distributed_init_method
from vllm.utils.system_utils import suppress_stdout
from vllm.utils.torch_utils import (
    direct_register_custom_op,
)

if TYPE_CHECKING:
    from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream


TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


class Handle(Protocol):
    """Minimal async work handle used by P2P send/recv methods."""

    def is_completed(self) -> bool: ...

    def wait(self) -> None: ...


def _split_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | Any],
) -> tuple[list[tuple[str, Any]], list[torch.Tensor]]:
    metadata_list: list[tuple[str, Any]] = []
    tensor_list: list[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size()))
            )
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


_group_name_counter: dict[str, int] = {}


def _get_unique_name(name: str) -> str:
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


_groups: dict[str, Callable[[], "GroupCoordinator | None"]] = {}


def _register_group(group: "GroupCoordinator") -> None:
    _groups[group.unique_name] = weakref.ref(group)


def _apply_to_device_comms(
    action: Callable[[DeviceCommunicatorBase], None],
) -> None:
    comms = []
    for group_ref in _groups.values():
        group = group_ref()
        if group is None:
            continue
        dc = group.device_communicator
        if dc is None:
            continue
        comms.append(dc)

    for dc in comms:
        action(dc)


def all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_reduce_out_place(tensor)


def all_reduce_fake(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    return torch.empty_like(tensor)


def reduce_scatter(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._reduce_scatter_out_place(tensor, dim)


def reduce_scatter_fake(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    new_shape = list(tensor.shape)
    new_shape[dim] = tensor.shape[dim] // world_size
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)


def all_gather(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_gather_out_place(tensor, dim)


def all_gather_fake(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    new_shape = list(tensor.shape)
    new_shape[dim] = tensor.shape[dim] * world_size
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)


def patched_fused_scaled_matmul_reduce_scatter_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    bias: torch.Tensor | None = None,
    result_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    if A_scale.numel() > 1:
        if A_scale.shape[:-1] != A.shape[:-1]:
            raise ValueError(
                "For row-wise scaling, the leading dims of A_scale "
                "must match the leading dims of A "
                f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
            )
        A_scale = A_scale.flatten(0, -2).contiguous()
    elif A_scale.numel() != 1:
        raise ValueError(
            "Invalid A_scale shape "
            f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
        )

    C = torch._scaled_mm(
        A.flatten(0, -2).contiguous(),
        B,
        A_scale,
        B_scale,
        bias,
        result_scale,
        out_dtype,
        use_fast_accum,
    )
    C = C.view(*output_shape[:-1], B.shape[1])
    res = funcol.reduce_scatter_tensor(
        C,
        reduce_op,
        orig_scatter_dim,
        group_name,
    )
    res = funcol.wait_tensor(res)
    return res


def _platform_device_type() -> str:
    from vllm.platforms import current_platform

    if current_platform.is_cuda_alike():
        return "cuda"
    elif current_platform.is_xpu():
        return "xpu"
    elif current_platform.is_out_of_tree():
        return current_platform.device_name
    else:
        return "cpu"


def _device_backend_str(torch_distributed_backend: str | Backend) -> str:
    backend_str = str(torch_distributed_backend)
    if ":" in backend_str:
        return backend_str
    return f"{_platform_device_type()}:{backend_str}"


def _create_subgroups_split_group(
    group_ranks: list[list[int]],
    group_name: str,
    torch_distributed_backend: str | Backend,
) -> tuple[ProcessGroup, ProcessGroup]:
    from vllm.distributed.utils import (
        get_cpu_distributed_timeout_or_none,
        get_distributed_timeout_or_none,
    )

    device_backend_str = _device_backend_str(torch_distributed_backend)
    self_device_group = torch.distributed.split_group(
        split_ranks=group_ranks,
        group_desc=f"{group_name}:device",
        backend=device_backend_str,
        timeout=get_distributed_timeout_or_none(),
    )
    self_cpu_group = torch.distributed.split_group(
        split_ranks=group_ranks,
        group_desc=f"{group_name}:cpu",
        backend=f"cpu:gloo,{device_backend_str}",
        timeout=get_cpu_distributed_timeout_or_none(),
    )
    return self_device_group, self_cpu_group


def patched_fused_scaled_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: str,
    output_shape: list[int],
    bias: torch.Tensor | None = None,
    result_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    return torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
        A,
        B,
        A_scale,
        B_scale,
        reduce_op,
        orig_scatter_dim,
        scatter_dim_after_maybe_reshape,
        group_name,
        output_shape,
        bias,
        result_scale,
        out_dtype,
        use_fast_accum,
    )


direct_register_custom_op(
    op_name="all_reduce", op_func=all_reduce, fake_impl=all_reduce_fake
)
direct_register_custom_op(
    op_name="reduce_scatter",
    op_func=reduce_scatter,
    fake_impl=reduce_scatter_fake,
)
direct_register_custom_op(
    op_name="all_gather", op_func=all_gather, fake_impl=all_gather_fake
)
direct_register_custom_op(
    op_name="patched_fused_scaled_matmul_reduce_scatter",
    op_func=patched_fused_scaled_matmul_reduce_scatter,
    fake_impl=patched_fused_scaled_matmul_reduce_scatter_fake,
)


class GroupCoordinator:
    rank: int
    ranks: list[int]
    world_size: int
    local_rank: int
    rank_in_group: int
    cpu_group: ProcessGroup
    device_group: ProcessGroup
    device_communicator: DeviceCommunicatorBase | None
    mq_broadcaster: Any | None

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
        use_all2all: bool = False,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_index = local_rank
        assert local_rank >= 0
        self_device_group = None
        self_cpu_group = None

        if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
            self_device_group, self_cpu_group = _create_subgroups_split_group(
                group_ranks, group_name, torch_distributed_backend
            )
            for ranks in group_ranks:
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    break
        else:
            from vllm.distributed.utils import (
                get_cpu_distributed_timeout_or_none,
                get_distributed_timeout_or_none,
            )

            timeout = get_cpu_distributed_timeout_or_none()
            device_timeout = get_distributed_timeout_or_none()
            for ranks in group_ranks:
                device_group = torch.distributed.new_group(
                    ranks,
                    backend=torch_distributed_backend,
                    timeout=device_timeout,
                )
                with suppress_stdout():
                    cpu_group = torch.distributed.new_group(
                        ranks, backend="gloo", timeout=timeout
                    )
                if self.rank in ranks:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self_device_group = device_group
                    self_cpu_group = cpu_group

        assert self_cpu_group is not None
        assert self_device_group is not None
        self.group_ranks = group_ranks
        self.torch_distributed_backend = torch_distributed_backend
        self.cpu_group = self_cpu_group
        self.device_group = self_device_group

        from vllm.platforms import current_platform

        if current_platform.is_cuda_alike():
            visible_device_index = (
                current_platform.logical_device_id_to_visible_device_id(
                    self.device_index
                )
            )
            self.device = torch.device(f"cuda:{visible_device_index}")
        elif current_platform.is_xpu():
            self.device = torch.device(f"xpu:{self.device_index}")
        elif current_platform.is_out_of_tree():
            self.device = torch.device(
                f"{current_platform.device_name}:{self.device_index}"
            )
        else:
            self.device = torch.device("cpu")

        self.use_device_communicator = use_device_communicator
        self.device_communicator = None
        if use_device_communicator and self.world_size > 1:
            device_comm_cls = resolve_obj_by_qualname(
                current_platform.get_device_communicator_cls()
            )
            self.device_communicator = device_comm_cls(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
                use_all2all=use_all2all,
            )

        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        self.mq_broadcaster: MessageQueue | None = None
        if use_message_queue_broadcaster and self.world_size > 1:
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6
            )
        self.use_custom_op_call = (
            current_platform.is_tpu() or current_platform.use_custom_op_collectives()
        )
        self.use_cpu_custom_send_recv = (
            current_platform.is_cpu()
            and self.device_communicator
            and getattr(self.device_communicator, "supports_tensor_dict", False)
        )

    def make_sibling_device_group(self, group_desc: str | None = None) -> ProcessGroup:
        from vllm.distributed.utils import get_distributed_timeout_or_none

        device_timeout = get_distributed_timeout_or_none()
        sibling: ProcessGroup | None = None
        for ranks in self.group_ranks:
            pg = torch.distributed.new_group(
                ranks,
                backend=self.torch_distributed_backend,
                group_desc=group_desc,
                timeout=device_timeout,
            )
            if self.rank in ranks:
                sibling = pg
        assert sibling is not None
        return sibling

    def create_mq_broadcaster(
        self, writer_rank=0, external_writer_handle=None, blocking=True
    ):
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        return MessageQueue.create_from_process_group(
            self.cpu_group,
            1 << 22,
            6,
            writer_rank=writer_rank,
            external_writer_handle=external_writer_handle,
            blocking=blocking,
        )

    def create_single_reader_mq_broadcasters(
        self, reader_rank_in_group=0, blocking=False
    ):
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        return MessageQueue.create_from_process_group_single_reader(
            self.cpu_group,
            1 << 22,
            6,
            reader_rank=self.ranks[reader_rank_in_group],
            blocking=blocking,
        )

    @property
    def first_rank(self):
        return self.ranks[0]

    @property
    def last_rank(self):
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        return self.ranks[(self.rank_in_group + 1) % self.world_size]

    @property
    def prev_rank(self):
        return self.ranks[(self.rank_in_group - 1) % self.world_size]

    @contextmanager
    def graph_capture(self, graph_capture_context: GraphCaptureContext | None = None):
        if graph_capture_context is None:
            stream = torch.cuda.Stream()
            graph_capture_context = GraphCaptureContext(stream)
        else:
            stream = graph_capture_context.stream
        maybe_ca_context = nullcontext()
        maybe_aiter_context = nullcontext()
        from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
        from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator

        if self.device_communicator is not None:
            assert isinstance(self.device_communicator, (CudaCommunicator, XpuCommunicator))
            ca_comm = self.device_communicator.ca_comm
            if ca_comm is not None:
                maybe_ca_context = ca_comm.capture()  # type: ignore
            from vllm._aiter_ops import rocm_aiter_ops

            if rocm_aiter_ops.is_enabled():
                aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
                if aiter_ar is not None:
                    maybe_aiter_context = aiter_ar.capture()  # type: ignore
        curr_stream = torch.cuda.current_stream()
        if curr_stream != stream:
            stream.wait_stream(curr_stream)
        with torch.cuda.stream(stream), maybe_ca_context, maybe_aiter_context:
            yield graph_capture_context

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if self.use_custom_op_call:
            return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)
        return self._all_reduce_out_place(input_)

    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_reduce(input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim()
        if self.use_custom_op_call:
            return torch.ops.vllm.all_gather(
                input_, dim, world_size, group_name=self.unique_name
            )
        return self._all_gather_out_place(input_, dim)

    def _all_gather_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_gather(input_, dim)

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_gatherv(input_, dim, sizes)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim()
        if self.use_custom_op_call:
            return torch.ops.vllm.reduce_scatter(
                input_, dim, world_size, group_name=self.unique_name
            )
        return self._reduce_scatter_out_place(input_, dim)

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.reduce_scatterv(input_, dim, sizes)

    def _reduce_scatter_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.reduce_scatter(input_, dim)

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor | None:
        if self.world_size == 1:
            return input_
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.gather(input_, dst, dim)

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        assert src < self.world_size
        if self.world_size == 1:
            return input_
        torch.distributed.broadcast(input_, src=self.ranks[src], group=self.device_group)
        return input_

    def broadcast_object(self, obj: Any | None = None, src: int = 0):
        assert src < self.world_size
        if self.world_size == 1:
            return obj
        if self.mq_broadcaster is not None:
            assert src == 0
            return self.mq_broadcaster.broadcast_object(obj)
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list([obj], src=self.ranks[src], group=self.cpu_group)
            return obj
        recv = [None]
        torch.distributed.broadcast_object_list(recv, src=self.ranks[src], group=self.cpu_group)
        return recv[0]

    def broadcast_object_list(self, obj_list: list[Any], src: int = 0, group: ProcessGroup | None = None):
        assert src < self.world_size
        if self.world_size == 1:
            return obj_list
        torch.distributed.broadcast_object_list(obj_list, src=self.ranks[src], group=self.device_group)
        return obj_list

    def send_object(self, obj: Any, dst: int) -> None:
        assert dst < self.world_size
        assert dst != self.rank_in_group
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)
        size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long, device="cpu")
        torch.distributed.send(size_tensor, dst=self.ranks[dst], group=self.cpu_group)
        torch.distributed.send(object_tensor, dst=self.ranks[dst], group=self.cpu_group)

    def recv_object(self, src: int) -> Any:
        assert src < self.world_size
        assert src != self.rank_in_group
        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
        rank_size = torch.distributed.recv(size_tensor, src=self.ranks[src], group=self.cpu_group)
        object_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device="cpu")  # type: ignore[call-overload]
        rank_object = torch.distributed.recv(object_tensor, src=self.ranks[src], group=self.cpu_group)
        assert rank_object == rank_size
        return pickle.loads(object_tensor.numpy().tobytes())

    def broadcast_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any] | None = None,
        src: int = 0,
        group: ProcessGroup | None = None,
        metadata_group: ProcessGroup | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict
        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size
        rank_in_group = self.rank_in_group
        if rank_in_group == src:
            assert isinstance(tensor_dict, dict)
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    continue
                comm_group = metadata_group if tensor.is_cpu else group
                async_handles.append(torch.distributed.broadcast(tensor, src=self.ranks[src], group=comm_group, async_op=True))
            for handle in async_handles:
                handle.wait()
        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                    if tensor.numel() == 0:
                        tensor_dict[key] = tensor
                        continue
                    comm_group = metadata_group if tensor.is_cpu else group
                    async_handles.append(torch.distributed.broadcast(tensor, src=self.ranks[src], group=comm_group, async_op=True))
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            for handle in async_handles:
                handle.wait()
        return tensor_dict

    def _should_use_all_gather(self, key: str, numel: int, all_gather_group: "GroupCoordinator | None", all_gather_tensors: dict[str, bool] | None) -> bool:
        if all_gather_group is None:
            return False
        use_all_gather = numel % all_gather_group.world_size == 0
        if all_gather_tensors is not None:
            use_all_gather = all_gather_tensors.get(key, use_all_gather)
        return use_all_gather

    def send_tensor_dict(self, tensor_dict: dict[str, torch.Tensor | Any], dst: int | None = None, all_gather_group: "GroupCoordinator | None" = None, all_gather_tensors: dict[str, bool] | None = None) -> dict[str, torch.Tensor | Any] | None:
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict
        handles = self.isend_tensor_dict(tensor_dict, dst=dst, all_gather_group=all_gather_group, all_gather_tensors=all_gather_tensors)
        for handle in handles:
            handle.wait()
        return None

    def isend_tensor_dict(self, tensor_dict: dict[str, torch.Tensor | Any], dst: int | None = None, all_gather_group: "GroupCoordinator | None" = None, all_gather_tensors: dict[str, bool] | None = None) -> list[Handle]:
        if self.world_size <= 1:
            return []
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size
        if self.use_cpu_custom_send_recv:
            if self.device_communicator is None:
                raise ValueError("No device communicator found")
            self.device_communicator.send_tensor_dict(tensor_dict, dst)  # type: ignore
            return []
        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = 0 if all_gather_group is None else all_gather_group.rank_in_group
        group = self.device_group
        metadata_group = self.cpu_group
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        self.send_object(metadata_list, dst=dst)
        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
        assert len(tensor_keys) == len(tensor_list)
        handles: list[Handle] = []
        for key, tensor in zip(tensor_keys, tensor_list):
            if tensor.numel() == 0:
                continue
            if self._should_use_all_gather(key, tensor.numel(), all_gather_group, all_gather_tensors):
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]
            comm_group = metadata_group if tensor.is_cpu else group
            handle = torch.distributed.isend(tensor, dst=self.ranks[dst], group=comm_group)
            if tensor.is_cuda:
                tensor.record_stream(torch.cuda.current_stream(tensor.device))
            handles.append(handle)
        return handles

    def recv_tensor_dict(self, src: int | None = None, all_gather_group: "GroupCoordinator | None" = None, all_gather_tensors: dict[str, bool] | None = None) -> dict[str, torch.Tensor | Any] | None:
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None
        tensor_dict, handles, postprocess = self.irecv_tensor_dict(src=src, all_gather_group=all_gather_group, all_gather_tensors=all_gather_tensors)
        for handle in handles:
            handle.wait()
        for fn in postprocess:
            fn()
        return tensor_dict

    def irecv_tensor_dict(self, src: int | None = None, all_gather_group: "GroupCoordinator | None" = None, all_gather_tensors: dict[str, bool] | None = None) -> tuple[dict[str, torch.Tensor | Any] | None, list[Handle], list[Callable[[], None]]]:
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None, [], []
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size
        if self.use_cpu_custom_send_recv:
            if self.device_communicator is None:
                raise ValueError("No device communicator found")
            return self.device_communicator.recv_tensor_dict(src), [], []  # type: ignore
        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = 0 if all_gather_group is None else all_gather_group.rank_in_group
        group = self.device_group
        metadata_group = self.cpu_group
        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: dict[str, Any] = {}
        handles: list[Handle] = []
        postprocess: list[Callable[[], None]] = []
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                full_tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                if full_tensor.numel() == 0:
                    tensor_dict[key] = full_tensor
                    continue
                if self._should_use_all_gather(key, full_tensor.numel(), all_gather_group, all_gather_tensors):
                    orig_shape = full_tensor.shape
                    slice_tensor = full_tensor.reshape(all_gather_size, -1)[all_gather_rank]
                    comm_group = metadata_group if slice_tensor.is_cpu else group
                    handles.append(torch.distributed.irecv(slice_tensor, src=self.ranks[src], group=comm_group))
                    def _postprocess(key: str = key, slice_tensor: torch.Tensor = slice_tensor, orig_shape: tuple[int, ...] = tuple(orig_shape), all_gather_group=all_gather_group) -> None:
                        assert all_gather_group is not None
                        tensor_dict[key] = all_gather_group.all_gather(slice_tensor, dim=0).reshape(orig_shape)
                    postprocess.append(_postprocess)
                    tensor_dict[key] = slice_tensor
                else:
                    comm_group = metadata_group if full_tensor.is_cpu else group
                    handles.append(torch.distributed.irecv(full_tensor, src=self.ranks[src], group=comm_group))
                    tensor_dict[key] = full_tensor
            else:
                tensor_dict[key] = value
        return tensor_dict, handles, postprocess

    def barrier(self):
        torch.distributed.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        self.device_communicator.send(tensor, dst)

    def recv(self, size: torch.Size, dtype: torch.dtype, src: int | None = None) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.recv(size, dtype, src)

    def destroy(self):
        if hasattr(self, "device_group"):
            torch.distributed.destroy_process_group(self.device_group)
            del self.device_group
        if hasattr(self, "cpu_group"):
            torch.distributed.destroy_process_group(self.cpu_group)
            del self.cpu_group
        if self.device_communicator is not None:
            self.device_communicator.destroy()
        if self.mq_broadcaster is not None:
            self.mq_broadcaster = None

    def prepare_communication_buffer_for_model(self, model: torch.nn.Module):
        if self.device_communicator is not None:
            self.device_communicator.prepare_communication_buffer_for_model(model)

    def dispatch_router_logits(self, hidden_states: torch.Tensor, router_logits: torch.Tensor, is_sequence_parallel: bool = False, extra_tensors: list[torch.Tensor] | None = None):
        if self.device_communicator is not None:
            return self.device_communicator.dispatch_router_logits(hidden_states, router_logits, is_sequence_parallel, extra_tensors)
        return hidden_states, router_logits

    def dispatch(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, is_sequence_parallel: bool = False, extra_tensors: list[torch.Tensor] | None = None):
        if self.device_communicator is not None:
            return self.device_communicator.dispatch(hidden_states, topk_weights, topk_ids, is_sequence_parallel, extra_tensors)
        return hidden_states, topk_weights, topk_ids

    def combine(self, hidden_states, is_sequence_parallel: bool = False) -> torch.Tensor:
        if self.device_communicator is not None:
            return self.device_communicator.combine(hidden_states, is_sequence_parallel)
        return hidden_states


_WORLD: GroupCoordinator | None = None
_INNER_DP_WORLD: GroupCoordinator | None = None
_NODE_COUNT: int | None = None


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def get_inner_dp_world_group() -> GroupCoordinator:
    assert _INNER_DP_WORLD is not None, "inner dp world group is not initialized"
    return _INNER_DP_WORLD


def init_world_group(ranks: list[int], local_rank: int, backend: str) -> GroupCoordinator:
    return GroupCoordinator(group_ranks=[ranks], local_rank=local_rank, torch_distributed_backend=backend, use_device_communicator=False, group_name="world")


def init_model_parallel_group(group_ranks: list[list[int]], local_rank: int, backend: str, use_message_queue_broadcaster: bool = False, group_name: str | None = None, use_device_communicator: bool = True, use_all2all: bool = False) -> GroupCoordinator:
    return GroupCoordinator(group_ranks=group_ranks, local_rank=local_rank, torch_distributed_backend=backend, use_device_communicator=use_device_communicator, use_message_queue_broadcaster=use_message_queue_broadcaster, group_name=group_name, use_all2all=use_all2all)


def _init_stateless_group(group_ranks: list[list[int]], group_name: str, host: str, backend: str, coord_store: Store, use_device_communicator: bool = True, use_all2all: bool = False) -> "StatelessGroupCoordinator":
    from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
    world = get_world_group()
    return StatelessGroupCoordinator(group_ranks=group_ranks, local_rank=world.local_rank, torch_distributed_backend=backend, use_device_communicator=use_device_communicator, group_name=group_name, host=host, coord_store=coord_store, global_rank=world.rank, global_world_size=world.world_size, use_all2all=use_all2all)


def _replace_active_groups(*, world: GroupCoordinator | None, dp: GroupCoordinator | None, ep: GroupCoordinator | None, eplb: GroupCoordinator | None, node_count: int | None) -> None:
    global _WORLD, _DP, _EP, _EPLB, _NODE_COUNT
    for group in (_DP, _EP, _WORLD, _EPLB):
        if group is not None:
            group.destroy()
    _WORLD = world
    _DP = dp
    _EP = ep
    _EPLB = eplb
    _NODE_COUNT = node_count


_TP: GroupCoordinator | None = None
_DCP: GroupCoordinator | None = None
_PP: GroupCoordinator | None = None
_DP: GroupCoordinator | None = None
_EP: GroupCoordinator | None = None
_EPLB: GroupCoordinator | None = None
_PCP: GroupCoordinator | None = None


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None
    return _TP

def get_dcp_group() -> GroupCoordinator:
    assert _DCP is not None
    return _DCP

def get_pp_group() -> GroupCoordinator:
    assert _PP is not None
    return _PP

def get_dp_group() -> GroupCoordinator:
    assert _DP is not None
    return _DP

def get_ep_group() -> GroupCoordinator:
    assert _EP is not None
    return _EP

def get_eplb_group() -> GroupCoordinator:
    assert _EPLB is not None
    return _EPLB

def get_pcp_group() -> GroupCoordinator:
    assert _PCP is not None
    return _PCP


@contextmanager
def graph_capture(device: torch.device, graph_capture_context: GraphCaptureContext | None = None):
    context = graph_capture_context or GraphCaptureContext(torch.cuda.Stream(device=device))
    with get_tp_group().graph_capture(context), get_pp_group().graph_capture(context):
        yield context


logger = init_logger(__name__)
_ENABLE_CUSTOM_ALL_REDUCE = True


def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable


def _init_process_group_for_split_group(*, backend: str, distributed_init_method: str, world_size: int, rank: int, local_rank: int, timeout: timedelta | None) -> None:
    if torch.accelerator.is_available() and backend != "gloo":
        init_backend = "cpu:gloo,cuda:nccl"
        from vllm.platforms import current_platform
        visible_device_index = current_platform.logical_device_id_to_visible_device_id(local_rank)
        device_id: torch.device | None = torch.device(f"cuda:{visible_device_index}")
    else:
        init_backend = "gloo"
        device_id = None
    torch.distributed.init_process_group(backend=init_backend, init_method=distributed_init_method, world_size=world_size, rank=rank, timeout=timeout, device_id=device_id)


def _validate_default_pg_for_split_group() -> None:
    default_pg = torch.distributed.distributed_c10d._get_default_group()
    assert default_pg.bound_device_id is not None
    try:
        default_pg._get_backend(torch.device("cpu"))
    except RuntimeError as e:
        raise RuntimeError("External launcher initialized the default process group without a CPU (gloo) backend.") from e


def _init_elastic_ep_world(config, local_rank: int, backend: str, rank: int, world_size: int) -> None:
    from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
    global _WORLD, _NODE_COUNT
    assert _WORLD is None
    parallel_config = config.parallel_config
    global_rank = parallel_config.data_parallel_rank * world_size + rank
    global_world_size = parallel_config.world_size_across_dp
    all_ranks = list(range(global_world_size))
    group_ranks = [all_ranks[i : i + 1] for i in range(global_world_size)]
    if global_rank in all_ranks:
        group_ranks = [all_ranks]
    coord_store = get_cached_tcp_store_client(parallel_config.data_parallel_master_ip, parallel_config._coord_store_port)
    world = StatelessGroupCoordinator(group_ranks=group_ranks, local_rank=local_rank, torch_distributed_backend=backend, use_device_communicator=False, group_name="world", host=parallel_config.data_parallel_master_ip, coord_store=coord_store, global_rank=global_rank, global_world_size=global_world_size)
    assert parallel_config.nnodes_within_dp == 1
    _NODE_COUNT = _node_count(world.tcp_store_group)
    _WORLD = world


def init_distributed_environment(world_size: int = -1, rank: int = -1, distributed_init_method: str = "env://", local_rank: int = -1, backend: str = "nccl", timeout: timedelta | None = None):
    logger.debug("world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s", world_size, rank, local_rank, distributed_init_method, backend)
    from vllm.config import get_current_vllm_config_or_none
    config = get_current_vllm_config_or_none()
    enable_elastic_ep = config is not None and config.parallel_config.enable_elastic_ep
    if config is not None and config.parallel_config.distributed_executor_backend != "external_launcher" and (config.parallel_config.nnodes > 1 or config.parallel_config.data_parallel_size > 1) and not enable_elastic_ep:
        parallel_config = config.parallel_config
        rank = parallel_config.data_parallel_rank * world_size + rank
        world_size = parallel_config.world_size_across_dp
        if parallel_config.nnodes > 1:
            ip = parallel_config.master_addr
            port = parallel_config.master_port
            distributed_init_method = get_distributed_init_method(ip, port)
        else:
            ip = parallel_config.data_parallel_master_ip
            port = parallel_config.get_next_dp_init_port()
            distributed_init_method = get_distributed_init_method(ip, port)
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None
        if not torch.distributed.is_backend_available(backend):
            assert torch.distributed.is_gloo_available()
            backend = "gloo"
        if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
            if local_rank == -1:
                local_rank = int(envs.LOCAL_RANK) if distributed_init_method == "env://" else rank
            _init_process_group_for_split_group(backend=backend, distributed_init_method=distributed_init_method, world_size=world_size, rank=rank, local_rank=local_rank, timeout=timeout)
        else:
            torch.distributed.init_process_group(backend=backend, init_method=distributed_init_method, world_size=world_size, rank=rank, timeout=timeout)
        if enable_elastic_ep:
            tp_pp_cpu_group = torch.distributed.new_group(backend="gloo", timeout=timeout)
            if _node_count(tp_pp_cpu_group) > 1:
                raise RuntimeError("Elastic EP is not yet supported with multi-node TP/PP")
    if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP and torch.accelerator.is_available():
        _validate_default_pg_for_split_group()
    if local_rank == -1:
        local_rank = envs.LOCAL_RANK if distributed_init_method == "env://" else rank
    global _WORLD, _NODE_COUNT, _INNER_DP_WORLD
    if enable_elastic_ep:
        _init_elastic_ep_world(config, local_rank, backend, rank, world_size)
        return
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
        if config is not None and config.parallel_config.nnodes > 1:
            _NODE_COUNT = config.parallel_config.nnodes
        else:
            _NODE_COUNT = _node_count(_WORLD.cpu_group)
    else:
        assert _WORLD.world_size == torch.distributed.get_world_size()
    if config is not None and config.parallel_config.nnodes_within_dp > 1:
        if parallel_config.data_parallel_size > 1:
            world_size_inner_dp = parallel_config.world_size
            group_ranks = [[dp_rank * world_size_inner_dp + i for i in range(world_size_inner_dp)] for dp_rank in range(parallel_config.data_parallel_size)]
            _INNER_DP_WORLD = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_message_queue_broadcaster=True, group_name="inner_dp_world", use_device_communicator=False)
        else:
            _INNER_DP_WORLD = _WORLD


def initialize_model_parallel(tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1, prefill_context_model_parallel_size: int = 1, decode_context_model_parallel_size: int | None = 1, backend: str | None = None) -> None:
    assert torch.distributed.is_initialized()
    from vllm.config import get_current_vllm_config
    config = get_current_vllm_config()
    data_parallel_size = config.parallel_config.data_parallel_size
    enable_elastic_ep = config.parallel_config.enable_elastic_ep
    parallel_config = config.parallel_config
    coord_store: Store | None = None
    if enable_elastic_ep:
        coord_store = get_cached_tcp_store_client(parallel_config.data_parallel_master_ip, parallel_config._coord_store_port)
        world_size = get_world_group().world_size
        rank = get_world_group().rank
        backend = backend or "nccl"
        tp_pp_pcp_size = tensor_model_parallel_size * pipeline_model_parallel_size * prefill_context_model_parallel_size
        local_all_ranks = torch.arange(tp_pp_pcp_size).reshape(pipeline_model_parallel_size, prefill_context_model_parallel_size, tensor_model_parallel_size)
    else:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    all_ranks = torch.arange(world_size).reshape(-1, data_parallel_size, pipeline_model_parallel_size, prefill_context_model_parallel_size, tensor_model_parallel_size)

    global _TP
    assert _TP is None
    group_ranks = [x.tolist() for x in all_ranks.view(-1, tensor_model_parallel_size).unbind(0)]
    if enable_elastic_ep:
        group_ranks = [x.tolist() for x in local_all_ranks.view(-1, tensor_model_parallel_size).unbind(0)]
    _TP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_message_queue_broadcaster=True, group_name="tp")

    global _DCP
    assert _DCP is None
    dcp_size = decode_context_model_parallel_size or 1
    dcp_ranks = local_all_ranks if enable_elastic_ep else all_ranks
    if dcp_size > 1:
        dcp_ranks = dcp_ranks.transpose(-1, -2)
    group_ranks = [x.tolist() for x in dcp_ranks.reshape(-1, dcp_size).unbind(0)]
    _DCP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_message_queue_broadcaster=True, group_name="dcp")

    global _PCP
    assert _PCP is None
    group_ranks = [x.tolist() for x in all_ranks.transpose(3, 4).reshape(-1, prefill_context_model_parallel_size).unbind(0)]
    if enable_elastic_ep:
        group_ranks = [x.tolist() for x in local_all_ranks.transpose(1, 2).reshape(-1, prefill_context_model_parallel_size).unbind(0)]
    from vllm.v1.worker.gpu.pcp_runahead_config import get_pcp_process_group_order
    pcp_order = get_pcp_process_group_order(config.additional_config, prefill_context_model_parallel_size)
    if pcp_order != tuple(range(prefill_context_model_parallel_size)):
        group_ranks = [[ranks[physical_rank] for physical_rank in pcp_order] for ranks in group_ranks]
    _PCP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="pcp")

    global _PP
    assert _PP is None
    group_ranks = [x.tolist() for x in all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size).unbind(0)]
    if enable_elastic_ep:
        group_ranks = [x.tolist() for x in local_all_ranks.transpose(0, 2).reshape(-1, pipeline_model_parallel_size).unbind(0)]
    _PP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="pp")

    global _DP
    assert _DP is None
    group_ranks = [x.tolist() for x in all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)]
    if enable_elastic_ep:
        _DP = _init_stateless_group(group_ranks, "dp", parallel_config.data_parallel_master_ip, backend, coord_store=coord_store)
    else:
        _DP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="dp")

    global _EP
    assert _EP is None
    if config.model_config is None or config.model_config.is_moe:
        group_ranks = [x.tolist() for x in all_ranks.transpose(1, 2).reshape(-1, data_parallel_size * prefill_context_model_parallel_size * tensor_model_parallel_size).unbind(0)]
        use_all2all = parallel_config.use_all2all
        if enable_elastic_ep:
            _EP = _init_stateless_group(group_ranks, "ep", parallel_config.data_parallel_master_ip, backend, coord_store=coord_store, use_all2all=use_all2all)
        else:
            _EP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="ep", use_all2all=use_all2all)
        global _EPLB
        assert _EPLB is None
        if config.parallel_config.enable_eplb:
            if enable_elastic_ep:
                _EPLB = _init_stateless_group(group_ranks, "eplb", parallel_config.data_parallel_master_ip, backend, coord_store=coord_store)
            else:
                _EPLB = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="eplb")

    logger.info_once("rank %s in world size %s is assigned as DP rank %s, PP rank %s, PCP rank %s, TP rank %s, EP rank %s, EPLB rank %s", rank, world_size, _DP.rank_in_group, _PP.rank_in_group, _PCP.rank_in_group, _TP.rank_in_group, _EP.rank_in_group if _EP is not None else "N/A", _EPLB.rank_in_group if _EPLB is not None else "N/A")


def ensure_model_parallel_initialized(tensor_model_parallel_size: int, pipeline_model_parallel_size: int, prefill_context_model_parallel_size: int = 1, decode_context_model_parallel_size: int | None = 1, backend: str | None = None) -> None:
    world_group = get_world_group()
    if hasattr(world_group, "backend"):
        backend = backend or world_group.backend
    else:
        backend = backend or torch.distributed.get_backend(world_group.device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, prefill_context_model_parallel_size, decode_context_model_parallel_size, backend)
        return
    assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size
    assert get_pp_group().world_size == pipeline_model_parallel_size
    assert get_pcp_group().world_size == prefill_context_model_parallel_size
    assert get_dcp_group().world_size == (decode_context_model_parallel_size or 1)


def prepare_communication_buffer_for_model(model: torch.nn.Module):
    for group in (_TP, _PCP, _PP, _DP, _EP, _EPLB):
        if group is not None:
            group.prepare_communication_buffer_for_model(model)


def checkpoint_prepare_distributed_state() -> None:
    torch.accelerator.synchronize()
    _apply_to_device_comms(lambda comm: comm.checkpoint_prepare())
    torch.accelerator.synchronize()


def checkpoint_restore_distributed_state() -> None:
    torch.accelerator.synchronize()
    _apply_to_device_comms(lambda comm: comm.checkpoint_restore())
    torch.accelerator.synchronize()


def model_parallel_is_initialized():
    return _TP is not None and _PP is not None


_TP_STATE_PATCHED = False


def get_tensor_model_parallel_world_size() -> int:
    return get_tp_group().world_size


def get_tensor_model_parallel_rank() -> int:
    return get_tp_group().rank_in_group


def get_node_count() -> int:
    assert _NODE_COUNT is not None
    return _NODE_COUNT


def destroy_model_parallel():
    global _TP, _DCP, _PCP, _PP, _DP, _EP, _EPLB
    for group in (_TP, _DCP, _PCP, _PP, _DP, _EP, _EPLB):
        if group:
            group.destroy()
    _TP = _DCP = _PCP = _PP = _DP = _EP = _EPLB = None


def destroy_distributed_environment():
    global _WORLD, _NODE_COUNT
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    _NODE_COUNT = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    logger.debug("[shutdown] Distributed: cleanup start shutdown_ray=%s", shutdown_ray)
    envs.disable_envs_cache()
    from vllm.platforms import current_platform
    if current_platform.is_rocm():
        from vllm._aiter_ops import rocm_aiter_ops
        rocm_aiter_ops.refresh_env_variables()
    gc.unfreeze()
    destroy_model_parallel()
    destroy_distributed_environment()
    if shutdown_ray:
        import ray
        ray.shutdown()
    gc.collect()
    from vllm.platforms import current_platform
    if not current_platform.is_cpu():
        torch.accelerator.empty_cache()
        try:
            torch._C._host_emptyCache()
        except AttributeError:
            logger.warning("torch._C._host_emptyCache() only available in Pytorch >=2.5")
    logger.debug_once("[shutdown] Distributed: cleanup complete")


def in_the_same_node_as(pg: ProcessGroup | StatelessProcessGroup, source_rank: int = 0) -> list[bool]:
    if isinstance(pg, ProcessGroup):
        assert torch.distributed.get_backend(pg) != torch.distributed.Backend.NCCL
        rank = torch.distributed.get_rank(group=pg)
        world_size = torch.distributed.get_world_size(group=pg)
        ranks = torch.distributed.get_process_group_ranks(pg)
    else:
        rank = pg.rank
        world_size = pg.world_size
        ranks = list(range(world_size))
    is_in_the_same_node = torch.tensor([0] * world_size, dtype=torch.int32, device="cpu")
    magic_message = b"magic_message"
    shm = None
    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                shm = shared_memory.SharedMemory(create=True, size=128)
                assert shm.buf is not None
                shm.buf[: len(magic_message)] = magic_message
                if isinstance(pg, ProcessGroup):
                    torch.distributed.broadcast_object_list([shm.name], src=ranks[source_rank], group=pg)
                else:
                    pg.broadcast_obj(shm.name, src=source_rank)
                is_in_the_same_node[rank] = 1
            else:
                if isinstance(pg, ProcessGroup):
                    recv = [None]
                    torch.distributed.broadcast_object_list(recv, src=ranks[source_rank], group=pg)
                    name = recv[0]
                else:
                    name = pg.broadcast_obj(None, src=source_rank)
                with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
                    shm = shared_memory.SharedMemory(name=name)
                assert shm.buf is not None
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()
    if isinstance(pg, ProcessGroup):
        torch.distributed.barrier(group=pg)
    else:
        pg.barrier()
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()
    if isinstance(pg, ProcessGroup):
        torch.distributed.all_reduce(is_in_the_same_node, group=pg)
        aggregated_data = is_in_the_same_node
    else:
        aggregated_data = torch.zeros_like(is_in_the_same_node)
        for i in range(world_size):
            rank_data = pg.broadcast_obj(is_in_the_same_node, src=i)
            aggregated_data += rank_data
    return [x == 1 for x in aggregated_data.tolist()]


def is_global_first_rank() -> bool:
    try:
        global _WORLD
        if _WORLD is not None:
            return _WORLD.is_first_rank
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0
    except Exception:
        return True


def is_local_first_rank() -> bool:
    try:
        global _WORLD
        if _WORLD is not None:
            return _WORLD.local_rank == 0
        if not torch.distributed.is_initialized():
            return True
        try:
            return int(envs.LOCAL_RANK) == 0  # type: ignore[arg-type]
        except Exception:
            return torch.distributed.get_rank() == 0
    except Exception:
        return True


def _node_count(pg: ProcessGroup | StatelessProcessGroup) -> int:
    if isinstance(pg, ProcessGroup):
        world_size = torch.distributed.get_world_size(group=pg)
    else:
        world_size = pg.world_size
    if world_size == 1:
        return 1
    node_assignment = [0] * world_size
    next_node_id = 0
    for current_rank in range(world_size):
        if node_assignment[current_rank] != 0:
            continue
        next_node_id += 1
        node_assignment[current_rank] = next_node_id
        same_node_flags = in_the_same_node_as(pg, current_rank)
        for other_rank, is_same_node in enumerate(same_node_flags):
            if is_same_node and node_assignment[other_rank] == 0:
                node_assignment[other_rank] = next_node_id
    return next_node_id
