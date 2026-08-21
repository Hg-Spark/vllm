# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from functools import wraps

from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.utils.torch_utils import _resolve_layer_name
from vllm.v1.attention.ops.pcp_runahead import get_pcp_runahead_runtime


def maybe_transfer_kv_layer(func: Callable) -> Callable:
    """Handle KV movement immediately before and after an attention layer.

    The PCP page-pull hook runs on attention entry. unified_attention's dummy
    dependency guarantees that the native KV-cache update custom op has already
    executed, so READY is published after the cache write without patching an
    attention backend.
    """
    # Import at runtime to avoid circular dependency
    from vllm.model_executor.layers.attention.attention import get_attention_context

    # Inspect the signature ONCE when the decorator is applied.
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Find the index of 'layer_name' parameter.
    try:
        layer_name_index = param_names.index("layer_name")
    except ValueError as e:
        raise TypeError(
            f"Function {func.__name__} must have a 'layer_name' parameter"
        ) from e

    @wraps(func)
    def wrapper(*args, **kwargs):
        runtime = get_pcp_runahead_runtime()
        page_pull = runtime is not None and runtime.transport == "page_pull"
        kv_transfer = has_kv_transfer_group() and is_v1_kv_transfer_group()
        if not page_pull and not kv_transfer:
            return func(*args, **kwargs)

        layer_name = _resolve_layer_name(args[layer_name_index])

        # Extract attention context (metadata, layer, kv_cache, layer_slot_mapping)
        attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)

        if page_pull:
            assert runtime is not None
            runtime.page_pull_after_cache_write(kv_cache)

        if not kv_transfer:
            return func(*args, **kwargs)

        connector = get_kv_transfer_group()
        if attn_metadata is None or not connector.has_connector_metadata():
            return func(*args, **kwargs)

        # Wait for KV layer on entry
        connector.wait_for_layer_load(layer_name)

        # Execute the function
        result = func(*args, **kwargs)

        # Save KV cache layer on exit
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)

        return result

    return wrapper
