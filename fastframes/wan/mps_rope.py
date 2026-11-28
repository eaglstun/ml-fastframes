"""Single-dispatch real RoPE for contiguous BF16 inference tensors on MPS."""
import os
from functools import lru_cache

import torch

__all__ = ["try_rope"]

_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void rotate(device bfloat* out, device const bfloat* x, device const float* table,
                   uint3 pos [[thread_position_in_grid]], uint3 size [[threads_per_grid]]) {
    // Eager rounds each FP32 multiply before the add/subtract; an FMA would round only once.
    #pragma METAL fp math_mode(safe)
    #pragma clang fp contract(off)
    // One thread rotates one pair. All heads and batches share the same table.
    uint i = 2 * ((pos.z * size.y + pos.y) * size.x + pos.x);
    uint j = 2 * (pos.y * 64 + pos.x % 64);
    float r = float(x[i]), v = float(x[i + 1]);
    float c = table[j], s = table[j + 1];
    out[i] = bfloat(r * c - v * s);
    out[i + 1] = bfloat(r * s + v * c);
}
"""


@lru_cache(maxsize=1)
def _kernel():
    try:
        return torch.mps.compile_shader(_SOURCE).rotate
    except (AttributeError, RuntimeError):
        return None


def try_rope(value, table, *, kernel_fn=None):
    """Return None for layouts, precisions or PyTorch versions that need eager rotation."""
    if os.environ.get("FASTFRAMES_MPS_ROPE") == "eager":
        return None
    if (value.device.type != "mps" or value.dtype != torch.bfloat16 or value.ndim != 4 or value.shape[-1] != 128
            or table.dtype != torch.float32 or table.device != value.device
            or table.shape != (value.shape[1], 1, 64, 2)
            or not value.is_contiguous() or not table.is_contiguous()
            or value.is_neg() or table.is_neg() or not 0 < value.numel() < 2**32
            or (torch.is_grad_enabled() and (value.requires_grad or table.requires_grad))):
        return None
    kernel = (_kernel if kernel_fn is None else kernel_fn)()
    if kernel is None:
        return None
    batch, tokens, heads, _ = value.shape
    out = torch.empty_like(value)
    kernel(out, value, table, threads=(heads * 64, tokens, batch))
    return out
