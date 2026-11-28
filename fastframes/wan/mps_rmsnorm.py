"""Single-dispatch Metal RMSNorm for BF16 Wan and LingBot inference tensors on MPS."""
import os
from functools import lru_cache

import torch

__all__ = ["try_rms_norm"]

_SOURCE = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

kernel void rms_norm_bf16(
    device bfloat* out [[buffer(0)]],
    device const bfloat* x [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    constant ulong2& dims [[buffer(3)]],    // D, row_stride
    constant float& eps [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
{
    #pragma METAL fp math_mode(safe)
    #pragma clang fp contract(off)
    threadgroup float shared[8];
    uint D = uint(dims.x);
    device const bfloat* row = x + ulong(tgid) * dims.y;
    float sum_sq = 0.0f;
    for (uint i = tid; i < D; i += lsize) {
        float v = float(row[i]);
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) shared[simd_id] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float total = (simd_lane < (lsize >> 5)) ? shared[simd_lane] : 0.0f;
        total = simd_sum(total);
        if (simd_lane == 0) shared[0] = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = shared[0];
    device bfloat* out_row = out + ulong(tgid) * D;
    // Match PyTorch's two-step BF16 rounding: round(x * rsqrt) -> bf16, then * weight -> bf16.
    for (uint i = tid; i < D; i += lsize) {
        bfloat normed = bfloat(float(row[i]) * inv);
        out_row[i] = bfloat(float(normed) * float(weight[i]));
    }
}
"""


@lru_cache(maxsize=1)
def _kernel():
    try:
        return torch.mps.compile_shader(_SOURCE).rms_norm_bf16
    except (AttributeError, RuntimeError):
        return None


def _rows_and_stride(x, dim):
    if x.ndim < 1 or x.shape[-1] != dim or x.is_neg():
        return None
    if x.is_contiguous():
        return x.numel() // dim, dim
    if x.stride(-1) != 1:
        return None
    if x.ndim == 2 and x.stride(0) >= dim:
        return x.shape[0], x.stride(0)
    if x.ndim == 3 and x.shape[0] == 1 and x.stride(1) >= dim:
        return x.shape[1], x.stride(1)
    return None


def try_rms_norm(x, weight, eps):
    """Run fused BF16 RMSNorm on MPS (including strided QKV chunk slices); return None on fallback."""
    if os.environ.get("FASTFRAMES_MPS_RMSNORM") == "eager":
        return None
    if (x.device.type != "mps" or x.dtype != torch.bfloat16
            or weight.device != x.device or weight.dtype != torch.bfloat16
            or weight.ndim != 1 or not weight.is_contiguous() or weight.is_neg()
            or (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))):
        return None
    dim = weight.shape[0]
    if dim == 0 or dim >= 2**31:
        return None
    layout = _rows_and_stride(x, dim)
    if layout is None:
        return None
    rows, row_stride = layout
    if rows == 0:
        return torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    if rows >= 2**23 or row_stride >= 2**31:
        return None
    kernel = _kernel()
    if kernel is None:
        return None
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    kernel(out, x, weight, (dim, row_stride), float(eps), threads=(256 * rows, 1, 1), group_size=(256, 1, 1))
    return out
