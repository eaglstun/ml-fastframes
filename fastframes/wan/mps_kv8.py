"""Metal quantize and dequantize kernels for int8 KV caches on MPS."""
import os
from functools import lru_cache

import torch

__all__ = ["try_dequantize_kv", "try_quantize_kv"]

_SOURCE = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

kernel void dequant_kv_bf16(
    device bfloat* out [[buffer(0)]],
    device const char* data [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    constant ulong4& strides [[buffer(3)]],  // nh_d4, out_b_stride, in_b_stride, scale_b_stride
    uint gid [[thread_position_in_grid]])
{
    #pragma METAL fp math_mode(safe)
    #pragma clang fp contract(off)
    // Matches (data.to(float16) * scale.to(float16)).to(bfloat16) bit for bit.
    ulong nh_d4 = strides.x;
    ulong b = ulong(gid) / nh_d4;
    ulong vec = ulong(gid) % nh_d4;
    ulong row = vec >> 5;  // 32 char4 vectors = 128 channels per head
    half s = half(scale[b * strides.w + row]);
    char4 q = *(device const char4*)(data + b * strides.z + vec * 4);
    *(device bfloat4*)(out + b * strides.y + vec * 4) = bfloat4(half4(q) * s);
}

#define DEFINE_QUANT_KV(NAME, T_IN)                                                          \\
kernel void NAME(                                                                            \\
    device char* out_data [[buffer(0)]],                                                     \\
    device float* out_scale [[buffer(1)]],                                                   \\
    device const T_IN* x [[buffer(2)]],                                                      \\
    constant ulong4& strides [[buffer(3)]],  /* nh, data_b_stride, scale_b_stride, x_b_stride */ \\
    uint tgid [[threadgroup_position_in_grid]],                                              \\
    uint lane [[thread_index_in_simdgroup]])                                                 \\
{                                                                                            \\
    _Pragma("METAL fp math_mode(safe)")                                                      \\
    _Pragma("clang fp contract(off)")                                                        \\
    /* One 32-lane SIMD-group quantizes one 128-channel (token, head) row (4 elems/lane). */ \\
    ulong nh = strides.x;                                                                    \\
    ulong b = ulong(tgid) / nh;                                                              \\
    ulong row = ulong(tgid) % nh;                                                            \\
    device const T_IN* src = x + b * strides.w + row * 128 + ulong(lane) * 4;                \\
    float4 vf = float4(float(src[0]), float(src[1]), float(src[2]), float(src[3]));          \\
    float amax = fmax(fmax(fabs(vf.x), fabs(vf.y)), fmax(fabs(vf.z), fabs(vf.w)));           \\
    amax = simd_max(amax);                                                                   \\
    float sc = fmax(amax, 1e-12f) / 127.0f;                                                  \\
    if (lane == 0) out_scale[b * strides.z + row] = sc;                                      \\
    float4 qf = clamp(rint(vf / sc), -127.0f, 127.0f);                                       \\
    *(device char4*)(out_data + b * strides.y + row * 128 + ulong(lane) * 4) = char4(int4(qf)); \\
}

DEFINE_QUANT_KV(quant_kv_bf16, bfloat)
DEFINE_QUANT_KV(quant_kv_fp32, float)
"""


@lru_cache(maxsize=1)
def _lib():
    try:
        return torch.mps.compile_shader(_SOURCE)
    except (AttributeError, RuntimeError):
        return None


def _row_contiguous_batch(t, head_dim):
    """True if ``t`` is ``[B, L, H, head_dim]`` with contiguous ``[L, H, head_dim]`` slices per batch."""
    if t.ndim != 4 or t.shape[-1] != head_dim or t.is_neg():
        return False
    _, length, heads, _ = t.shape
    s1, s2, s3 = t.stride(1), t.stride(2), t.stride(3)
    if s3 != 1:
        return False
    if head_dim > 1 and s2 != head_dim:
        return False
    if heads > 1 and s1 != heads * head_dim:
        return False
    return t.shape[0] == 1 or t.stride(0) >= length * heads * head_dim


def try_quantize_kv(data_slice, scale_slice, x):
    """Quantize ``x`` ``[B, L, H, 128]`` into ``(data_slice, scale_slice)`` on MPS; returns bool."""
    if os.environ.get("FASTFRAMES_MPS_KV8") == "eager":
        return False
    if (x.device.type != "mps" or x.dtype not in (torch.bfloat16, torch.float32)
            or data_slice.device != x.device or scale_slice.device != x.device
            or data_slice.dtype != torch.int8 or scale_slice.dtype != torch.float32
            or x.shape != data_slice.shape or scale_slice.shape != (*x.shape[:-1], 1)
            or not _row_contiguous_batch(x, 128) or not _row_contiguous_batch(data_slice, 128)
            or not _row_contiguous_batch(scale_slice, 1)
            or data_slice.storage_offset() % 4 != 0
            or (torch.is_grad_enabled() and x.requires_grad)):
        return False
    b, length, heads, _ = x.shape
    rows = b * length * heads
    if rows == 0:
        return True
    if rows >= 2**31:
        return False
    lib = _lib()
    if lib is None:
        return False
    fn = lib.quant_kv_bf16 if x.dtype == torch.bfloat16 else lib.quant_kv_fp32
    strides = (length * heads, data_slice.stride(0), scale_slice.stride(0), x.stride(0))
    fn(data_slice, scale_slice, x, strides, threads=(32 * rows, 1, 1), group_size=(32, 1, 1))
    return True


def try_dequantize_kv(data_slice, scale_slice, out=None):
    """Dequantize ``(data_slice, scale_slice)`` ``[B, L, H, 128]`` to BF16 (into ``out`` if given)."""
    if os.environ.get("FASTFRAMES_MPS_KV8") == "eager":
        return None
    if (data_slice.device.type != "mps" or scale_slice.device != data_slice.device
            or data_slice.dtype != torch.int8 or scale_slice.dtype != torch.float32
            or scale_slice.shape != (*data_slice.shape[:-1], 1)
            or not _row_contiguous_batch(data_slice, 128) or not _row_contiguous_batch(scale_slice, 1)
            or data_slice.storage_offset() % 4 != 0):
        return None
    if out is not None:
        if (out.device != data_slice.device or out.dtype != torch.bfloat16 or out.shape != data_slice.shape
                or not _row_contiguous_batch(out, 128) or out.storage_offset() % 4 != 0):
            return None
    b, length, heads, _ = data_slice.shape
    vecs = b * length * heads * 32
    if vecs >= 2**32:
        return None
    lib = _lib()
    if lib is None:
        return None
    if out is None:
        out = torch.empty(data_slice.shape, dtype=torch.bfloat16, device=data_slice.device)
    if vecs == 0:
        return out
    strides = (length * heads * 32, out.stride(0), data_slice.stride(0), scale_slice.stride(0))
    lib.dequant_kv_bf16(out, data_slice, scale_slice, strides, threads=(vecs, 1, 1))
    return out
