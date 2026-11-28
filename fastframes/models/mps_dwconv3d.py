"""Metal causal 3x3x3 depthwise convolution for PixelDecoder and StreamingDecoder on MPS."""
import os
from functools import lru_cache

import torch

__all__ = ["try_causal_dwconv3d"]

_SOURCE = """
#include <metal_stdlib>
using namespace metal;

#define DEFINE_CAUSAL_DWCONV3D(NAME, T_IO)                                                 \\
kernel void NAME(                                                                          \\
    device T_IO* out [[buffer(0)]],                                                        \\
    device const T_IO* x [[buffer(1)]],                                                    \\
    device const T_IO* hist [[buffer(2)]],                                                 \\
    device const T_IO* weight [[buffer(3)]],                                               \\
    device const T_IO* bias [[buffer(4)]],                                                 \\
    constant ulong4& dims [[buffer(5)]],   /* C, T_out, H, W */                            \\
    constant ulong2& flags [[buffer(6)]],  /* has_hist, has_bias */                        \\
    uint3 gid [[thread_position_in_grid]])                                                 \\
{                                                                                          \\
    _Pragma("METAL fp math_mode(safe)")                                                    \\
    _Pragma("clang fp contract(off)")                                                      \\
    uint C = uint(dims.x), T_out = uint(dims.y), H = uint(dims.z), W = uint(dims.w);       \\
    uint HW = H * W;                                                                       \\
    uint p = gid.x;                                                                        \\
    if (p >= HW) return;                                                                   \\
    uint t = gid.y;                                                                        \\
    uint bc = gid.z;                                                                       \\
    uint c = bc % C;                                                                       \\
    int h = int(p / W), w = int(p % W);                                                    \\
    device const T_IO* x_bc = x + ulong(bc) * T_out * HW;                                  \\
    device const T_IO* hist_bc = hist + ulong(bc) * 2 * HW;                                \\
    device const T_IO* w_c = weight + ulong(c) * 27;                                       \\
    float acc = flags.y ? float(bias[c]) : 0.0f;                                           \\
    for (int kt = 0; kt < 3; ++kt) {                                                       \\
        int src_t = int(t) + kt - 2;                                                       \\
        for (int kh = -1; kh <= 1; ++kh) {                                                 \\
            int ih = h + kh;                                                               \\
            for (int kw = -1; kw <= 1; ++kw) {                                             \\
                int iw = w + kw;                                                           \\
                float val = 0.0f;                                                          \\
                if (ih >= 0 && ih < int(H) && iw >= 0 && iw < int(W)) {                    \\
                    ulong sp = ulong(uint(ih)) * W + uint(iw);                             \\
                    if (src_t >= 0) {                                                      \\
                        val = float(x_bc[ulong(uint(src_t)) * HW + sp]);                   \\
                    } else if (flags.x) {                                                  \\
                        val = float(hist_bc[ulong(uint(src_t + 2)) * HW + sp]);            \\
                    }                                                                      \\
                }                                                                          \\
                acc += val * float(w_c[(kt * 3 + (kh + 1)) * 3 + (kw + 1)]);               \\
            }                                                                              \\
        }                                                                                  \\
    }                                                                                      \\
    out[(ulong(bc) * T_out + t) * HW + p] = T_IO(acc);                                     \\
}

DEFINE_CAUSAL_DWCONV3D(causal_dwconv3d_bf16, bfloat)
DEFINE_CAUSAL_DWCONV3D(causal_dwconv3d_fp32, float)
"""


@lru_cache(maxsize=1)
def _lib():
    try:
        return torch.mps.compile_shader(_SOURCE)
    except (AttributeError, RuntimeError):
        return None


def try_causal_dwconv3d(x, weight, bias, stride, dilation, groups, pad, *, history=None):
    """Run causal 3x3x3 depthwise Conv3d on MPS; return None to fall back to native ``F.conv3d``."""
    if (os.environ.get("FASTFRAMES_MPS_DECODER_CONV") == "native"
            or os.environ.get("FASTFRAMES_MPS_CONV3D") == "native"):
        return None
    if (x.device.type != "mps" or x.ndim != 5 or weight.device != x.device
            or tuple(stride) != (1, 1, 1) or tuple(dilation) != (1, 1, 1)
            or tuple(pad) != (1, 1, 1, 1, 2, 0)):
        return None
    b, c, t_out, h, w = x.shape
    if groups != c or c == 0 or weight.shape != (c, 1, 3, 3, 3):
        return None
    if bias is not None and (bias.device != x.device or bias.shape != (c,)):
        return None
    if history is not None and (history.device != x.device or history.shape != (b, c, 2, h, w)):
        return None
    if torch.is_grad_enabled() and (
        x.requires_grad or weight.requires_grad
        or (bias is not None and bias.requires_grad)
        or (history is not None and history.requires_grad)
    ):
        return None
    dtype = torch.get_autocast_dtype("mps") if torch.is_autocast_enabled("mps") else x.dtype
    if dtype not in (torch.bfloat16, torch.float32):
        return None
    hw = h * w
    if b * c == 0 or t_out == 0 or hw == 0:
        return torch.empty((b, c, t_out, h, w), dtype=dtype, device=x.device)
    if hw >= 2**31 or t_out >= 2**31 or b * c >= 2**31:
        return None
    lib = _lib()
    if lib is None:
        return None
    x_c = x.to(dtype).contiguous()
    w_c = weight.to(dtype).contiguous()
    has_hist = 1 if history is not None else 0
    hist_c = history.to(dtype).contiguous() if history is not None else x_c[:1, :1, :1, :1, :1]
    has_bias = 1 if bias is not None else 0
    bias_c = bias.to(dtype).contiguous() if bias is not None else w_c.flatten()[:1]
    out = torch.empty((b, c, t_out, h, w), dtype=dtype, device=x.device)
    fn = lib.causal_dwconv3d_bf16 if dtype == torch.bfloat16 else lib.causal_dwconv3d_fp32
    fn(out, x_c, hist_c, w_c, bias_c, (c, t_out, h, w), (has_hist, has_bias), threads=(hw, t_out, b * c))
    return out
