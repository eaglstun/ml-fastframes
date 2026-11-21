#!/usr/bin/env python3
"""LingBot DiT block 0 in MLX on the inputs scripts/lingbot_block_capture.py saved: compare with
PyTorch's output and time it (eager and mx.compile).

    <venv with mlx>/bin/python scripts/lingbot_block_mlx.py outputs/block0


Mirrors apps/lingbot/dit.py LingBotDiT._block under BF16 autocast: FP32 residual stream and norm
statistics, BF16 matmuls (bias added in BF16), FP32 RoPE rotations rounded to BF16, the cache rows of
this chunk overwritten with its keys/values before attention.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

stem = Path(sys.argv[1])
w = mx.load(str(stem.with_suffix(".safetensors")))
meta = json.loads(stem.with_suffix(".json").read_text())
T, H = meta["tokens"], meta["heads"]
dim = w["value"].shape[-1]
D = dim // H
end = meta["end"]
f32, bf16 = mx.float32, mx.bfloat16
scale = D ** -0.5


def linear(x, weight, bias):  # torch F.linear under BF16 autocast: BF16 in, BF16 out
    return mx.addmm(bias.astype(bf16), x.astype(bf16), weight.astype(bf16).T)


def layer_norm(x, eps, weight=None, bias=None):  # WanLayerNorm: FP32 statistics, result in x's dtype
    out = mx.fast.layer_norm(x.astype(f32), None if weight is None else weight.astype(f32),
                             None if bias is None else bias.astype(f32), eps)
    return out.astype(x.dtype)


def rms_norm(x, weight, eps):
    xf = x.astype(f32)
    normed = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return normed.astype(x.dtype) * weight


def rope_real(x, table):  # x [1, T, H, D]; table [T, 1, D/2, 2] FP32
    pairs = x.astype(f32).reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    xr, xi = pairs[..., 0], pairs[..., 1]
    fr, fi = table[..., 0], table[..., 1]
    return mx.stack([xr * fr - xi * fi, xr * fi + xi * fr], axis=-1).reshape(x.shape).astype(x.dtype)


def attention(q, k, v):  # [1, L, H, D] layout, as dit.attention
    out = mx.fast.scaled_dot_product_attention(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3),
                                               v.transpose(0, 2, 1, 3), scale=scale)
    return out.transpose(0, 2, 1, 3)


def block(value, cache_k, cache_v):
    e = mx.split(w["modulation"][None] + w["time_e0"], 6, axis=2)  # [1, 1, 1, dim] each
    e = [t.squeeze(2) for t in e]
    hidden = layer_norm(value, meta["eps"]).astype(f32) * (1 + e[1]) + e[0]
    query, key, val = mx.split(linear(hidden, w["qkv_weight"], w["qkv_bias"]), 3, axis=-1)
    query = rms_norm(query, w["norm_q"], meta["attn_eps"]).reshape(1, T, H, D)
    key = rms_norm(key, w["norm_k"], meta["attn_eps"]).reshape(1, T, H, D)
    val = val.reshape(1, T, H, D)
    query = rope_real(query, w["rope"])
    physical_key = rope_real(key, w["rope"])
    cache_k[:, end - T:end] = physical_key
    cache_v[:, end - T:end] = val
    attended = attention(query, cache_k, cache_v)
    value = value + linear(attended.reshape(1, T, dim), w["o_weight"], w["o_bias"]) * e[2]
    cam = w["cam"]
    value = (1.0 + cam[0]) * value + cam[1]
    q2 = linear(layer_norm(value, meta["norm3_eps"], w["norm3_weight"], w["norm3_bias"]),
                w["cross_q_weight"], w["cross_q_bias"])
    q2 = rms_norm(q2, w["cross_norm_q"], meta["cross_eps"]).reshape(1, T, H, D)
    value = value + linear(attention(q2, w["text_k"], w["text_v"]).reshape(1, T, dim),
                           w["cross_o_weight"], w["cross_o_bias"])
    hidden = layer_norm(value, meta["eps"]).astype(f32) * (1 + e[4]) + e[3]
    up = linear(hidden, w["ffn_up_weight"], w["ffn_up_bias"]).astype(f32)
    up = (0.5 * up * (1 + mx.tanh(0.7978845608028654 * (up + 0.044715 * up * up * up)))).astype(bf16)
    return value + linear(up, w["ffn_down_weight"], w["ffn_down_bias"]) * e[5], cache_k, cache_v


def run(fn):
    out, ck, _ = fn(w["value"], mx.array(w["cache_k"]), mx.array(w["cache_v"]))
    mx.eval(out, ck)
    return out, ck


def timed(fn):
    for _ in range(3):
        run(fn)
    samples = []
    for _ in range(10):
        ck, cv = mx.array(w["cache_k"]), mx.array(w["cache_v"])
        mx.eval(ck, cv)
        t = time.perf_counter()
        mx.eval(*fn(w["value"], ck, cv))
        samples.append((time.perf_counter() - t) * 1000)
    return statistics.median(samples)


def compare(name, ours, theirs):
    a, b = ours.astype(f32), theirs.astype(f32)
    rel = (mx.sqrt(mx.sum((a - b) ** 2)) / mx.sqrt(mx.sum(b * b))).item()
    print(f"  {name:14s} relative L2 {rel:.2e}   max abs {mx.max(mx.abs(a - b)).item():.3g}   "
          f"(|ref| max {mx.max(mx.abs(b)).item():.3g})")


print(f"block 0 at {meta['size']}: {T} query tokens, {end} cache rows; dtypes match the capture")
out, ck = run(block)
print("MLX (eager) vs PyTorch MPS:")
compare("block output", out, w["output"])
compare("cache keys", ck, w["cache_k_after"])
compiled = mx.compile(block)
out_c, _ = run(compiled)
compare("compiled out", out_c, w["output"])
eager_ms, compiled_ms = timed(block), timed(compiled)
torch_ms = meta["torch_block_ms"]
print(f"time per block: PyTorch MPS {torch_ms:.1f} ms | MLX eager {eager_ms:.1f} ms ({torch_ms / eager_ms:.2f}x) | "
      f"MLX compiled {compiled_ms:.1f} ms ({torch_ms / compiled_ms:.2f}x)")
print(f"-> DiT per chunk (30 blocks x 4 passes): PyTorch {torch_ms * 0.12:.1f} s, MLX eager {eager_ms * 0.12:.1f} s, "
      f"MLX compiled {compiled_ms * 0.12:.1f} s")
