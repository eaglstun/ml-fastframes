#!/usr/bin/env python3
"""FastFrames-LingBot's DiT block shapes on this Mac, to compare with an iPad (or another device).

The shapes of one chunk of the camera-control demo: 4 latent frames of queries against a rolling KV
cache of 18 frames, Wan 2.1 1.3B blocks (dim 1536, 12 heads, FFN 8960), BF16 operands, at the
demo's three frame sizes. The same benchmark runs on an iPad as `DiTBench.swift` in the
exo-touch app (mac/run_dit_bench.sh there); together they say how a chunk's tokens should be
split between two devices.

    .venv/bin/python scripts/bench_dit_block.py --backend torch     # what the demo runs (MPS)
    <venv with mlx>/bin/python scripts/bench_dit_block.py --backend mlx

Stop other GPU jobs first (a demo server, Photos' analysis shows up as noise).
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

DIM, HEADS, HEAD_DIM, FFN, TEXT = 1536, 12, 128, 8960, 512
BLOCKS, PASSES, CHUNK_FRAMES, CACHE_FRAMES = 30, 4, 4, 18
SIZES = (("832x464", 52 * 29), ("576x320", 36 * 20), ("416x240", 26 * 15))


def report(size, name, ms, flops=None):
    rate = f" {flops / ms / 1e9:6.2f} TFLOPS" if flops else ""
    print(f"{size} {name:<28s} {ms:8.2f} ms{rate}", flush=True)


def bench_torch():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import torch.nn.functional as F

    from apps.lingbot.dit import attention  # the demo's attention: the Metal flash kernel on MPS
    from fastframes.device import default_device, limit_mps_memory, synchronize

    limit_mps_memory(0.6)
    device = default_device()

    def timed(body):
        for _ in range(3):
            body()
        synchronize(device)
        samples = []
        for _ in range(10):
            start = time.perf_counter()
            body()
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000)
        return statistics.median(samples)

    print(f"device: {device}, torch {torch.__version__}; bf16, median of 10 after 3 warm-ups", flush=True)
    with torch.inference_mode():
        for size, frame_tokens in SIZES:
            q, k = CHUNK_FRAMES * frame_tokens, CACHE_FRAMES * frame_tokens
            bf16 = dict(device=device, dtype=torch.bfloat16)
            x = torch.randn(q, DIM, **bf16)
            weights = {name: torch.randn(out, inp, **bf16) * 0.02 for name, (out, inp) in {
                "qkv": (3 * DIM, DIM), "out": (DIM, DIM), "camera": (2 * DIM, DIM), "q2": (DIM, DIM),
                "o2": (DIM, DIM), "up": (FFN, DIM), "down": (DIM, FFN)}.items()}
            hidden = torch.randn(q, FFN, **bf16)
            query = torch.randn(1, q, HEADS, HEAD_DIM, **bf16)
            cache_k, cache_v = (torch.randn(1, k, HEADS, HEAD_DIM, **bf16) for _ in range(2))
            text_k, text_v = (torch.randn(1, TEXT, HEADS, HEAD_DIM, **bf16) for _ in range(2))
            residual = torch.randn(q, DIM, device=device)
            for name, a, w in (("gemm qkv", x, "qkv"), ("gemm out", x, "out"), ("gemm camera", x, "camera"),
                               ("gemm ffn up", x, "up"), ("gemm ffn down", hidden, "down")):
                weight = weights[w]
                report(size, name, timed(lambda a=a, weight=weight: F.linear(a, weight)),
                       2 * a.shape[0] * a.shape[1] * weight.shape[0])
            report(size, f"self-attention {q}x{k}", timed(lambda: attention(query, cache_k, cache_v)),
                   4 * HEADS * q * k * HEAD_DIM)
            report(size, "cross-attention", timed(lambda: attention(query, text_k, text_v)))

            def block():
                value = residual
                normed = F.layer_norm(value, (DIM,), eps=1e-6).to(torch.bfloat16)
                qq, kk, vv = F.linear(normed, weights["qkv"]).view(1, q, 3, HEADS, HEAD_DIM).unbind(2)
                keys = torch.cat([cache_k[:, q:], kk], 1)
                values = torch.cat([cache_v[:, q:], vv], 1)
                value = value + F.linear(attention(qq, keys, values).reshape(q, DIM), weights["out"]).float()
                scale, shift = F.linear(value.to(torch.bfloat16), weights["camera"]).float().chunk(2, -1)
                value = value * (1 + scale) + shift
                crossq = F.linear(F.layer_norm(value, (DIM,), eps=1e-6).to(torch.bfloat16), weights["q2"])
                crossed = attention(crossq.view(1, q, HEADS, HEAD_DIM), text_k, text_v).reshape(q, DIM)
                value = value + F.linear(crossed, weights["o2"]).float()
                up = F.gelu(F.linear(F.layer_norm(value, (DIM,), eps=1e-6).to(torch.bfloat16), weights["up"]),
                            approximate="tanh")
                return value + F.linear(up, weights["down"]).float()

            ms = timed(block)
            print(f"{size} {'whole block':<28s} {ms:8.2f} ms  -> DiT per chunk (30 blocks x 4 passes) "
                  f"{ms * BLOCKS * PASSES / 1000:.2f} s", flush=True)


def bench_mlx():
    import mlx.core as mx

    def timed(body):
        for _ in range(3):
            mx.eval(body())
        samples = []
        for _ in range(10):
            start = time.perf_counter()
            mx.eval(body())
            samples.append((time.perf_counter() - start) * 1000)
        return statistics.median(samples)

    def normal(*shape, dtype=mx.bfloat16):
        return mx.random.normal(shape).astype(dtype)

    print(f"device: {mx.default_device()}, mlx {mx.__version__}; bf16, median of 10 after 3 warm-ups", flush=True)
    scale = HEAD_DIM ** -0.5
    for size, frame_tokens in SIZES:
        q, k = CHUNK_FRAMES * frame_tokens, CACHE_FRAMES * frame_tokens
        x, hidden = normal(q, DIM), normal(q, FFN)
        weights = {name: normal(inp, out) * 0.02 for name, (inp, out) in {
            "qkv": (DIM, 3 * DIM), "out": (DIM, DIM), "camera": (DIM, 2 * DIM), "q2": (DIM, DIM),
            "o2": (DIM, DIM), "up": (DIM, FFN), "down": (FFN, DIM)}.items()}
        query = normal(1, HEADS, q, HEAD_DIM)
        cache_k, cache_v = normal(1, HEADS, k, HEAD_DIM), normal(1, HEADS, k, HEAD_DIM)
        text_k, text_v = normal(1, HEADS, TEXT, HEAD_DIM), normal(1, HEADS, TEXT, HEAD_DIM)
        residual = normal(q, DIM, dtype=mx.float32)
        mx.eval(x, hidden, query, cache_k, cache_v, text_k, text_v, residual, *weights.values())
        for name, a, w in (("gemm qkv", x, "qkv"), ("gemm out", x, "out"), ("gemm camera", x, "camera"),
                           ("gemm ffn up", x, "up"), ("gemm ffn down", hidden, "down")):
            weight = weights[w]
            report(size, name, timed(lambda a=a, weight=weight: a @ weight),
                   2 * a.shape[0] * a.shape[1] * weight.shape[1])
        sdpa = mx.fast.scaled_dot_product_attention
        report(size, f"self-attention {q}x{k}", timed(lambda: sdpa(query, cache_k, cache_v, scale=scale)),
               4 * HEADS * q * k * HEAD_DIM)
        report(size, "cross-attention", timed(lambda: sdpa(query, text_k, text_v, scale=scale)))

        def heads(t):
            return t.reshape(1, q, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        def merge(t):
            return t.transpose(0, 2, 1, 3).reshape(q, DIM)

        def block():
            value = residual
            normed = mx.fast.layer_norm(value, None, None, 1e-6).astype(mx.bfloat16)
            qq, kk, vv = mx.split(normed @ weights["qkv"], 3, axis=-1)
            keys = mx.concatenate([cache_k[:, :, q:], heads(kk)], axis=2)
            values = mx.concatenate([cache_v[:, :, q:], heads(vv)], axis=2)
            value = value + (merge(sdpa(heads(qq), keys, values, scale=scale)) @ weights["out"]).astype(mx.float32)
            cam_scale, cam_shift = mx.split(value.astype(mx.bfloat16) @ weights["camera"], 2, axis=-1)
            value = value * (1 + cam_scale.astype(mx.float32)) + cam_shift.astype(mx.float32)
            crossq = heads(mx.fast.layer_norm(value, None, None, 1e-6).astype(mx.bfloat16) @ weights["q2"])
            value = value + (merge(sdpa(crossq, text_k, text_v, scale=scale)) @ weights["o2"]).astype(mx.float32)
            up = mx.fast.layer_norm(value, None, None, 1e-6).astype(mx.bfloat16) @ weights["up"]
            up = up * mx.sigmoid(1.702 * up)
            return value + (up @ weights["down"]).astype(mx.float32)

        ms = timed(block)
        print(f"{size} {'whole block':<28s} {ms:8.2f} ms  -> DiT per chunk (30 blocks x 4 passes) "
              f"{ms * BLOCKS * PASSES / 1000:.2f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("torch", "mlx"), default="torch")
    {"torch": bench_torch, "mlx": bench_mlx}[parser.parse_args().backend]()
