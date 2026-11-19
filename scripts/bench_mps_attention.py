"""Correctness and speed of the Metal flash-attention kernel vs per-head MPS SDPA.

    .venv/bin/python scripts/bench_mps_attention.py [--tiling RT,NK] [--skip-check] [--skip-sdpa]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastframes.wan import mps_flash_attention as mfa  # noqa: E402

H, D, LQ = 12, 128, 1560
LENGTHS = [n * LQ for n in range(1, 22)]


def sdpa_per_head(q, k, v):  # [B, L, H, D] in/out, as in fastframes.wan.attention on MPS
    q, k, v = (u.transpose(1, 2) for u in (q, k, v))
    out = torch.cat([F.scaled_dot_product_attention(q[:, h:h + 1], k[:, h:h + 1], v[:, h:h + 1])
                     for h in range(q.shape[1])], dim=1)
    return out.transpose(1, 2).contiguous()


def make(lk, gain=1.0, seed=0):
    g = torch.Generator().manual_seed(seed + lk)
    q = (torch.randn(1, LQ, H, D, generator=g) * gain).to(torch.bfloat16)
    k = torch.randn(1, lk, H, D, generator=g).to(torch.bfloat16)
    v = torch.randn(1, lk, H, D, generator=g).to(torch.bfloat16)
    return q, k, v


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        torch.mps.synchronize()
        t = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiling", default=None)
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--skip-sdpa", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    tiling = tuple(int(x) for x in args.tiling.split(",")) if args.tiling else None
    assert mfa.is_available(D), "kernel unavailable"
    flash = lambda q, k, v: mfa.flash_attention(q, k, v, tiling=tiling)  # noqa: E731

    if not args.skip_check:
        print("correctness vs fp32 CPU SDPA (abs error; outputs are O(1/sqrt(L))..O(1))")
        print(f"{'L':>6} {'gain':>4} | {'flash max':>10} {'flash mean':>10} | {'sdpa max':>10} {'sdpa mean':>10} | ref rms")
        for lk in [512] + LENGTHS:
            for gain in (1.0, 4.0):
                if gain != 1.0 and lk not in (512, 1560, 15600, 32760):
                    continue
                q, k, v = make(lk, gain)
                ref = F.scaled_dot_product_attention(*(u.float().transpose(1, 2) for u in (q, k, v))).transpose(1, 2)
                qm, km, vm = (u.to("mps") for u in (q, k, v))
                e = (flash(qm, km, vm).float().cpu() - ref).abs()
                if args.skip_sdpa:
                    es = torch.full((1,), float("nan"))
                else:
                    es = (sdpa_per_head(qm, km, vm).float().cpu() - ref).abs()
                assert torch.isfinite(e).all()
                print(f"{lk:>6} {gain:>4.0f} | {e.max():>10.2e} {e.mean():>10.2e} | {es.max():>10.2e} {es.mean():>10.2e} |"
                      f" {ref.pow(2).mean().sqrt():.2e}")
                del qm, km, vm
        torch.mps.empty_cache()

    print(f"\nspeed (ms, median of {args.reps}, warm, synchronized); driver memory after each shape")
    print(f"{'L':>6} | {'flash':>8} {'TFLOPS':>6} {'drv GB':>7} | {'sdpa/head':>9} {'drv GB':>7}")
    tot_f = tot_s = 0.0
    for lk in [512] + LENGTHS:
        q, k, v = (u.to("mps") for u in make(lk))
        flash(q, k, v)
        tf = timed(lambda: flash(q, k, v), args.reps)
        torch.mps.synchronize(); torch.mps.empty_cache()
        mem_f = torch.mps.driver_allocated_memory() / 2**30
        ts = mem_s = float("nan")
        if not args.skip_sdpa:
            sdpa_per_head(q, k, v)
            ts = timed(lambda: sdpa_per_head(q, k, v), args.reps)
            torch.mps.synchronize(); torch.mps.empty_cache()
            mem_s = torch.mps.driver_allocated_memory() / 2**30
        if lk != 512:
            tot_f += tf
            tot_s += ts
        print(f"{lk:>6} | {tf:>8.2f} {4 * LQ * lk * D * H / tf / 1e9:>6.2f} {mem_f:>7.2f} | {ts:>9.2f} {mem_s:>7.2f}")
        del q, k, v
    print(f"sum over the 21 self-attention lengths: flash {tot_f:.1f} ms, per-head SDPA {tot_s:.1f} ms")


if __name__ == "__main__":
    main()
