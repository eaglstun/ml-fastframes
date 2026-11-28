#!/usr/bin/env python3
"""Benchmark MPS fused RMSNorm (isolated, real DiT block, and paired clips)."""
import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from apps.lingbot.dit import rms_norm  # noqa: E402
from fastframes.device import limit_mps_memory, release_mps_cache  # noqa: E402
from fastframes.wan.model import WanRMSNorm  # noqa: E402
from fastframes.wan.mps_rmsnorm import _kernel  # noqa: E402


def memory():
    return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()


def psnr_db(a, b):
    mse = (a.float() - b.float()).square().mean().item()
    return float("inf") if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)


def bench_isolated():
    configs = (
        ("LingBot 416x240 Q+K+CrossQ", 1560, 1536),
        ("LingBot 576x320 Q+K+CrossQ", 2880, 1536),
        ("LingBot 832x464 Q+K+CrossQ", 6032, 1536),
        ("FF-S Q+K+CrossQ", 1560, 1536),
        ("FF-L Q+K+CrossQ", 1560, 5120),
    )
    for label, tokens, dim in configs:
        qkv = torch.randn(1, tokens, 3 * dim, device="mps", dtype=torch.bfloat16)
        qs, ks, _ = qkv.chunk(3, dim=-1)
        cq = torch.randn(1, tokens, dim, device="mps", dtype=torch.bfloat16)
        nq = WanRMSNorm(dim, eps=1e-6).to("mps", dtype=torch.bfloat16)
        nk = WanRMSNorm(dim, eps=1e-6).to("mps", dtype=torch.bfloat16)
        nc = WanRMSNorm(dim, eps=1e-6).to("mps", dtype=torch.bfloat16)

        for mode in ("eager", "metal"):
            os.environ["FASTFRAMES_MPS_RMSNORM"] = mode
            release_mps_cache("mps")
            before = memory()
            out = (rms_norm(qs, nq), rms_norm(ks, nk), rms_norm(cq, nc))
            torch.mps.synchronize()
            print(f"{label}, {mode}: live/driver {before} -> {memory()}", flush=True)
            del out

        samples = {"eager": [], "metal": []}
        for repeat in range(13):
            for mode in (samples if repeat % 2 == 0 else reversed(samples)):
                os.environ["FASTFRAMES_MPS_RMSNORM"] = mode
                torch.mps.synchronize()
                t0 = time.perf_counter()
                out = (rms_norm(qs, nq), rms_norm(ks, nk), rms_norm(cq, nc))
                torch.mps.synchronize()
                if repeat >= 4:
                    samples[mode].append((time.perf_counter() - t0) * 1000)
                del out
        e_med = statistics.median(samples["eager"])
        m_med = statistics.median(samples["metal"])
        print(f"{label}: eager {e_med:.3f} ms -> metal {m_med:.3f} ms ({e_med / m_med:.1f}x, "
              f"-{e_med - m_med:.3f} ms/block)", flush=True)
        del qkv, cq, nq, nk, nc
        gc.collect()
        release_mps_cache("mps")


def run_lingbot_block(args):
    from apps.lingbot.generate import apply_size, import_upstream, load_pipeline, read_condition

    apply_size(args, 1)
    import_upstream(args.lingbot_root)
    image, prompt, rays, chunks, seed = read_condition(args)
    pipe = load_pipeline(args, 1, 0)
    try:
        pipe.start(image, prompt, seed=seed, latents=chunks * 4)
        original = pipe.dit._block_run

        def measure_first(*inputs):
            pipe.dit._block_run = original
            samples = {"eager": [], "metal": []}
            for repeat in range(14):
                for mode in (samples if repeat % 2 == 0 else reversed(samples)):
                    os.environ["FASTFRAMES_MPS_RMSNORM"] = mode
                    torch.mps.synchronize()
                    t0 = time.perf_counter()
                    original(*inputs)
                    torch.mps.synchronize()
                    if repeat >= 4:
                        samples[mode].append(round((time.perf_counter() - t0) * 1000, 3))
            for mode, vals in samples.items():
                print(f"{args.size[0]}x{args.size[1]} block 0, {inputs[-2]} cache frames, {mode}: "
                      f"median {statistics.median(vals):.3f} ms; samples {vals}", flush=True)
            os.environ["FASTFRAMES_MPS_RMSNORM"] = "metal"
            return original(*inputs)

        for c in range(chunks):
            if c in (0, 5):
                pipe.dit._block_run = measure_first
            pipe.step(rays(c))
    finally:
        pipe.close()


def run_lingbot_clips(args):
    from apps.lingbot.generate import apply_size, import_upstream, load_pipeline, read_condition

    apply_size(args, 1)
    import_upstream(args.lingbot_root)
    image, prompt, rays, chunks, seed = read_condition(args)
    pipe = load_pipeline(args, 1, 0)
    eager_first = ("eager", "metal")
    ref_frames = None
    stream = args.output.open("w") if args.output else None
    try:
        for pair in range(args.pairs + 1):
            order = eager_first if pair == 0 or pair % 2 == 1 else reversed(eager_first)
            for mode in order:
                os.environ["FASTFRAMES_MPS_RMSNORM"] = mode
                pipe.start(image, prompt, seed=seed, latents=chunks * 4)
                torch.mps.synchronize()
                chunk_s, footprints, all_frames = [], [], []
                wall_t0 = time.perf_counter()
                for c in range(chunks):
                    torch.mps.synchronize()
                    t0 = time.perf_counter()
                    fr = pipe.step(rays(c))
                    torch.mps.synchronize()
                    chunk_s.append(time.perf_counter() - t0)
                    footprints.append(memory())
                    all_frames.append(fr.cpu())
                clip_u8 = torch.cat(all_frames, dim=0)
                wall_s = time.perf_counter() - wall_t0
                if ref_frames is None and mode == "eager":
                    ref_frames = clip_u8
                if pair == 1 and args.save_frames_dir:
                    from PIL import Image
                    args.save_frames_dir.mkdir(parents=True, exist_ok=True)
                    for idx in (0, len(clip_u8) // 2, len(clip_u8) - 1):
                        Image.fromarray(clip_u8[idx].numpy()).save(
                            args.save_frames_dir / f"lb_{args.size[0]}x{args.size[1]}_f{idx:02d}_{mode}.png")
                psnr = psnr_db(clip_u8, ref_frames) if ref_frames is not None else None
                rec = {
                    "pair": pair, "mode": mode, "size": args.size, "frames": int(clip_u8.shape[0]),
                    "generation_s": round(sum(chunk_s), 4), "wall_s": round(wall_s, 4),
                    "chunk_s": [round(x, 4) for x in chunk_s],
                    "chunk_end_live_driver_bytes": footprints,
                    "psnr_vs_eager_warmup_db": None if psnr is None or math.isinf(psnr) else round(psnr, 2),
                    "exact_match_vs_eager_warmup": bool(ref_frames is not None and torch.equal(clip_u8, ref_frames)),
                }
                if stream:
                    print(json.dumps(rec), file=stream, flush=True)
                tag = "warm-up" if pair == 0 else "measured"
                print(f"pair {pair} ({tag}), {mode}: gen {rec['generation_s']:.3f}s, "
                      f"exact={rec['exact_match_vs_eager_warmup']}, psnr={rec['psnr_vs_eager_warmup_db']}", flush=True)
                del clip_u8, all_frames
    finally:
        if stream:
            stream.close()
        pipe.close()


@torch.inference_mode()
def main():
    from apps.lingbot.generate import add_model_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    for mode in ("block", "lingbot"):
        clip = sub.add_parser(mode)
        add_model_arguments(clip)
        clip.add_argument("--example", type=Path, required=True)
        clip.set_defaults(frames=93, seed=0, image=None, prompt=None)
        if mode == "lingbot":
            clip.add_argument("--pairs", type=int, default=4)
            clip.add_argument("--output", type=Path)
            clip.add_argument("--save-frames-dir", type=Path)
    args = parser.parse_args()

    frac = 0.6 if (args.cmd in ("block", "lingbot") and tuple(args.size) == (832, 464)) else 0.45
    limit_mps_memory(frac)
    if not torch.backends.mps.is_available() or _kernel() is None:
        raise RuntimeError("An MPS device with the Metal RMSNorm shader is required")
    if args.cmd == "block":
        return run_lingbot_block(args)
    if args.cmd == "lingbot":
        return run_lingbot_clips(args)
    torch.manual_seed(0)
    bench_isolated()


if __name__ == "__main__":
    main()
