#!/usr/bin/env python3
"""Benchmark FastFrames MPS RoPE (isolated, real DiT block, and paired T2V clips)."""
import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from fastframes.device import limit_mps_memory, release_mps_cache  # noqa: E402
from fastframes.wan.model import causal_rope_apply, rope_params  # noqa: E402
from fastframes.wan.mps_rope import _kernel  # noqa: E402


def memory():
    return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()


def bench_isolated():
    freqs = torch.cat([rope_params(1024, 44), rope_params(1024, 42), rope_params(1024, 42)], dim=1)
    grid = torch.tensor([[1, 30, 52]], dtype=torch.long)
    for label, heads in (("FF-S (12 heads)", 12), ("FF-L (40 heads)", 40)):
        x = torch.randn(1, 1560, heads, 128, device="mps", dtype=torch.bfloat16)
        samples = {"eager": [], "metal": []}
        for mode in samples:
            os.environ["FASTFRAMES_MPS_ROPE"] = mode
            release_mps_cache("mps")
            before = memory()
            out = causal_rope_apply(x, grid, freqs, start_frame=7)
            torch.mps.synchronize()
            print(f"{label}, {mode}: live/driver bytes {before} -> {memory()}", flush=True)
            del out
        for repeat in range(13):
            for mode in (samples if repeat % 2 == 0 else reversed(samples)):
                os.environ["FASTFRAMES_MPS_ROPE"] = mode
                torch.mps.synchronize()
                t0 = time.perf_counter()
                out = causal_rope_apply(x, grid, freqs, start_frame=7)
                torch.mps.synchronize()
                if repeat >= 4:
                    samples[mode].append((time.perf_counter() - t0) * 1000)
                del out
        for mode, vals in samples.items():
            joined = " ".join(f"{v:.3f}" for v in vals)
            print(f"{mode} {label}: median {statistics.median(vals):.3f} ms; samples {joined}", flush=True)
        os.environ["FASTFRAMES_MPS_ROPE"] = "eager"
        ref = causal_rope_apply(x, grid, freqs, start_frame=7)
        os.environ["FASTFRAMES_MPS_ROPE"] = "metal"
        cand = causal_rope_apply(x, grid, freqs, start_frame=7)
        assert torch.equal(ref.view(torch.int16), cand.view(torch.int16))
        del x, ref, cand
        release_mps_cache("mps")


def measure_clip(model, prompt_embeds, seed, latent_frames, kv_bits, mode):
    os.environ["FASTFRAMES_MPS_ROPE"] = mode
    release_mps_cache(model.device)
    torch.mps.synchronize()
    gen_start = time.perf_counter()
    frames = model.generate("", seed, prompt_embeds=prompt_embeds, num_frames=latent_frames, kv_bits=kv_bits)
    torch.mps.synchronize()
    gen_s = time.perf_counter() - gen_start
    end_mem = memory()
    hashes = []
    digest = hashlib.sha256()
    for frame in frames.numpy():
        data = frame.tobytes()
        digest.update(data)
        hashes.append(hashlib.sha256(data).hexdigest())
    wall_s = time.perf_counter() - gen_start
    del frames
    return {
        "mode": mode, "generation_s": round(gen_s, 4), "wall_s": round(wall_s, 4),
        "end_live_driver_bytes": end_mem, "frames": len(hashes), "rgb_sha256": digest.hexdigest(),
    }, hashes


def run_clips(args):
    from fastframes.pipeline import MODELS, FastFrames, configure_backends

    spec = MODELS[args.model]
    root = args.checkpoint_dir
    configure_backends(spec.config)
    model = FastFrames.from_pretrained(
        spec.config, root / spec.wan_dir, root / spec.dit, root / spec.refiner, root / spec.decoder,
        dit_bits=args.dit_bits)
    prompt_embeds = model.encode_prompt(args.prompt)
    model.text_encoder = None
    gc.collect()
    release_mps_cache(model.device)

    latent_frames = (args.frames - 1) // 4 + 1
    eager_first = ("eager", "metal")
    reference = None
    stream = args.output.open("w") if args.output else None
    try:
        for pair in range(args.pairs + 1):
            order = eager_first if pair == 0 or pair % 2 == 1 else reversed(eager_first)
            for mode in order:
                rec, hashes = measure_clip(model, prompt_embeds, args.seed, latent_frames, args.kv_bits, mode)
                if reference is None:
                    reference = hashes
                elif hashes != reference:
                    idx = next(i for i, (a, b) in enumerate(zip(hashes, reference)) if a != b)
                    raise RuntimeError(f"Frame {idx} differs from eager warm-up in pair {pair} ({mode})")
                rec.update(pair=pair, model=args.model, frames_requested=args.frames, seed=args.seed,
                           kv_bits=args.kv_bits, dit_bits=args.dit_bits)
                if stream:
                    print(json.dumps(rec), file=stream, flush=True)
                tag = "warm-up" if pair == 0 else "measured"
                print(f"pair {pair} ({tag}), {mode}, {rec['frames']} frames: "
                      f"gen {rec['generation_s']:.3f}s, SHA-256 {rec['rgb_sha256'][:16]}", flush=True)
    finally:
        if stream:
            stream.close()


def bench_blocks():
    from fastframes.wan.causal_model import CausalWanAttentionBlock
    from fastframes.wan.model import WAN_CONFIGS

    freqs = torch.cat([rope_params(1024, 44), rope_params(1024, 42), rope_params(1024, 42)], dim=1)
    grid = torch.tensor([[1, 30, 52]], dtype=torch.long)
    for name in ("Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B"):
        cfg = WAN_CONFIGS[name]
        head_dim = cfg.dim // cfg.num_heads
        block = CausalWanAttentionBlock(
            cfg.dim, cfg.ffn_dim, cfg.num_heads, cfg.qk_norm, cfg.cross_attn_norm, cfg.eps
        ).to("mps", dtype=torch.bfloat16).eval().requires_grad_(False)
        x = torch.randn(1, 1560, cfg.dim, device="mps", dtype=torch.bfloat16)
        e0 = torch.randn(1, 1, 6, cfg.dim, device="mps", dtype=torch.bfloat16)
        cross_kv = {
            "k": torch.randn(1, 512, cfg.num_heads, head_dim, device="mps", dtype=torch.bfloat16),
            "v": torch.randn(1, 512, cfg.num_heads, head_dim, device="mps", dtype=torch.bfloat16),
        }
        kv_cache = {
            "k": torch.randn(1, 21 * 1560, cfg.num_heads, head_dim, device="mps", dtype=torch.bfloat16),
            "v": torch.randn(1, 21 * 1560, cfg.num_heads, head_dim, device="mps", dtype=torch.bfloat16),
        }
        use_ac = name == "Wan2.1-T2V-14B"
        for frame in (0, 20):
            outputs = []
            for mode in ("eager", "metal"):
                os.environ["FASTFRAMES_MPS_ROPE"] = mode
                with torch.autocast("mps", dtype=torch.bfloat16, enabled=use_ac):
                    outputs.append(block(x, e0, grid, freqs, cross_kv, kv_cache=kv_cache, frame=frame))
            assert torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
            del outputs
            samples = {"eager": [], "metal": []}
            for repeat in range(14):
                for mode in (samples if repeat % 2 == 0 else reversed(samples)):
                    os.environ["FASTFRAMES_MPS_ROPE"] = mode
                    torch.mps.synchronize()
                    t0 = time.perf_counter()
                    with torch.autocast("mps", dtype=torch.bfloat16, enabled=use_ac):
                        block(x, e0, grid, freqs, cross_kv, kv_cache=kv_cache, frame=frame)
                    torch.mps.synchronize()
                    if repeat >= 4:
                        samples[mode].append(round((time.perf_counter() - t0) * 1000, 3))
            for mode, vals in samples.items():
                print(f"{name} block, frame {frame}, {mode}: median {statistics.median(vals):.3f} ms; "
                      f"samples {vals}; live/driver bytes {memory()}", flush=True)
        del block, x, e0, cross_kv, kv_cache
        gc.collect()
        release_mps_cache("mps")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("block")
    clip = sub.add_parser("clip")
    clip.add_argument("--model", choices=("ff-s", "ff-l"), default="ff-s")
    clip.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    clip.add_argument("--prompt", default="A corgi surfing a wave at sunset")
    clip.add_argument("--seed", type=int, default=0)
    clip.add_argument("--frames", type=int, default=81)
    clip.add_argument("--kv-bits", type=int, choices=(8, 16), default=16)
    clip.add_argument("--dit-bits", type=int, choices=(8, 16), default=16)
    clip.add_argument("--pairs", type=int, default=4)
    clip.add_argument("--output", type=Path)
    args = parser.parse_args()

    frac = 0.95 if (args.cmd == "clip" and args.model == "ff-l") else 0.6
    limit_mps_memory(frac)
    if not torch.backends.mps.is_available() or _kernel() is None:
        raise RuntimeError("An MPS device with the Metal RoPE shader is required")
    if args.cmd == "clip":
        return run_clips(args)
    if args.cmd == "block":
        torch.manual_seed(0)
        return bench_blocks()
    torch.manual_seed(0)
    bench_isolated()


if __name__ == "__main__":
    main()
