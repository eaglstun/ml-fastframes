#!/usr/bin/env python3
"""Benchmark MPS PixelDecoder and StreamingDecoder depthwise 3D conv and verify visual/PSNR parity."""
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

from apps.lingbot.decoder import StreamingDecoder  # noqa: E402
from fastframes.device import limit_mps_memory, release_mps_cache  # noqa: E402
from fastframes.models.mps_dwconv3d import _lib  # noqa: E402
from fastframes.models.pixel_decoder import BridgedPixelDecoder, PixelDecoder  # noqa: E402


def memory():
    return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()


def psnr_db(a, b):
    mse = (a.float() - b.float()).square().mean().item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)


def bench_isolated():
    print("=== LingBot StreamingDecoder (4 latent frames -> 16 RGB frames) ===", flush=True)
    for size, (rows, width) in (("416x240", (15, 26)), ("576x320", (20, 36)), ("832x464", (29, 52))):
        dec = PixelDecoder(1536).to("mps", dtype=torch.bfloat16).eval()
        for p in dec.parameters():
            p.normal_(std=0.02)
        sdec = StreamingDecoder(dec, rows=rows, width=width)
        feat = torch.randn(1, 4 * rows * width, 1536, device="mps", dtype=torch.float32)
        mod = torch.randn(1, 4, 1, 1536, device="mps", dtype=torch.float32)

        for mode in ("native", "metal"):
            os.environ["FASTFRAMES_MPS_DECODER_CONV"] = mode
            sdec.reset()
            release_mps_cache("mps")
            before = memory()
            _ = sdec(feat, mod)
            torch.mps.synchronize()
            print(f"{size}, {mode}: live/driver {before} -> {memory()}", flush=True)

        samples = {"native": [], "metal": []}
        for repeat in range(13):
            for mode in (samples if repeat % 2 == 0 else reversed(samples)):
                os.environ["FASTFRAMES_MPS_DECODER_CONV"] = mode
                sdec.reset()
                torch.mps.synchronize()
                t0 = time.perf_counter()
                _ = sdec(feat, mod)
                torch.mps.synchronize()
                if repeat >= 4:
                    samples[mode].append((time.perf_counter() - t0) * 1000)
        n_med = statistics.median(samples["native"])
        m_med = statistics.median(samples["metal"])
        print(f"{size}: native {n_med:.2f} ms -> metal {m_med:.2f} ms ({n_med / m_med:.2f}x, "
              f"-{n_med - m_med:.2f} ms/decode, -{(n_med - m_med) * 3:.1f} ms per 2-preview demo turn)", flush=True)
        del dec, sdec, feat, mod
        gc.collect()
        release_mps_cache("mps")

    print("\n=== FastFrames Clip PixelDecoder (21 latent frames -> 81 RGB frames) ===", flush=True)
    for label, cls, dtype, dim in (
        ("FF-S PixelDecoder (FP32 weights, BF16 autocast)", PixelDecoder, torch.float32, 1536),
        ("FF-L BridgedPixelDecoder (BF16 weights)", BridgedPixelDecoder, torch.bfloat16, 5120),
    ):
        dec = cls(dim).to("mps", dtype=dtype).eval()
        feat = torch.randn(1, 21 * 1560, dim, device="mps", dtype=torch.bfloat16)
        mod = torch.randn(1, 21, 1, dim, device="mps", dtype=torch.bfloat16)
        grid = torch.tensor([[21, 30, 52]], device="mps", dtype=torch.long)

        for mode in ("native", "metal"):
            os.environ["FASTFRAMES_MPS_DECODER_CONV"] = mode
            release_mps_cache("mps")
            before = memory()
            with torch.autocast("mps", dtype=torch.bfloat16):
                _ = dec(feat, mod, grid)
            torch.mps.synchronize()
            print(f"{label}, {mode}: live/driver {before} -> {memory()}", flush=True)

        samples = {"native": [], "metal": []}
        for repeat in range(9):
            for mode in (samples if repeat % 2 == 0 else reversed(samples)):
                os.environ["FASTFRAMES_MPS_DECODER_CONV"] = mode
                torch.mps.synchronize()
                t0 = time.perf_counter()
                with torch.autocast("mps", dtype=torch.bfloat16):
                    _ = dec(feat, mod, grid)
                torch.mps.synchronize()
                if repeat >= 3:
                    samples[mode].append((time.perf_counter() - t0) * 1000)
        n_med = statistics.median(samples["native"])
        m_med = statistics.median(samples["metal"])
        print(f"{label}: native {n_med:.2f} ms -> metal {m_med:.2f} ms ({n_med / m_med:.2f}x)", flush=True)
        del dec, feat, mod
        gc.collect()
        release_mps_cache("mps")


def check_real_ffs(args):
    from fastframes.pipeline import MODELS, FastFrames, configure_backends, to_uint8

    spec = MODELS["ff-s"]
    root = args.checkpoint_dir
    configure_backends(spec.config)
    model = FastFrames.from_pretrained(
        spec.config, root / spec.wan_dir, root / spec.dit, root / spec.refiner, root / spec.decoder)
    embeds = model.encode_prompt(args.prompt)
    model.text_encoder = None
    gc.collect()
    release_mps_cache(model.device)

    _, features, modulation = model.rollout(embeds, args.seed)
    release_mps_cache(model.device)

    # Compare both MPS paths against CPU FP32 reference decode on real rollout features!
    dec_cpu = PixelDecoder(1536).cpu().float().eval()
    dec_cpu.load_state_dict(model.decoder.state_dict())
    grid_cpu = torch.tensor([[modulation.shape[1], 30, 52]], dtype=torch.long)
    from fastframes.models.pixel_decoder import align_frames
    ref_pixels = align_frames(dec_cpu(features.cpu().float(), modulation.cpu().float(), grid_cpu), 81)
    ref_u8 = to_uint8(ref_pixels[0])
    del dec_cpu, ref_pixels

    results_u8 = {}
    samples = {"native": [], "metal": []}
    for mode in ("native", "metal"):
        os.environ["FASTFRAMES_MPS_DECODER_CONV"] = mode
        results_u8[mode] = to_uint8(model.decode(features, modulation)[0]).cpu()
        print(f"Real FF-S 81-frame decode ({mode}) vs CPU FP32 ref: "
              f"mean PSNR {psnr_db(results_u8[mode], ref_u8):.2f} dB, "
              f"frame 0 {psnr_db(results_u8[mode][0], ref_u8[0]):.2f} dB, "
              f"frame 40 {psnr_db(results_u8[mode][40], ref_u8[40]):.2f} dB, "
              f"frame 80 {psnr_db(results_u8[mode][80], ref_u8[80]):.2f} dB, "
              f"max uint8 diff {(results_u8[mode].int() - ref_u8.int()).abs().max().item()}", flush=True)
    print(f"Metal vs Native MPS uint8 PSNR: {psnr_db(results_u8['metal'], results_u8['native']):.2f} dB, "
          f"max uint8 diff {(results_u8['metal'].int() - results_u8['native'].int()).abs().max().item()}", flush=True)

    if args.save_frames_dir:
        from PIL import Image
        args.save_frames_dir.mkdir(parents=True, exist_ok=True)
        for idx in (0, 40, 80):
            for mode in ("native", "metal"):
                Image.fromarray(results_u8[mode][idx].numpy()).save(args.save_frames_dir / f"ffs_f{idx:02d}_{mode}.png")

    for repeat in range(11):
        for mode in (samples if repeat % 2 == 0 else reversed(samples)):
            os.environ["FASTFRAMES_MPS_DECODER_CONV"] = mode
            torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = model.decode(features, modulation)
            torch.mps.synchronize()
            if repeat >= 3:
                samples[mode].append(round((time.perf_counter() - t0) * 1000, 2))
    for mode, vals in samples.items():
        print(f"Real FF-S 81-frame decode {mode}: median {statistics.median(vals):.2f} ms; samples {vals}", flush=True)


def run_lingbot_clips(args):
    from apps.lingbot.generate import apply_size, import_upstream, load_pipeline, read_condition

    apply_size(args, 1)
    import_upstream(args.lingbot_root)
    image, prompt, rays, chunks, seed = read_condition(args)
    pipe = load_pipeline(args, 1, 0)
    native_first = ("native", "metal")
    ref_frames = None
    stream = args.output.open("w") if args.output else None
    try:
        for pair in range(args.pairs + 1):
            order = native_first if pair == 0 or pair % 2 == 1 else reversed(native_first)
            for mode in order:
                os.environ["FASTFRAMES_MPS_DECODER_CONV"] = mode
                pipe.start(image, prompt, seed=seed, latents=chunks * 4)
                torch.mps.synchronize()
                chunk_s, footprints, all_frames = [], [], []
                preview_count = 0

                def on_preview(_idx, _fr):
                    nonlocal preview_count
                    preview_count += 1

                wall_t0 = time.perf_counter()
                for c in range(chunks):
                    torch.mps.synchronize()
                    t0 = time.perf_counter()
                    fr = pipe.step(rays(c), preview=on_preview if args.previews else None)
                    torch.mps.synchronize()
                    chunk_s.append(time.perf_counter() - t0)
                    footprints.append(memory())
                    all_frames.append(fr.cpu())
                clip_u8 = torch.cat(all_frames, dim=0)
                wall_s = time.perf_counter() - wall_t0
                if ref_frames is None and mode == "native":
                    ref_frames = clip_u8
                psnr = psnr_db(clip_u8, ref_frames) if ref_frames is not None else None
                max_u8 = int((clip_u8.int() - ref_frames.int()).abs().max().item()) if ref_frames is not None else 0
                rec = {
                    "pair": pair, "mode": mode, "size": args.size, "previews": args.previews,
                    "preview_calls": preview_count, "frames": int(clip_u8.shape[0]),
                    "generation_s": round(sum(chunk_s), 4), "wall_s": round(wall_s, 4),
                    "chunk_s": [round(x, 4) for x in chunk_s],
                    "chunk_end_live_driver_bytes": footprints,
                    "psnr_vs_native_warmup_db": None if psnr is None or math.isinf(psnr) else round(psnr, 2),
                    "max_u8_diff_vs_native_warmup": max_u8,
                }
                if stream:
                    print(json.dumps(rec), file=stream, flush=True)
                tag = "warm-up" if pair == 0 else "measured"
                print(f"pair {pair} ({tag}), {mode} (previews={preview_count}): gen {rec['generation_s']:.3f}s, "
                      f"psnr={rec['psnr_vs_native_warmup_db']} dB, max_u8={max_u8}", flush=True)
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
    real = sub.add_parser("ffs")
    real.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    real.add_argument("--prompt", default="A corgi surfing a wave at sunset")
    real.add_argument("--seed", type=int, default=0)
    real.add_argument("--save-frames-dir", type=Path)
    lb = sub.add_parser("lingbot")
    add_model_arguments(lb)
    lb.add_argument("--example", type=Path, required=True)
    lb.set_defaults(frames=93, seed=0, image=None, prompt=None)
    lb.add_argument("--pairs", type=int, default=4)
    lb.add_argument("--previews", action="store_true", default=True)
    lb.add_argument("--no-previews", dest="previews", action="store_false")
    lb.add_argument("--output", type=Path)
    args = parser.parse_args()

    frac = 0.6 if (args.cmd == "ffs" or (args.cmd == "lingbot" and tuple(args.size) == (832, 464))) else 0.45
    limit_mps_memory(frac)
    if not torch.backends.mps.is_available() or _lib() is None:
        raise RuntimeError("An MPS device with the Metal decoder shader is required")
    if args.cmd == "ffs":
        return check_real_ffs(args)
    if args.cmd == "lingbot":
        return run_lingbot_clips(args)
    torch.manual_seed(0)
    bench_isolated()


if __name__ == "__main__":
    main()
