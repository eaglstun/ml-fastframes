#!/usr/bin/env python3
"""A FastFrames-LingBot world generated on this Mac alone and with an iPad sharing the DiT's tokens.

The iPad runs SplitStage.swift from the exo-touch app (`SPIKE_MODE=split`, mac/run_split_stage.sh
there); apps/lingbot/ipad.py (IPadAssist) is this side. Generates the same world (example 00's camera
path, seed 0) three times: Mac alone, split with the iPad, Mac alone again. The two Mac-alone runs
bracket the split one in time (background GPU work moves this Mac's timings) and must be identical.
Reports every chunk's time and how close the split frames are to the Mac's, and writes both videos
and a side-by-side sheet to --out.

    .venv/bin/python scripts/lingbot_ipad_split.py --model lingbot-1.3b \\
        --lingbot-root upstream/lingbot-world-v2 --out outputs/ipad_split [--ipad-share 0.55] [--chunks 10]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # repository root
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from apps.lingbot.conditioning import Trajectory  # noqa: E402
from apps.lingbot.generate import add_model_arguments, apply_size, load_pipeline  # noqa: E402
from apps.lingbot.ipad import DEFAULT_PORT, IPadAssist  # noqa: E402
from apps.lingbot.upstream import import_upstream  # noqa: E402
from fastframes.device import default_device, limit_mps_memory, synchronize  # noqa: E402
from fastframes.video import save_video  # noqa: E402


def generate(pipe, image, prompt, trajectory):
    """Every chunk's frames (uint8, CPU) and time, as generate.py makes them."""
    frames, times = [], []
    with torch.inference_mode():
        pipe.start(image, prompt, seed=0, latents=trajectory.latents)
        for chunk in range(trajectory.num_chunks):
            synchronize(pipe.device)
            start = time.perf_counter()
            frames.append(pipe.step(trajectory.rays(chunk)).cpu())
            synchronize(pipe.device)
            times.append(time.perf_counter() - start)
            print(f"  chunk {chunk}: {times[-1]:.2f} s", flush=True)
    return frames, times


def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--ipad", default="169.254.154.199", help="the iPad's address on the USB link")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ipad-share", type=float, default=0.55, help="fraction of each chunk's tokens on the iPad")
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-only", action="store_true", help="profile the split run alone (no Mac-alone runs)")
    parser.add_argument("--fixed-share", action="store_true", help="keep --ipad-share for every chunk")
    parser.add_argument("--no-row-sync", action="store_true",
                        help="diagnostic: don't send the refined rows to the iPad (its frames go wrong)")
    args = parser.parse_args()
    limit_mps_memory(0.75)
    apply_size(args, 1)
    wan = import_upstream(args.lingbot_root)
    example = Path(wan.__file__).resolve().parents[1] / "examples" / "00"
    trajectory = Trajectory(np.load(example / "poses.npy"), np.load(example / "intrinsics.npy"),
                            13 + 16 * (args.chunks - 1), device=default_device())
    image = Image.open(example / "image.jpg").convert("RGB")
    prompt = (example / "prompt.txt").read_text().strip()
    pipe = load_pipeline(args, 1, 0)
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.split_only:
        print("Mac alone", flush=True)
        alone, alone_times = generate(pipe, image, prompt, trajectory)
    started = time.perf_counter()
    assist = IPadAssist.attach(pipe, args.ipad, args.port, share=args.ipad_share, adaptive=not args.fixed_share)
    if args.no_row_sync:
        assist.sync_rows = lambda read_frames: None
    print(f"with the iPad ({1 - assist.n / assist.tokens:.0%} of the tokens; setup "
          f"{time.perf_counter() - started:.1f} s, iPad footprint {assist.footprint / 1e9:.1f} GB)", flush=True)
    split, split_times = generate(pipe, image, prompt, trajectory)
    passes, waited = assist.stats["passes"], assist.stats["wait"]
    shares, thermal = assist.stats["shares"], assist.stats["thermal"]
    per_pass = {k: v / max(passes, 1) for k, v in assist.link.timing.items()}
    print(f"per pass on this Mac: encode K/V {per_pass['encode']:.2f} s (GPU sync + copy), waiting on the socket "
          f"{per_pass['socket']:.2f} s, decoding the iPad's {per_pass['decode']:.2f} s", flush=True)
    print("iPad share per chunk: " + " ".join(f"{share:.0%}" for share in shares), flush=True)
    if args.split_only:
        assist.detach()
        return
    assist.detach()
    print("Mac alone again", flush=True)
    again, again_times = generate(pipe, image, prompt, trajectory)

    same = all(torch.equal(a, b) for a, b in zip(alone, again))
    print(f"\nthe two Mac-alone runs are {'bit-identical' if same else 'DIFFERENT'}")
    print(f"{'chunk':>5} {'alone':>7} {'split':>7} {'again':>7} {'iPad share':>11}  {'PSNR split vs alone':>20}")
    for c in range(args.chunks):
        print(f"{c:5d} {alone_times[c]:6.2f}s {split_times[c]:6.2f}s {again_times[c]:6.2f}s {shares[c]:10.0%}  "
              f"{psnr(split[c], alone[c]):17.1f} dB", flush=True)
    steady = slice(5, None)  # full cache
    alone_median = statistics.median(alone_times[steady] + again_times[steady])
    split_median = statistics.median(split_times[steady])
    print(f"full-cache chunks (5+): Mac alone median {alone_median:.2f} s, split {split_median:.2f} s "
          f"({alone_median / split_median:.2f}x); the Mac waited {waited / max(passes, 1):.2f} s per pass "
          f"for the iPad's K/V; the iPad ended {thermal}")

    save_video(args.out / "alone.mp4", torch.cat(alone))
    save_video(args.out / "split.mp4", torch.cat(split))
    picks = [1, args.chunks // 2, args.chunks - 1]
    rows = [np.concatenate([alone[c][8].numpy(), split[c][8].numpy()], axis=1) for c in picks]
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out / "alone_vs_split.png")
    print(f"wrote {args.out}/alone.mp4, split.mp4 and alone_vs_split.png (left Mac alone, right split; "
          f"chunks {picks}, middle frame)")


if __name__ == "__main__":
    main()
