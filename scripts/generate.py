#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Text-to-video generation with FastFrames-S / FastFrames-L.

  python scripts/generate.py --model ff-s --checkpoint-dir checkpoints \
      --prompt "A corgi surfing a wave at sunset" --seed 0 --output outputs/corgi.mp4

  # FastFrames-L on a 64 GB Mac: a 41-frame clip (its KV cache for 81 frames doesn't fit)
  python scripts/generate.py --model ff-l --prompt "..." --frames 41 --output outputs/ffl.mp4

  # one prompt per line, sharded over GPUs (run one process per GPU)
  CUDA_VISIBLE_DEVICES=3 python scripts/generate.py --model ff-l --checkpoint-dir checkpoints \
      --prompt-file prompts.txt --output-dir outputs/ff-l --shard 3 --num-shards 8
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastframes.device import limit_mps_memory  # noqa: E402
from fastframes.pipeline import MODELS, FastFrames, configure_backends  # noqa: E402
from fastframes.video import save_video  # noqa: E402


def load_model(name, checkpoint_dir):
    """Build a released model from ``<checkpoint_dir>`` (see docs/installation.md#checkpoints)."""
    spec = MODELS[name]
    root = Path(checkpoint_dir)
    configure_backends(spec.config)
    return FastFrames.from_pretrained(
        spec.config, root / spec.wan_dir, root / spec.dit, root / spec.refiner, root / spec.decoder)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--seed", type=int, default=0, help="seed of the first prompt (prompt i uses seed + i)")
    parser.add_argument("--output", type=Path, help="output path for --prompt")
    parser.add_argument("--output-dir", type=Path, help="output directory for --prompt-file")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--frames", type=int, default=81,
                        help="video frames, 1 + 4k up to 81; fewer stop the rollout early with a smaller KV cache")
    parser.add_argument("--kv-bits", type=int, choices=(8, 16), default=16,
                        help="KV cache precision; 8 halves its memory (FastFrames-L at 81 frames on a 64 GB Mac)")
    args = parser.parse_args()
    if args.frames % 4 != 1 or not 1 <= args.frames <= 81:
        parser.error("--frames must be 1 + 4k, at most 81")
    latent_frames = (args.frames - 1) // 4 + 1

    if args.prompt is not None:
        jobs = [(args.prompt, args.seed, args.output or Path(f"{args.model}_seed{args.seed}.mp4"))]
    else:
        prompts = [line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()]
        out = args.output_dir or Path("outputs") / args.model
        jobs = [(p, args.seed + i, out / f"{i:04d}.mp4") for i, p in enumerate(prompts)][args.shard::args.num_shards]

    limit_mps_memory()
    model = load_model(args.model, args.checkpoint_dir)
    embeds = {}
    if model.device.type == "mps":
        # Unified memory is shared: encode every prompt up front, then drop the 11 GB umT5 encoder
        # so the DiT rollout has the room (moving it to the CPU would still hold the 11 GB).
        import gc

        import torch
        embeds = {p: model.encode_prompt(p) for p in dict.fromkeys(p for p, _, _ in jobs)}
        model.text_encoder = None
        gc.collect()
        torch.mps.empty_cache()
    for prompt, seed, path in jobs:
        start = time.perf_counter()
        frames = model.generate(prompt, seed, prompt_embeds=embeds.get(prompt), num_frames=latent_frames,
                                kv_bits=args.kv_bits)
        save_video(path, frames)
        print(f"{path}  seed={seed}  {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
