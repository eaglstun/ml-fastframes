#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Generate a FastFrames-LingBot video from a first image, a prompt and a camera path.

Use ``--example`` for an upstream scene or ``--condition`` for a JSON camera path
or action sequence. Paths in the JSON are relative to the file.
See docs/lingbot.md#generation for commands and the condition-file format.
"""
import argparse
import json
import os
import sys
import time
import warnings
from datetime import timedelta
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])  # repository root, not this directory
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from PIL import Image  # noqa: E402

from apps.lingbot.conditioning import (CHUNK, DEFAULT_SIZE, KeyboardCamera, LiveCamera, Trajectory,  # noqa: E402
                                       encode_structure, parse_size, set_size)
from apps.lingbot.pipeline import MODELS, FastFramesLingBot, configure_backends  # noqa: E402
from apps.lingbot.upstream import import_upstream  # noqa: E402
from fastframes.device import default_device, synchronize  # noqa: E402
from fastframes.video import save_video  # noqa: E402

OFFICIAL = {"lingbot-1.3b": "lingbot-world-v2-1.3b-causal-fast"}
SHARED_ASSETS = "lingbot-world-v2-14b-causal-fast"


def add_model_arguments(parser):
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--lingbot-root", help="LingBot-World clone (default: $LINGBOT_WORLD_ROOT)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                        help="holds <model>/ (FastFrames weights) and the official LingBot-World weights")
    parser.add_argument("--lingbot-checkpoint", type=Path, help="official weights (default: <checkpoint-dir>/"
                        "lingbot-world-v2-1.3b-causal-fast)")
    parser.add_argument("--assets-dir", type=Path, help="T5/VAE/tokenizer if not in --lingbot-checkpoint "
                        "(default: <checkpoint-dir>/lingbot-world-v2-14b-causal-fast)")
    parser.add_argument("--eager", action="store_true", help="no torch.compile and no CUDA graphs")
    parser.add_argument("--size", type=parse_size, default=DEFAULT_SIZE, metavar="WxH",
                        help="frame size, multiples of 16 (default 832x464, the trained size; smaller is faster "
                             "and out of distribution, e.g. 576x320, see conditioning.set_size)")


def apply_size(args, world):
    """Set the frame size of ``--size``; call before building cameras or the pipeline."""
    if tuple(args.size) != DEFAULT_SIZE and world > 1:
        raise ValueError("--size other than 832x464 was only tried on one device")
    set_size(*args.size)


def init_distributed():
    """(world, rank); eight ranks must share one visible-device namespace (``torchrun`` does)."""
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world not in (1, 8):
        raise RuntimeError(f"FastFrames-LingBot runs on 1 or 8 GPUs, not {world}")
    if not torch.cuda.is_available():
        if world > 1:
            raise RuntimeError("multi-device LingBot needs CUDA")
        return world, rank
    torch.cuda.set_device(local_rank)
    if world > 1:
        # Context parallel with expandable segments ended in an illegal memory access during warm-up.
        for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            if "expandable_segments:true" in os.environ.get(name, "").replace(" ", "").lower():
                raise RuntimeError(f"{name}={os.environ[name]!r}: expandable segments are not supported with "
                                   f"more than one GPU; unset {name}")
        dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device("cuda", local_rank))
    return world, rank


def load_pipeline(args, world, rank):
    configure_backends()
    root = args.checkpoint_dir
    eager = args.eager or not torch.cuda.is_available()
    return FastFramesLingBot.from_pretrained(
        args.model, lingbot_root=args.lingbot_root,
        lingbot_checkpoint=args.lingbot_checkpoint or root / OFFICIAL[args.model],
        assets_dir=args.assets_dir or root / SHARED_ASSETS, checkpoint_dir=root / args.model,
        # torch.compile (inductor max-autotune) and CUDA graphs are CUDA-only here: eager elsewhere
        world=world, rank=rank, compile=not eager, graphs=not eager)


def read_condition(args):
    """-> (image, prompt, camera, number of chunks, seed)."""
    if args.example is not None:
        base, spec = args.example, {"image": "image.jpg", "poses": "poses.npy", "intrinsics": "intrinsics.npy"}
        prompt_file = base / "prompt.txt"
        spec["prompt"] = prompt_file.read_text().strip() if prompt_file.is_file() else ""
        if not spec["prompt"] and args.prompt is None:
            warnings.warn(f"{base} has no prompt.txt; generating with an empty prompt (pass --prompt)")
    else:
        base, spec = args.condition.parent, json.loads(args.condition.read_text())
    image = Image.open(args.image or base / spec["image"]).convert("RGB")
    prompt = args.prompt if args.prompt is not None else spec["prompt"]
    seed = args.seed if args.seed is not None else int(spec.get("seed", 0))
    frames = args.frames or spec.get("frames")
    if "actions" in spec:
        actions = list(spec["actions"])
        if frames:
            actions = actions[:(frames + 3) // 16]
        keyboard, live = KeyboardCamera(), LiveCamera(device=default_device())
        return image, prompt, (lambda chunk: live.rays(keyboard.chunk(actions[chunk]))), len(actions), seed
    poses = np.load(base / spec["poses"])
    intrinsics = np.load(base / spec["intrinsics"])
    trajectory = Trajectory(poses, intrinsics, frames or len(poses), device=default_device())
    return image, prompt, trajectory.rays, trajectory.num_chunks, seed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--condition", type=Path, help="JSON condition file")
    source.add_argument("--example", type=Path, help="an upstream example directory")
    parser.add_argument("--prompt", help="override the prompt")
    parser.add_argument("--image", type=Path, help="override the first image")
    parser.add_argument("--structure", type=Path,
                        help="directory of frames at 16 fps (e.g. blender_stage.py --structure) that steer every "
                        "frame, SDEdit style; frame 0 is the first image's view")
    parser.add_argument("--structure-step", type=int, default=2,
                        help="denoising step the structure starts at, 1-3: higher follows it more closely, rougher")
    parser.add_argument("--steps", default="0,1,3", type=lambda v: tuple(int(i) for i in v.split(",")),
                        help="denoising steps to run, indices into the 4-step schedule from 0 to 3 "
                        "(default 0,1,3: three DiT passes per chunk; 0,1,2,3 runs all four, as trained)")
    parser.add_argument("--frames", type=int, help="RGB frames to generate (rounded down to whole chunks)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    world, rank = init_distributed()
    apply_size(args, world)
    import_upstream(args.lingbot_root)
    image, prompt, rays, chunks, seed = read_condition(args)
    pipe = load_pipeline(args, world, rank)
    start = time.perf_counter()
    structure = None
    if args.structure is not None:
        paths = sorted(p for p in args.structure.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        # before start(): both use the VAE encoder's feature cache
        structure = encode_structure(pipe.upstream.vae, [Image.open(p) for p in paths], CHUNK * chunks,
                                     pipe.device, first_image=image)
    pipe.start(image, prompt, seed=seed, latents=CHUNK * chunks)
    frames = [pipe.step(rays(chunk), structure=None if structure is None else
                        structure[:, CHUNK * chunk:CHUNK * (chunk + 1)], structure_step=args.structure_step,
                        steps=args.steps)
              for chunk in range(chunks)]
    synchronize(pipe.device)
    if rank == 0:
        video = torch.cat(frames).cpu()
        save_video(args.output, video)
        print(f"{args.output}  {len(video)} frames  seed={seed}  {time.perf_counter() - start:.1f}s", flush=True)
    pipe.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
