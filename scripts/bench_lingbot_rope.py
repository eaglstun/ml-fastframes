#!/usr/bin/env python3
"""Benchmark MPS RoPE at all three demo sizes or on a real model (--help)."""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from apps.lingbot.dit import rope_real  # noqa: E402
from apps.lingbot.mps_rope import _kernel  # noqa: E402
from fastframes.device import limit_mps_memory, release_mps_cache  # noqa: E402


def memory():
    return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()


def run(size, tokens, batch):
    x = torch.randn(batch, tokens, 12, 128, device="mps", dtype=torch.bfloat16)
    table = torch.randn(tokens, 1, 64, 2, device="mps")
    samples = {"eager": [], "metal": []}
    for mode in samples:
        os.environ["FASTFRAMES_MPS_ROPE"] = mode
        release_mps_cache("mps")
        before = memory()
        output = rope_real(x, table)
        torch.mps.synchronize()
        print(f"{size}, batch {batch}, {mode}: live/driver bytes {before} -> {memory()}", flush=True)
        del output

    for repeat in range(12):
        for mode in (samples if repeat % 2 == 0 else reversed(samples)):
            os.environ["FASTFRAMES_MPS_ROPE"] = mode
            torch.mps.synchronize()
            start = time.perf_counter()
            output = rope_real(x, table)
            torch.mps.synchronize()
            if repeat >= 3:
                samples[mode].append((time.perf_counter() - start) * 1000)
            del output
    for mode, values in samples.items():
        times = " ".join(f"{ms:.3f}" for ms in values)
        print(f"{mode} {size} batch {batch}: median {statistics.median(values):.3f} ms; samples {times}", flush=True)

    os.environ["FASTFRAMES_MPS_ROPE"] = "eager"
    expected = rope_real(x, table)
    os.environ["FASTFRAMES_MPS_ROPE"] = "metal"
    actual = rope_real(x, table)
    assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))


def measure_block(original, *args):
    # _block overwrites the current cache slot before reading it, so replays use identical inputs.
    outputs = []
    for mode in ("eager", "metal"):
        os.environ["FASTFRAMES_MPS_ROPE"] = mode
        outputs.append(original(*args))
    assert torch.equal(*outputs)
    del outputs
    samples = {"eager": [], "metal": []}
    for repeat in range(14):
        for mode in (samples if repeat % 2 == 0 else reversed(samples)):
            os.environ["FASTFRAMES_MPS_ROPE"] = mode
            torch.mps.synchronize()
            start = time.perf_counter()
            original(*args)
            torch.mps.synchronize()
            if repeat >= 4:
                samples[mode].append(round(1000 * (time.perf_counter() - start), 3))
    for mode, values in samples.items():
        print(f"block 0, {args[-2]} cache frames, {mode}: median {statistics.median(values):.3f} ms; "
              f"samples {values}; live/driver bytes {memory()}", flush=True)
    os.environ["FASTFRAMES_MPS_ROPE"] = "metal"
    return original(*args)


def measure_clip(pipe, image, prompt, rays, chunks, seed, mode):
    os.environ["FASTFRAMES_MPS_ROPE"] = mode
    pipe.start(image, prompt, seed=seed, latents=chunks * 4)
    torch.mps.synchronize()
    wall_start = time.perf_counter()
    chunk_seconds, footprints, hashes = [], [], []
    digest = hashlib.sha256()
    for chunk in range(chunks):
        torch.mps.synchronize()
        start = time.perf_counter()
        frames = pipe.step(rays(chunk))
        torch.mps.synchronize()
        chunk_seconds.append(time.perf_counter() - start)
        footprints.append(memory())
        for frame in frames.cpu().numpy():
            data = frame.tobytes()
            digest.update(data)
            hashes.append(hashlib.sha256(data).hexdigest())
        del frames
    return {
        "mode": mode, "generation_s": sum(chunk_seconds), "wall_s": time.perf_counter() - wall_start,
        "chunk_s": chunk_seconds, "chunk_end_live_driver_bytes": footprints,
        "frames": len(hashes), "rgb_sha256": digest.hexdigest(),
    }, hashes


def compare_clips(args, pipe, image, prompt, rays, chunks, seed):
    # Pair 0 primes both paths all the way through a full cache. AB/BA balances order thereafter.
    eager_first = ("eager", "metal")
    reference = None
    stream = args.output.open("x") if args.output else None
    try:
        for pair in range(args.pairs + 1):
            order = eager_first if pair == 0 or pair % 2 == 1 else reversed(eager_first)
            for mode in order:
                record, hashes = measure_clip(pipe, image, prompt, rays, chunks, seed, mode)
                if len(hashes) != args.frames:
                    raise RuntimeError(f"Expected {args.frames} frames, got {len(hashes)}")
                if reference is None:
                    reference = hashes
                    record["reference_frame_sha256"] = hashes
                elif hashes != reference:
                    first = next(i for i, (a, b) in enumerate(zip(hashes, reference)) if a != b)
                    raise RuntimeError(f"Frame {first} differs from the eager warm-up in pair {pair} ({mode})")
                record.update(pair=pair, size=args.size, seed=seed)
                if stream:
                    print(json.dumps(record), file=stream, flush=True)
                print(f"pair {pair} ({'warm-up' if pair == 0 else 'measured'}), {mode}, "
                      f"{record['frames']} frames: generation {record['generation_s']:.3f} s, "
                      f"wall {record['wall_s']:.3f} s, SHA-256 {record['rgb_sha256']}", flush=True)
    finally:
        if stream:
            stream.close()


def run_model(args):
    from apps.lingbot.generate import apply_size, import_upstream, load_pipeline, read_condition

    apply_size(args, 1)
    import_upstream(args.lingbot_root)
    image, prompt, rays, chunks, seed = read_condition(args)
    pipe = load_pipeline(args, 1, 0)
    try:
        if args.mode == "block":
            pipe.start(image, prompt, seed=seed, latents=chunks * 4)
            original = pipe.dit._block_run

            def first_block(*inputs):
                pipe.dit._block_run = original
                return measure_block(original, *inputs)

            for chunk in range(chunks):
                if chunk in (0, 5):
                    pipe.dit._block_run = first_block
                pipe.step(rays(chunk))
            return

        compare_clips(args, pipe, image, prompt, rays, chunks, seed)
    finally:
        pipe.close()


@torch.inference_mode()
def main():
    from apps.lingbot.generate import add_model_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode")
    for mode in ("block", "clip"):
        model = commands.add_parser(mode)
        add_model_arguments(model)
        model.add_argument("--example", type=Path, required=True)
        model.set_defaults(frames=93, seed=0, image=None, prompt=None)
        if mode == "clip":
            model.add_argument("--pairs", type=int, choices=(4, 6), default=4,
                               help="measured AB/BA pairs after warm-up")
            model.add_argument("--output", type=Path, help="write raw timings and hashes to a new JSONL file")
    args = parser.parse_args()
    # Full-resolution model + umT5 during prompt setup needs more than the isolated benchmark.
    limit_mps_memory(0.6 if args.mode and tuple(args.size) == (832, 464) else 0.45)
    if not torch.backends.mps.is_available() or _kernel() is None:
        raise RuntimeError("An MPS device with the Metal RoPE shader is required")
    if args.mode:
        return run_model(args)
    torch.manual_seed(0)
    for size, tokens in (("416x240", 1560), ("576x320", 2880), ("832x464", 6032)):
        for batch in (1, 30):
            run(size, tokens, batch)
            release_mps_cache("mps")


if __name__ == "__main__":
    main()
