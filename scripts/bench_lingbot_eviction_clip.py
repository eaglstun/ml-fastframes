"""Compare synchronized LingBot rollouts with the baseline and Metal cache shifts."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from apps.lingbot import pipeline  # noqa: E402
from apps.lingbot.generate import (add_model_arguments, apply_size, import_upstream, load_pipeline,  # noqa: E402
                                   read_condition)
from fastframes.device import limit_mps_memory  # noqa: E402


def baseline_method(pipe):
    # Snapshot of FastFramesLingBot._evict at 974f473, with the current frame size.
    from apps.lingbot.pipeline import CACHE_FRAMES, CHUNK, FRAME_TOKENS, SINK_FRAMES

    def _evict(self, chunk):
        """Before writing ``chunk``: drop the oldest non-sink frames of a full cache (one shift per chunk)."""
        previous = min(CHUNK * chunk, CACHE_FRAMES)
        evicted = max(0, previous + CHUNK - CACHE_FRAMES)
        if evicted:
            sink, shift = SINK_FRAMES * FRAME_TOKENS, evicted * FRAME_TOKENS
            rolled = (previous - evicted - SINK_FRAMES) * FRAME_TOKENS
            # Contiguous per-layer copies in blocks of ``shift`` tokens, moving left: no block reads a
            # region that an earlier block overwrote, so no temporary copy is needed.
            for cache in (self.cache_k, self.cache_v):
                for layer in cache:
                    for start in range(sink, sink + rolled, shift):
                        end = min(start + shift, sink + rolled)
                        layer[:, start:end].copy_(layer[:, start + shift:end + shift])
            if self.assist is not None:
                self.assist.evict(sink, shift, rolled)

    return MethodType(_evict, pipe)


def measure_clip(pipe, image, prompt, rays, chunks, seed, mode):
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
        footprints.append((torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()))
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
    order_ab = ("baseline", "metal")
    methods = {"baseline": baseline_method(pipe), "metal": pipe._evict}
    reference = None
    stream = args.output.open("x")
    try:
        for pair in range(args.pairs + 1):
            order = order_ab if pair == 0 or pair % 2 == 1 else reversed(order_ab)
            for mode in order:
                pipe._evict = methods[mode]
                record, hashes = measure_clip(pipe, image, prompt, rays, chunks, seed, mode)
                if len(hashes) != args.frames:
                    raise RuntimeError(f"Expected {args.frames} frames, got {len(hashes)}")
                if reference is None:
                    reference = hashes
                    record["reference_frame_sha256"] = hashes
                elif hashes != reference:
                    first = next(i for i, (a, b) in enumerate(zip(hashes, reference)) if a != b)
                    raise RuntimeError(f"Frame {first} differs from the baseline warm-up in pair {pair} ({mode})")
                record.update(pair=pair, size=args.size, seed=seed)
                print(json.dumps(record), file=stream, flush=True)
                print(f"pair {pair} ({'warm-up' if pair == 0 else 'measured'}), {mode}, "
                      f"{record['frames']} frames: generation {record['generation_s']:.3f} s, "
                      f"wall {record['wall_s']:.3f} s, SHA-256 {record['rgb_sha256']}", flush=True)
    finally:
        pipe._evict = methods["metal"]
        stream.close()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True, help="new JSONL file for timings and reference hashes")
    parser.set_defaults(frames=93, seed=0, image=None, prompt=None)
    args = parser.parse_args()
    if args.pairs < 2 or args.pairs % 2:
        parser.error("--pairs must be a positive even number")
    limit_mps_memory(0.45)
    if not torch.backends.mps.is_available() or pipeline._mps_eviction_kernel() is None:
        raise RuntimeError("Requires MPS with the Metal eviction kernel")
    apply_size(args, 1)
    import_upstream(args.lingbot_root)
    image, prompt, rays, chunks, seed = read_condition(args)
    pipe = load_pipeline(args, 1, 0)
    try:
        if not pipeline._mps_eviction_supported(pipe.cache_k, pipe.cache_v):
            raise RuntimeError("The Metal eviction kernel does not support this cache layout")
        compare_clips(args, pipe, image, prompt, rays, chunks, seed)
    finally:
        pipe.close()


if __name__ == "__main__":
    main()
