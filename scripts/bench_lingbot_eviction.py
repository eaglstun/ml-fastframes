"""Time one synchronized LingBot KV eviction on MPS, using the real 1.3B cache dimensions."""

import argparse
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apps.lingbot import pipeline  # noqa: E402
from apps.lingbot.conditioning import set_size  # noqa: E402
from fastframes.device import limit_mps_memory, release_mps_cache  # noqa: E402


def allocations():
    return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()


def measure(pipe, chunk, batched):
    kernel = pipeline._mps_eviction_kernel() if batched else None
    with patch.object(pipeline, "_mps_eviction_kernel", lambda: kernel):
        torch.mps.synchronize()
        start = time.perf_counter()
        pipe._evict(chunk)
        torch.mps.synchronize()
        return (time.perf_counter() - start) * 1000


def benchmark(width, height, repeats):
    set_size(width, height)
    pipe = pipeline.FastFramesLingBot.__new__(pipeline.FastFramesLingBot)
    shape = (30, 1, pipeline.CACHE_FRAMES * pipeline.FRAME_TOKENS, 12, 128)
    pipe.cache_k = torch.zeros(shape, device="mps", dtype=torch.bfloat16)
    pipe.cache_v = torch.zeros_like(pipe.cache_k)
    pipe.assist = None
    for chunk in (3, 4, 5):  # growth, first overflow, then steady state
        timings = {False: [], True: []}
        for batched in timings:
            release_mps_cache("mps")
            before = allocations()
            measure(pipe, chunk, batched)
            print(f"{width}x{height} chunk={chunk} batched={batched} tensor/driver bytes {before} -> {allocations()}")
        for sample in range(repeats):
            for batched in ((False, True) if sample % 2 == 0 else (True, False)):
                timings[batched].append(measure(pipe, chunk, batched))
        for batched, samples in timings.items():
            print(f"  batched={batched} median {statistics.median(samples):.3f} ms;"
                  f" samples (ms) {' '.join(f'{t:.3f}' for t in samples)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    limit_mps_memory(0.35)
    if not torch.backends.mps.is_available() or pipeline._mps_eviction_kernel() is None:
        raise SystemExit("Requires MPS with runtime Metal shader compilation")
    for size in ((416, 240), (576, 320), (832, 464)):
        benchmark(*size, args.repeats)
        release_mps_cache("mps")
