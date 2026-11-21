# AGENTS.md

Notes for coding agents working on this fork of
[apple-aiml-research/ml-fastframes](https://github.com/apple-aiml-research/ml-fastframes).
The fork adds Apple Silicon (MPS) inference for FastFrames-S text-to-video and the LingBot
camera-control demo (`apps/lingbot/`). Read `MAC.md` for
the full change list and measurements. Everything outside inference is upstream code that still
targets Linux + NVIDIA.

## Layout

- `fastframes/pipeline.py`: `FastFrames`, the per-frame autoregressive rollout (DiT → KV
  refinement → pixel decoder). Inference entry point.
- `fastframes/wan/`: the Wan2.1 backbone (`causal_model.py` is the frame-wise KV-cache model,
  `attention.py` is the attention used at inference, `t5.py`/`text_encoder.py` is umT5-XXL).
- `fastframes/models/`: KV refinement network and pixel decoder (these replace the Wan VAE at
  inference).
- `fastframes/device.py`: device selection and float64 placement (added by this fork).
- `fastframes/training/`: training only, CUDA/NCCL/Triton. Not ported.
- `scripts/generate.py`: T2V CLI. `MAC.md`: the Mac port write-up.
- `apps/lingbot/`: LingBot-World v2 camera control. `generate.py` (video), `demo.py` +
  `demo_worker.py` (browser demo, websocket at `/ws`), `pipeline.py` (chunked rollout),
  `upstream.py` (imports the pinned clone in `upstream/lingbot-world-v2`; treat that clone as
  read-only third-party code and patch around it here).
- `apps/lingbot/web/`: TypeScript source of the demo page (`src/app.ts`). It compiles to
  `apps/lingbot/static/app.js`, which is committed: edit the `.ts`, then `npm run build` and
  `npm run verify` there. `npm run e2e` drives a running demo in headless Chrome.

## Environment

- **Mac:** use `.venv` built with `uv venv` + `uv pip install` (see the README's _Apple Silicon_
  section). **Never run `uv sync` on a Mac**: `pyproject.toml` pins the `+cu128` torch wheels.
  Run tools with `.venv/bin/python` and `.venv/bin/hf`. pyenv shims may resolve `hf` to some
  other Python.
- **Linux/CUDA:** upstream instructions (`uv sync`, `uv run ...`) apply unchanged.
- `checkpoints/`, `outputs/`, and `.venv/` are gitignored. Never commit weights or videos.
- Device override: `FASTFRAMES_DEVICE=cpu|mps|cuda`.

## Run / verify

```bash
.venv/bin/python scripts/generate.py --model ff-s --checkpoint-dir checkpoints \
    --prompt "A corgi surfing a wave at sunset" --seed 0 --output outputs/corgi.mp4
```

Expected on an M4 Max: ~45-50 s per video (flash attention; ~65-85 s with `FASTFRAMES_MPS_ATTN=sdpa`), an 81-frame 832×480 16 fps H.264 file, ~18 GB peak.
There are no unit tests for the inference path. Verify a change by:

1. Generating with a fixed prompt and seed, before and after the change.
2. Comparing frames (e.g. extract frames 0/40/80 with ffmpeg). A change meant to be
   numerically neutral should produce pixel-identical video on the same device. Anything else
   gets compounded over 21 autoregressive frames, so judge those changes by looking at the
   frames, not by PSNR.
3. Timing the rollout with `torch.mps.synchronize()` around it, and checking memory with
   `torch.mps.driver_allocated_memory()` (live tensors alone hide the problem below).

## MPS invariants (each one cost real debugging time)

- **No float64 or complex128 tensors on MPS.** Do float64 math on CPU via
  `device.f64_device()` and move the results back. Use `x.to(cpu).to(torch.float64)` in that
  order: `.to(device, dtype)` casts before moving and fails on MPS. The complex128 RoPE table
  (`model.freqs`) deliberately stays on CPU under MPS.
- **Don't call SDPA with all heads at once on MPS.** MPS caches one MPSGraph _plus its FP32
  score buffer_ for every input shape, and `empty_cache()` cannot free it. The self-attention
  key length grows every frame (21 shapes), so all-heads calls retain ~60 GB. Any attention
  path must keep memory flat across key lengths.
- **Attention on MPS defaults to the Metal flash kernel** (`fastframes/wan/mps_flash_attention.py`,
  compiled at runtime with `torch.mps.compile_shader`). It's ~2x faster than SDPA and closer to an
  fp32 reference, with no per-shape state. `FASTFRAMES_MPS_ATTN=sdpa` switches back to per-head
  SDPA, which is also the automatic fallback for causal/dropout/tiny inputs or if the lane-layout
  probe fails. The kernel relies on the simdgroup-matrix lane layout (unspecified by MSL, probed
  at first use) and on straight-line generated source: the loop/array version was ~2.5x slower.
  Benchmark changes with `scripts/bench_mps_attention.py`.
- **The per-frame `synchronize()` + `empty_cache()`** in `FastFrames.rollout` keeps the
  allocator from hoarding odd-sized buffers. `empty_cache()` without the sync frees nothing
  while command buffers are still in flight.
- **Unified memory is shared** between CPU and GPU. umT5 (11 GB) gets moved to CPU after
  prompts are encoded, and it's built on the meta device (random FP32 init cost 48 s and 22 GB).
- **LingBot conditioning is encoded lazily** (`conditioning.LazyImageCondition`), one latent per
  chunk. Encoding the whole horizon up front on MPS corrupts the final latent (0.4 max abs error
  vs CPU) and costs minutes plus ~40 GB. Check any change here against a CPU encode.
- **LingBot runs eager off CUDA.** torch.compile, CUDA graphs, CUDA streams, and the warm-up
  passes that feed them are gated on `device.type == "cuda"`. Verify the demo with a scripted
  websocket client (`start`, then `action` / `displayed` messages) and time a _second_ session:
  the first one includes Metal shader compiles.
- **Never let an MPS job outgrow memory.** MPS swaps past 1.7x its working set by default, and a
  run that outgrew memory stalled the Mac into a watchdog panic (reboot). Long or large jobs
  call `device.limit_mps_memory()` first; the latency profilers do. Shut down other GPU jobs (a
  leftover demo server holds 13+ GB) before benchmarking. Pipelines on MPS release allocator
  blocks with `device.release_mps_cache()` (sync + `empty_cache()`) per frame/chunk.
- **Latency profiling** (`evaluation/latency`, `apps/lingbot/latency.py`) runs on one MPS device:
  wall-clock timing, eager, Metal flash attention. Numbers are in `MAC.md#latency-benchmarks`.
- **MPS `conv3d` is ~4x slower than per-tap `conv2d`** at the Wan VAE's shapes. Wan `CausalConv3d`
  classes go through `fastframes/wan/mps_conv3d.py` on MPS (`FASTFRAMES_MPS_CONV3D=native` for A/B).
  MPS `conv2d` gives **wrong results for non-contiguous weight views** (`w[:, :, k]`): always
  pass contiguous weights.
- **Benchmark A/B, not absolutes, when the Mac is busy.** Photos' `mediaanalysisd` and
  VideoToolbox use the GPU after reboots and imports and move timings by 10-30%. Compare
  variants back to back or interleaved in one process.
- Measured dead ends: `fast_ops` (fused QKV) is slower on MPS, and fp16 is slower than bf16 on
  M4. cuDNN SDPA is CUDA-only; on other devices `attention()` falls through to plain SDPA.

## Conventions

- Keep upstream behaviour on CUDA unchanged: gate Mac-specific code on the device type (MPS) or on
  `sys.platform`, not by editing the CUDA path.
- Keep changes small and commented with the _why_, since upstream may be merged in again.
- Line length is 120 (ruff config in `pyproject.toml`).
- Upstream doesn't accept issues or PRs. Mac-port discussion lives at
  https://huggingface.co/apple/FastFrames-S/discussions/1
