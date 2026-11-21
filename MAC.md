# FastFrames on Apple Silicon (MPS)

Tested on an M4 Max (64 GB), macOS, PyTorch 2.9.1: FastFrames-S T2V (`scripts/generate.py`) and the
LingBot camera-control demo ([below](#lingbot-camera-control-interactive-demo)).
Output is the same 832x480, 81-frame, 16 fps video as on CUDA.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python "torch==2.9.1" "torchvision==0.24.1" "transformers>=5.10" \
    safetensors einops ftfy regex pyyaml numpy huggingface_hub
HF=.venv/bin/hf
$HF download apple/FastFrames-S --include 'fastframes-s/*' --local-dir checkpoints
$HF download zhuhz22/Causal-Forcing causal-forcing++/framewise-2step.pt --local-dir checkpoints
$HF download Wan-AI/Wan2.1-T2V-1.3B --exclude 'diffusion_pytorch_model*' --local-dir checkpoints/Wan2.1-T2V-1.3B
```

Do not `uv sync`: pyproject pins the `+cu128` wheels. ffmpeg with libx264 must be on PATH (`brew install ffmpeg`).

```bash
.venv/bin/python scripts/generate.py --model ff-s --checkpoint-dir checkpoints \
    --prompt "A corgi surfing a wave at sunset" --seed 0 --output outputs/corgi.mp4
```

## Performance (M4 Max)

| | |
|---|---|
| load | ~5 s |
| text encode | ~0.5 s |
| rollout + decode + encode, per video | ~45-49 s (was ~63-78 s with MPS SDPA, see below) |
| peak memory footprint | ~18 GB |

The paper reports 36 ms per latent frame on a B200. The MPS rollout is compute-bound: with PyTorch's
SDPA about 2/3 of the time was self-attention over the growing KV cache (up to 32,760 keys) and the
rest is GEMMs, which run at ~13 TFLOPS bf16 here, about 100x below a B200. bf16 is slightly faster
than fp16 on M4. Interactive latency isn't reachable on this hardware; batch T2V works fine.

## Metal flash-attention kernel

On MPS, attention runs in a custom Metal compute kernel (`fastframes/wan/mps_flash_attention.py`,
compiled at first use with `torch.mps.compile_shader`; no build step). `FASTFRAMES_MPS_ATTN=sdpa`
switches back to PyTorch SDPA (one head per call); `flash` is the default.

- FlashAttention-style: each SIMD-group owns 16 query rows and walks the keys 24 at a time with an
  online softmax, so the `[Lq, Lk]` score matrix is never materialised. `Q K^T` and `P V` use the
  GPU's 8x8 `simdgroup_matrix` multiply-accumulate with BF16 operand tiles and FP32 accumulators
  (BF16 in, BF16 out; the mixed form is bit-identical to widening the operands first).
- Operand tiles are loaded three steps before the multiply that uses them (software pipelining,
  `_LOOKAHEAD`). A load placed right before its use stalls the SIMD-group for the load latency:
  hoisting the loads took the kernel from 10.6 to 12.5 TFLOPS with bit-identical output. The softmax
  is base 2 (`log2(e)` folded into the scale, `exp2`), worth another ~2%.
- Shapes are runtime arguments: one pipeline serves every KV length (and the L=512 text
  cross-attention), so nothing is cached per shape. It reads and writes the model's `[B, L, H, D]`
  layout directly, so there are no transpose copies.
- The kernel source is generated as straight-line code (every 8x8 tile a named variable, every
  offset a literal). The same algorithm written with arrays of matrices in loops was 2-4x slower.
  16 query rows per SIMD-group is the limit: 24 or 32 fall out of registers and run ~1.6x slower.
  8 rows multiply faster but double the SIMD-groups streaming K/V and end up memory-bound (7 TFLOPS).
- Tried without success: staging K/V tiles in threadgroup memory for 8 SIMD-groups (the copy and
  the two barriers per step cost what the sharing saves), `simdgroup_load` (1.4x slower than typed
  per-thread loads), Q held in registers, permuting dims so one 8-byte load feeds two tiles,
  rescaling the running output only when a row maximum changes, larger key blocks. Putting heads
  on the fast dispatch axis is +2% at 1,560 query rows but -10% at 6,032, so it stays as is.
- It relies on the lane-to-element layout of `simdgroup_matrix` (the trick MLX's steel kernels
  use), which the MSL spec doesn't guarantee. The layout is probed at first use and `attention()`
  falls back to SDPA if it doesn't hold. Credits are in the file header.

Measured on the M4 Max (`scripts/bench_mps_attention.py`; q `[1, 1560, 12, 128]`, warm,
synchronized, median):

| keys | flash | SDPA per head |
|---|---|---|
| 512 (cross-attention) | 1.1 ms | 1.8 ms |
| 1,560 | 1.5 ms | 3.0 ms |
| 15,600 | 11.8 ms | 28.5 ms |
| 32,760 | 24.9 ms | 64.2 ms |
| sum over the 21 KV lengths | **275 ms** | 673 ms |

That is ~12.6 TFLOPS for the whole attention (13.0 at LingBot's 6,032 x 27,144), against ~5 for SDPA
and 13-14.7 for a bare MPS GEMM; a loop of nothing but register-to-register 8x8 multiplies reaches ~14.
Driver memory stays flat across all 22 shapes (1.03 GB in the benchmark, which is the test tensors).
Against an FP32 CPU reference the kernel is closer than MPS SDPA, because it accumulates in FP32:
on tensors captured from the rollout at 32,760 keys, max/mean abs error 7.8e-3 / 6.7e-4 (output
rms 0.61) vs 7.9e-2 / 1.8e-3 for SDPA; on random inputs 1.2e-4 / 1.0e-5 vs 3.6e-4 / 3.2e-5.

End to end (corgi prompt, seed 0, three runs each, same session): 45.3 / 48.4 / 48.6 s with the
kernel, 78.0 / 69.8 / 74.8 s with SDPA. The video is not bit-identical to the SDPA one (the
autoregressive rollout amplifies rounding differences: frame PSNR 35 dB at frame 0, ~22-24 dB by
frames 40 and 80), but it is the same scene at the same quality, and is deterministic run to run.

## LingBot camera control (interactive demo)

The LingBot-World v2 1.3B app (`apps/lingbot/`: image + prompt in, a world you steer with
WASD/arrow keys out) also runs on MPS, both the browser demo and `generate.py`. It runs eagerly:
torch.compile and CUDA graphs are switched off automatically off CUDA. Setup on top of the
FastFrames-S venv above (weights are CC BY-NC-SA 4.0, non-commercial):

```bash
uv pip install -p .venv/bin/python "diffusers==0.31.0" easydict imageio scipy tqdm \
    "websockets>=15,<17" "Pillow>=12.3"
mkdir -p upstream
git clone https://github.com/robbyant/lingbot-world-v2 upstream/lingbot-world-v2
git -C upstream/lingbot-world-v2 checkout 1895d300d8ac936401689b26389f51cbd36530eb
$HF download apple/FastFrames-S --include 'lingbot-1.3b/*' --local-dir checkpoints
$HF download robbyant/lingbot-world-v2-1.3b-causal-fast \
    --local-dir checkpoints/lingbot-world-v2-1.3b-causal-fast/transformers
$HF download robbyant/lingbot-world-v2-14b-causal-fast \
    --include 'models_t5_umt5-xxl-enc-bf16.pth' --include 'Wan2.1_VAE.pth' --include 'google/umt5-xxl/*' \
    --local-dir checkpoints/lingbot-world-v2-14b-causal-fast

# browser demo at http://127.0.0.1:8000
.venv/bin/python apps/lingbot/demo.py --gpus 1 --lingbot-root upstream/lingbot-world-v2
# or a video (2 chunks = 29 frames; --frames 961 for the full 60 chunks)
.venv/bin/python apps/lingbot/generate.py --model lingbot-1.3b --lingbot-root upstream/lingbot-world-v2 \
    --example upstream/lingbot-world-v2/examples/00 --frames 29 --output outputs/lingbot.mp4
```

Measured on the M4 Max with the demo server driven by a scripted websocket client (scene 00,
idle GPU). Each turn is one chunk: 16 frames, 1 s of 16 fps video at 832x464.

| | first session after server start | warm session (same server) |
|---|---|---|
| world ready | 16.0 s | 0.0 s (scene reused) |
| chunk 0 (13 frames) | 19.6 s | 12.5 s |
| turn 1 | 24.2 s | 14.2 s |
| turn 2 | 26.3 s | 18.1 s |
| turn 3 | 29.3 s | 22.0 s |
| turns 4-8 | 31.2-34.3 s | 23.5-24.0 s |

The first session pays one-time Metal shader compiles and allocations for new shapes. Turns
get slower while the rolling KV cache fills (it holds 18 latent frames, full after about five
chunks) and then hold at ~24 s, in line with `latency.py`'s 25.9 s per full-cache chunk. That's
~12-24x slower than real time: you explore shot by shot rather than play. Peak memory went from
83 GB (upstream conditioning) to 43 GB on a 2-chunk `generate.py` run; since the pipeline releases
cached blocks after each chunk, `latency.py` holds a flat ~23 GB of driver memory.

What changed for LingBot:

| Problem on Mac | Fix |
|---|---|
| Upstream `wan/modules/t5.py` evaluates `torch.cuda.current_device()` as a default argument at import time | `upstream.py` stubs it to `"cpu"` only during that import, without CUDA; the clone isn't modified |
| The image condition was VAE-encoded for the whole 60-chunk horizon before the first frame: minutes of work, ~40 GB extra, and on MPS the **last latent comes out wrong** (0.4 max abs error vs a CPU reference) | `conditioning.LazyImageCondition` runs the same causal encoder (same feature cache) one latent at a time, just before each chunk needs it; it matches the CPU reference to 7e-6 |
| umT5 built with random FP32 init: 82 s to load | Meta-device build + `load_state_dict(assign=True)` from an mmapped checkpoint: 0.9 s, bit-identical embeddings |
| complex128 RoPE table, float64 timestep sinusoids and x0 update | Computed on CPU, results moved to MPS (as for FastFrames-S) |
| Attention uses split-KV SDPA tuned for CUDA | On MPS it calls `fastframes.wan.attention`, i.e. the Metal flash kernel |
| `torch.autocast("cuda")`, CUDA streams for KV-refinement head groups and decoder warm-up, `torch.cuda.synchronize` | Autocast on the tensor's device type; head groups run sequentially and warm-ups are skipped off CUDA; `fastframes.device.synchronize` |
| torch.compile (inductor max-autotune), CUDA graphs, demo warm-up passes that only exist to trigger them | Eager off CUDA (`generate.py` picks it automatically; the demo worker passes `compile=False, graphs=False`) |
| `from_pretrained(device=None)` left the refiner/decoder on CPU | The device is resolved once at the top (`default_device()`) |
| Demo memory stats call `torch.cuda` | MPS reports `current_allocated_memory` / `driver_allocated_memory` |

## Latency benchmarks

`evaluation/latency/benchmark.py` (component profiles, Table 1) and `apps/lingbot/latency.py`
(Fig. 5) run on one MPS device. Off CUDA the profilers time synchronized wall clock instead of
CUDA events and run eager (no max-autotune, CUDA graphs, CUDA streams or Triton); attention is the
Metal flash kernel and the split-KV setting is ignored. The protocol is otherwise upstream's.

```bash
.venv/bin/python evaluation/latency/benchmark.py ff-s --output-dir outputs/latency/ff-s_mps
.venv/bin/python apps/lingbot/latency.py --model lingbot-1.3b --lingbot-root upstream/lingbot-world-v2 \
    --frames 253 --min-samples 16 --output outputs/fig5_mps/ff_1.3b.json
```

FastFrames-S, one step (one latent frame, 4 video frames) with a full 21-frame cache, M4 Max:

| Component | M4 Max | B200 (README) |
|---|---:|---:|
| DiT evaluation, t=1000 | 1340.3 ms | 16.1 ms |
| DiT evaluation, t=833 | 1676.8 ms | 16.2 ms |
| KV refinement | 119.6 ms | 2.2 ms |
| Pixel decoder | 57.4 ms | 0.9 ms |
| **Total** | **3194 ms** | **35.4 ms** |

The two DiT rows do the same work; the second is slower because `benchmark.py` runs the
components back to back and the chip is hotter by then. Profiled alone after a 5-minute
cool-down, t=833 takes 1341 ms (then t=1000, 2 minutes later, 1406 ms). On a cool chip a step is
~2.9 s (2 x 1.34 s + 0.12 s + 0.06 s). Within one component's 60 s window the timing drifts by
only ~1%.

For comparison, the Wan VAE decode that the baselines use instead of the pixel decoder takes
1.5 s per latent frame here with the conv2d path below (4.9 s with PyTorch's MPS `conv3d`; two
streamed passes each, back to back), ~26x the pixel decoder.

LingBot 1.3B, one chunk (16 frames, 1 s of video) with a full 18-frame cache: **25.9 s** median
(16 chunks, range 25.3-26.4 s) against 220.7 ms on a B200. The run uses 16-chunk videos
(`--frames 253`) and 16 samples instead of upstream's 60-chunk videos and 30 samples: a
60-chunk pass takes ~25 minutes here. Eager mode has nothing to compile, so one warm-up pass
is enough off CUDA.

Measured on an idle machine (no other GPU jobs, Photos analysis finished). A first run right
after a reboot, with Photos' `mediaanalysisd` busy, gave 4013 ms per FastFrames-S step and 27.1 s
per LingBot chunk: background GPU work inflates these by up to ~30%.

- **Memory cap.** By default MPS may allocate 1.7x its recommended working set and then swaps.
  An early run of these profilers outgrew memory alongside a running demo server and stalled
  the Mac into a watchdog panic (reboot). On MPS the profilers now cap PyTorch with
  `torch.mps.set_per_process_memory_fraction`: 0.6 of the working set (~31 GB on 64 GB) for
  `evaluation/latency`, 0.75 (~39 GB) for LingBot. Too little fails with a clean out-of-memory
  error; raise it with `FASTFRAMES_MPS_MEMORY_FRACTION`. Peak driver memory measured: DiT 9.8 GB,
  KV refinement 18.1 GB, pixel decoder 7.8 GB, LingBot 23 GB.
- **No whole-clip VAE decode.** The VAE profile's untimed warm-up decoded all 21 latents at once;
  off CUDA it's skipped (the streamed warm-up passes remain).
- **LingBot frees cached memory.** Nothing in the LingBot pipeline returned allocator blocks: the
  11 GB umT5 moved back to the CPU but its GPU blocks stayed reserved, and per-chunk temporaries
  piled up (34 GB after chunk 0, past 45 GB in chunk 1). On MPS the pipeline now calls
  `synchronize()` + `empty_cache()` after the prompt is encoded and after each chunk: 23 GB, flat.
  This also applies to the demo (its turn timings above include it).
- **Numerical checks.** The pixel-decoder profile's streaming output is bit-identical to the
  clip-level decoder. The refinement profile's fast path differs from the release refinement by
  1.6% / 1.9% relative L2 (K / V); a CPU run gives 1.5% / 1.8%, so that's the BF16 fast path
  itself, not MPS.

Not ported: the multi-GPU rows (`--gpus 8`, NVLink symmetric memory), the 720p/14B rows, and the
Causal Forcing++ / Self-Forcing baseline DiTs (`baselines.py`, needs their upstream clones).

## Performance work: VAE convolutions, attention pipelining, condition prefetch

Profiling a full-cache LingBot chunk and a FastFrames-S DiT step on the M4 Max found three wins.
Background GPU work (Photos analysis) was running during most of these runs, so each number below
is an A/B measured back to back or interleaved, not an absolute benchmark.

- **Wan VAE 3D convolutions as per-tap `conv2d`** (`fastframes/wan/mps_conv3d.py`). MPS `conv3d`
  runs the VAE's 3x3x3 layers at ~3.3 TFLOPS in every dtype; the same convolution written as the
  sum over its temporal taps of batched `conv2d` calls runs at 13-16 TFLOPS in FP32 (same values to
  ~1e-6 relative L2). Only `_conv_forward` of the Wan `CausalConv3d` classes is replaced (ours, and
  the LingBot clone's at import), so causal padding and feature caches are unchanged.
  `FASTFRAMES_MPS_CONV3D=native` restores `conv3d`.
  - LingBot image condition: 3.19 s -> 0.95 s per black-frame latent, 4 latents per chunk. With
    the kernel change below, a new world's first demo session went from 24.2 / 26.3 / 29.3 / 31.2 s
    to 14.2 / 17.0 / 20.4 / 21.7 s for turns 1-4 (no prefetch).
  - Wan VAE decode (baseline rows): 4.9 s -> 1.5 s per latent frame.
  - Found on the way: MPS `conv2d` silently returns wrong results for a non-contiguous weight view
    such as `weight[:, :, k]` (PyTorch 2.9.1). The per-tap weights are made contiguous once.
- **Software-pipelined loads in the flash-attention kernel** (by Fable; details in the kernel
  section above): 10.6 -> 12.7 TFLOPS at 1,560 x 32,760 and 11.8 -> 13.0 at LingBot's 6,032 x 27,144,
  same error. Attention is ~70% of a FastFrames-S DiT step: one DiT evaluation went from 1294 to
  1147 ms in a quiet run (-11%) and by 6% on average in four back-to-back pairs under load
  (1325/1286, 1353/1249, 1554/1441, 1584/1478 ms). `latency.py`'s
  LingBot chunk went from 25.9 s to 24.2 s (its timed chunks reuse a cached image condition, so it
  doesn't see the VAE change).
- **Condition prefetch in the demo** (`FastFramesLingBot.prefetch_condition`). The next chunk's VAE
  condition doesn't depend on the camera action, so on MPS the demo worker encodes it right after
  replying, while the player looks at the frames. Alternating full-cache chunks in one process:
  26.3 s per step with the encode inline, 21.9 s prefetched. `LINGBOT_DEMO_PREFETCH=0` disables it.
  The win shows when the player takes more than ~4 s between turns.

What's left in a full-cache LingBot chunk is mostly attention (~60%) and GEMMs, which run at
12.4-14.2 TFLOPS (MPS's own matmul), close to what this GPU does.

## What was changed for FastFrames-S (see `git diff`)

| Problem on Mac | Fix |
|---|---|
| `device="cuda"` hardcoded in the loader, pipeline, and autocast | `fastframes/device.py`: `$FASTFRAMES_DEVICE`, else cuda > mps > cpu |
| `sdpa_kernel(CUDNN_ATTENTION)` | Used only on CUDA |
| MPS has no float64/complex128 (timestep sinusoids, scheduler x0, RoPE) | float64 math runs on CPU (bit-exact); RoPE uses a cached complex128 table applied as FP32 rotations (within 1 bf16 ulp, 99.998% identical) |
| MPS SDPA caches one MPSGraph **plus its FP32 score buffer** per shape; the KV prefix grows every frame, so 21 shapes retained ~56-68 GB that `empty_cache()` can't free | Default: the Metal flash-attention kernel above (no per-shape state). SDPA fallback: one head per call, ~6 GB retained |
| Allocator hoards odd-sized buffers | `synchronize()` + `empty_cache()` after each frame on MPS |
| umT5 (11 GB) stays resident during the rollout | `generate.py` encodes all prompts first, then moves T5 to CPU |
| T5 built with random FP32 init (48 s, 22 GB RAM) before loading bf16 weights | Built on the meta device and assigned the checkpoint (same result) |
| OpenH264 binary is linux64-only (ctypes) | Non-Linux `save_video` uses ffmpeg/libx264, BT.601 limited range, faststart |

Not ported: training (NCCL, Triton flex_attention, CUDA RNG state), FastFrames-L (14B: ~28 GB DiT
+ ~26 GB KV cache + T5 does not fit in 64 GB without offloading), multi-device LingBot
(`--gpus 8`, torchrun), and the multi-GPU and baseline latency rows.
