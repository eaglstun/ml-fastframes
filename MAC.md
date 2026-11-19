# FastFrames on Apple Silicon (MPS)

Tested on an M4 Max (64 GB), macOS, PyTorch 2.9.1, FastFrames-S T2V (`scripts/generate.py`).
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
  GPU's 8x8 `simdgroup_matrix` multiply-accumulate in FP32 (BF16 in, BF16 out).
- Shapes are runtime arguments: one pipeline serves every KV length (and the L=512 text
  cross-attention), so nothing is cached per shape. It reads and writes the model's `[B, L, H, D]`
  layout directly, so there are no transpose copies.
- The kernel source is generated as straight-line code (every 8x8 tile a named variable, every
  offset a literal). The same algorithm written with arrays of matrices in loops was 2-4x slower.
  16 query rows per SIMD-group is the limit: 24 or 32 fall out of registers and run ~1.6x slower.
- It relies on the lane-to-element layout of `simdgroup_matrix` (the trick MLX's steel kernels
  use), which the MSL spec doesn't guarantee. The layout is probed at first use and `attention()`
  falls back to SDPA if it doesn't hold. Credits are in the file header.

Measured on the M4 Max (`scripts/bench_mps_attention.py`; q `[1, 1560, 12, 128]`, warm,
synchronized, median):

| keys | flash | SDPA per head |
|---|---|---|
| 512 (cross-attention) | 0.6 ms | 2.0 ms |
| 1,560 | 1.6 ms | 3.5 ms |
| 15,600 | 14.0 ms | 28.8 ms |
| 32,760 | 29.7 ms | 64.9 ms |
| sum over the 21 KV lengths | **327 ms** | 684 ms |

That is ~10.6 TFLOPS for the whole attention, against ~5 for SDPA and 13-14.7 for a bare MPS GEMM.
Driver memory stays flat across all 22 shapes (1.03 GB in the benchmark, which is the test tensors).
Against an FP32 CPU reference the kernel is closer than MPS SDPA, because it accumulates in FP32:
on tensors captured from the rollout at 32,760 keys, max/mean abs error 7.8e-3 / 6.7e-4 (output
rms 0.61) vs 7.9e-2 / 1.8e-3 for SDPA; on random inputs 1.2e-4 / 1.0e-5 vs 3.6e-4 / 3.2e-5.

End to end (corgi prompt, seed 0, three runs each, same session): 45.3 / 48.4 / 48.6 s with the
kernel, 78.0 / 69.8 / 74.8 s with SDPA. The video is not bit-identical to the SDPA one (the
autoregressive rollout amplifies rounding differences: frame PSNR 35 dB at frame 0, ~22-24 dB by
frames 40 and 80), but it is the same scene at the same quality, and is deterministic run to run.

## What was changed (see `git diff`)

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

Not ported: training (NCCL, Triton flex_attention, CUDA RNG state), the latency benchmark, and
FastFrames-L (14B: ~28 GB DiT + ~26 GB KV cache + T5 does not fit in 64 GB without offloading).
The LingBot demo was not tried.
