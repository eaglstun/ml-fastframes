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
| rollout (21 latent frames) | ~62 s, ~3.0 s per latent frame (~0.75 s per output frame) |
| pixel decode | ~4 s |
| peak memory footprint | ~20 GB |

The paper reports 36 ms per latent frame on a B200. The MPS rollout is compute-bound: about 2/3 of
the time is self-attention over the growing KV cache (up to 32,760 keys) and the rest is GEMMs. Those
run at ~4-13 TFLOPS bf16 here, about 100x below a B200. bf16 is slightly faster than fp16 on M4.
Interactive latency isn't reachable on this hardware; batch T2V works fine.

## What was changed (see `git diff`)

| Problem on Mac | Fix |
|---|---|
| `device="cuda"` hardcoded in the loader, pipeline, and autocast | `fastframes/device.py`: `$FASTFRAMES_DEVICE`, else cuda > mps > cpu |
| `sdpa_kernel(CUDNN_ATTENTION)` | Used only on CUDA |
| MPS has no float64/complex128 (timestep sinusoids, scheduler x0, RoPE) | float64 math runs on CPU (bit-exact); RoPE uses a cached complex128 table applied as FP32 rotations (within 1 bf16 ulp, 99.998% identical) |
| MPS SDPA caches one MPSGraph **plus its FP32 score buffer** per shape; the KV prefix grows every frame, so 21 shapes retained ~56-68 GB that `empty_cache()` can't free | Attention is called one head at a time on MPS: ~6 GB retained, and ~12% faster |
| Allocator hoards odd-sized buffers | `synchronize()` + `empty_cache()` after each frame on MPS |
| umT5 (11 GB) stays resident during the rollout | `generate.py` encodes all prompts first, then moves T5 to CPU |
| T5 built with random FP32 init (48 s, 22 GB RAM) before loading bf16 weights | Built on the meta device and assigned the checkpoint (same result) |
| OpenH264 binary is linux64-only (ctypes) | Non-Linux `save_video` uses ffmpeg/libx264, BT.601 limited range, faststart |

Not ported: training (NCCL, Triton flex_attention, CUDA RNG state), the latency benchmark, and
FastFrames-L (14B: ~28 GB DiT + ~26 GB KV cache + T5 does not fit in 64 GB without offloading).
The LingBot demo was not tried.
