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

|                                      |                                                  |
| ------------------------------------ | ------------------------------------------------ |
| load                                 | ~5 s                                             |
| text encode                          | ~0.5 s                                           |
| rollout + decode + encode, per video | ~45-49 s (was ~63-78 s with MPS SDPA, see below) |
| peak memory footprint                | ~18 GB                                           |

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

| keys                       | flash      | SDPA per head |
| -------------------------- | ---------- | ------------- |
| 512 (cross-attention)      | 1.1 ms     | 1.8 ms        |
| 1,560                      | 1.5 ms     | 3.0 ms        |
| 15,600                     | 11.8 ms    | 28.5 ms       |
| 32,760                     | 24.9 ms    | 64.2 ms       |
| sum over the 21 KV lengths | **275 ms** | 673 ms        |

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

Measured on the M4 Max with the demo server driven by a scripted websocket client: scene 00,
first session after server start, 5 s between turns, previews and prefetch on (the defaults).
Each turn is one chunk: 16 frames, 1 s of 16 fps video. Times are from the key press.

|                          | 832x464 (default)    | `--size 576x320`  |
| ------------------------ | -------------------- | ----------------- |
| world ready (cold start) | 18.6 s               | 10.7 s            |
| chunk 0 (13 frames)      | 12.5 s               | 5.5 s             |
| first preview, turns 1-8 | 3.5-6.5 s            | **1.4-1.9 s**     |
| turns 1-3                | 11.7 / 15.0 / 17.3 s | 4.5 / 4.7 / 5.3 s |
| turns 4-8                | 18.2-25.8 s          | **5.8-6.2 s**     |

During a turn the demo shows a rough still of the chunk being generated (`PREVIEW 1/4`, then
`2/4`), then plays the 16 frames. Turns get slower while the rolling KV cache fills (it holds 18
latent frames, full after about five chunks). At 832x464 a turn is ~12-25x slower than real time,
so you explore shot by shot; at 576x320 it's ~5-6x, close to interactive, at lower fidelity (see
[Smaller frames](#smaller-frames-for-speed---size)). Before the performance work below, a first
session took 24-34 s per turn at 832x464 with nothing on screen in between.

Every world is saved when it ends (Stop, restart, disconnect, idle timeout or the end of the
horizon): the server pipes the JPEG frames it already sent to the page through ffmpeg into an H.264
MP4 at 16 fps, and writes the start image (PNG) and a JSON file with the prompt, seed, size and
camera actions next to it, in `outputs/lingbot_demo/` (`--save-dir DIR`; `--save-dir ''` turns it
off). The page links to the video when it's written. That includes uploaded images and prompts,
so turn it off on a demo other people use if they shouldn't be recorded.

Peak memory went from 83 GB (upstream conditioning) to 43 GB on a 2-chunk `generate.py` run; since
the pipeline releases cached blocks after each chunk, `latency.py` holds a flat ~23 GB of driver
memory (a 576x320 run peaks at ~13 GB). The demo peaked ~7 GB higher until two changes: off CUDA the
11 GB umT5 encoder is dropped after encoding a prompt and rebuilt from its memory-mapped
checkpoint for the next one (+0.5 s per prompt change, bit-identical embeddings), and the MPS
cache is released after the between-turn condition prefetch (~4 GB of VAE scratch blocks that
otherwise sat under the next chunk). At 832x464 the lowest system-available memory during a
7-chunk run went from 8.1 GB (of 46.2 GB at start) to 13.1 GB (of 44.0 GB): peak use 38.1 ->
30.9 GB, with no measurable change in chunk time.

### Smaller frames for speed (`--size`)

`generate.py`, `latency.py` and `demo.py` take `--size WxH` (multiples of 16, one device). Only
832x464 is what LingBot and the FastFrames refiner/decoder were trained at, so smaller sizes trade
fidelity for speed. On a Mac the demo page also has a **frame size** toggle (Full 832x464, Fast
576x320, Quarter 416x240; `--size` picks the default): the worker rebuilds the size-dependent
buffers around the loaded weights when a world starts at another size, freeing the old ones first.
Restarting at a new size took ~13 s (576x320) and ~7 s (416x240) to the first chunk, including
re-encoding the prompt. Quarter-size demo turns took 2.6-3.6 s. Example 00, seed 0, same camera
path, M4 Max:

| size    | tokens/frame | full-cache chunk (`generate`-style, condition inline) | demo turn (5 s think time)          | look                                                                                |
| ------- | ------------ | ----------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------- |
| 832x464 | 1508         | 30.9 s                                                | 11.7-25.8 s, preview at 3.5-6.5 s   | the model as trained                                                                |
| 576x320 | 720          | 7.2 s                                                 | **4.5-6.2 s**, preview at 1.4-1.9 s | softer detail, coherent; a different take on the scene; follows the camera less far |
| 416x240 | 390          | 3.6 s                                                 | not measured                        | melts within ~8 chunks (the horse smears) and stops following the camera            |

At 576x320 attention shrinks faster than the token count (both the chunk and the cache get
smaller), so a chunk is 4.3x faster for 2.1x fewer tokens. A 125-frame `generate.py` video takes
60 s, and the demo peaks at ~13 GB of MPS memory against ~18 GB at full size. A different size
draws different noise, so it's a different world from the same image and prompt, not a
downscaled one. `latency.py` records the size and `summarize.py` skips records not taken at
832x464.

```bash
# ~5-6 s turns with a rough preview in under 2 s (previews and prefetch are on by default)
.venv/bin/python apps/lingbot/demo.py --gpus 1 --lingbot-root upstream/lingbot-world-v2 --size 576x320
```

What changed for LingBot:

| Problem on Mac                                                                                                                                                                                                  | Fix                                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upstream `wan/modules/t5.py` evaluates `torch.cuda.current_device()` as a default argument at import time                                                                                                       | `upstream.py` stubs it to `"cpu"` only during that import, without CUDA; the clone isn't modified                                                                               |
| The image condition was VAE-encoded for the whole 60-chunk horizon before the first frame: minutes of work, ~40 GB extra, and on MPS the **last latent comes out wrong** (0.4 max abs error vs a CPU reference) | `conditioning.LazyImageCondition` runs the same causal encoder (same feature cache) one latent at a time, just before each chunk needs it; it matches the CPU reference to 7e-6 |
| umT5 built with random FP32 init: 82 s to load                                                                                                                                                                  | Meta-device build + `load_state_dict(assign=True)` from an mmapped checkpoint: 0.9 s, bit-identical embeddings                                                                  |
| complex128 RoPE table, float64 timestep sinusoids and x0 update                                                                                                                                                 | Computed on CPU, results moved to MPS (as for FastFrames-S)                                                                                                                     |
| Attention uses split-KV SDPA tuned for CUDA                                                                                                                                                                     | On MPS it calls `fastframes.wan.attention`, i.e. the Metal flash kernel                                                                                                         |
| `torch.autocast("cuda")`, CUDA streams for KV-refinement head groups and decoder warm-up, `torch.cuda.synchronize`                                                                                              | Autocast on the tensor's device type; head groups run sequentially and warm-ups are skipped off CUDA; `fastframes.device.synchronize`                                           |
| torch.compile (inductor max-autotune), CUDA graphs, demo warm-up passes that only exist to trigger them                                                                                                         | Eager off CUDA (`generate.py` picks it automatically; the demo worker passes `compile=False, graphs=False`)                                                                     |
| `from_pretrained(device=None)` left the refiner/decoder on CPU                                                                                                                                                  | The device is resolved once at the top (`default_device()`)                                                                                                                     |
| Demo memory stats call `torch.cuda`                                                                                                                                                                             | MPS reports `current_allocated_memory` / `driver_allocated_memory`                                                                                                              |

### From a Blender grey box

Block out a scene in Blender, animate its camera, and LingBot flies the same path through a photoreal
version of it. `apps/lingbot/blender_stage.py` runs inside Blender and exports the camera path at 16 fps
as OpenCV camera-to-world poses plus intrinsics (`poses.npy`, `intrinsics.npy`, `condition.json`). It
also writes the start frame twice: `staging.png` (the render) and `depth.png` (inverse depth, near white,
normalized over the visible pixels). A depth ControlNet turns `depth.png` into the real first frame, and
`generate.py --image` starts from it:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b scene.blend --python apps/lingbot/blender_stage.py -- \
    --out outputs/stage --prompt "A rainy city street at dusk, wet asphalt reflecting neon signs"
# first frame: FLUX.1-dev + Union Pro 2.0 depth in Draw Things (any depth ControlNet works), 2x size
sips -z 960 1664 outputs/stage/depth.png --out outputs/stage/depth_2x.png
draw-things-cli generate --model flux_1_dev_q8p.ckpt --prompt "Photograph of a rainy city street ..." \
    --image outputs/stage/depth_2x.png --strength 1.0 --config-file union_depth.json \
    --width 1664 --height 960 --steps 20 --cfg 3.5 --output outputs/stage/keyframe.png
.venv/bin/python apps/lingbot/generate.py --model lingbot-1.3b --lingbot-root upstream/lingbot-world-v2 \
    --condition outputs/stage/condition.json --image outputs/stage/keyframe.png --output outputs/stage.mp4
```

`union_depth.json` holds one `controls` entry: `controlnet_union_pro_flux_1_dev_2.0_q5p.ckpt`, weight
0.8, `inputOverride` `depth`, guidance 0-0.7.

Measured on a 5 s street dolly (boxes for buildings and cars, a camera that yaws right, then left):
the export takes 4 s, the 1664x960 keyframe 228 s, and the 77-frame video 105 s (13.5 GB minimum
available memory). The video follows the Blender camera through the dolly and both turns.

What to expect:

- Without `--structure` (below), only the first frame sees your blocking. LingBot follows the camera
  exactly but invents everything the first frame doesn't show: a doorway built off-screen comes back
  as a different doorway.
- The ControlNet reads shape, not intent: 1.4 m boxes standing in for cars became concrete planters.
  Rough-model the things that need to read correctly, or name them in the prompt.
- Plain image-to-image from `staging.png` doesn't work. FLUX keeps the grey palette and paints fog
  (strength 0.65), or loses the layout (0.8). Use the depth map.
- Scale doesn't matter. LingBot normalizes the path's translations, so only the shape of the move
  counts. `--start`/`--end` pick the frame range; the path is cut to `4k + 1` frames (at least 13).

#### Steering every frame (`--structure`)

`blender_stage.py --structure` also renders every path frame to `structure/`, and
`generate.py --structure outputs/stage/structure` steers each chunk with them, SDEdit style. The
grey-box video is VAE-encoded one latent at a time, and each chunk starts from its latents noised
to denoising step `--structure-step` (default 2) instead of from pure noise. The steps before that
are skipped.

The structure latents are recoloured to the first image first: their per-channel mean and std
are matched to its latent. Without that, a flat grey render survives the noise as its colour and
brightness, and every frame comes out as grey fog, even at step 1 (6% structure).

The structure frames are rendered shadowless by default (`--structure-lighting studio`:
Workbench, studio light, one grey, no shadows), so surfaces read by their orientation alone. With
the scene's own lighting (`--structure-lighting scene`), the sun's cast-shadow boundaries came out
of the generation as painted walls.

Same street and seed, from the same keyframe:

| Start                      | Structure light | Steps | Video (77 frames)   | Result                                                                                       |
| -------------------------- | --------------- | ----- | ------------------- | -------------------------------------------------------------------------------------------- |
| pure noise (no structure)  | -               | 4     | 98-105 s            | the grey box only shapes frame 0                                                             |
| `--structure-step 1` (6%)  | scene           | 3     | 81 s                | as clean as no structure; follows the blocking loosely                                       |
| `--structure-step 2` (17%) | scene           | 2     | 66 s (+18 s encode) | car-sized boxes become cars where they stand; murkier, brown cast                            |
| `--structure-step 2`       | studio          | 2     | 86 s with encode    | cars at every box in every frame, clearer streets; still hazy, light sky (see below)          |
| `--structure-step 3` (38%) | scene           | 1     | 69 s with encode    | traces the blocking exactly but looks painted; shadow edges drawn as walls                   |
| `--structure-step 3`       | studio          | 1     | 65 s with encode    | clean painted grey box: boxes stay boxes, no shadow walls; no photographic detail            |
| step 2 without recolouring | scene           | 2     | 58 s                | grey fog                                                                                     |

Step 2 with studio light is the default and the best balance. Step 3 is a painted look: one step
can paint the layout but not rebuild it. Recolouring matches statistics over the whole frame, so
the grey box's light sky turns the keyframe's black night sky hazy and bright. A more detailed
grey box should help step 2 most, since less is left to invent. Without `--structure`, `step()`
is unchanged: the video is frame-identical to before.

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

| Component              |      M4 Max | B200 (README) |
| ---------------------- | ----------: | ------------: |
| DiT evaluation, t=1000 |   1340.3 ms |       16.1 ms |
| DiT evaluation, t=833  |   1676.8 ms |       16.2 ms |
| KV refinement          |    119.6 ms |        2.2 ms |
| Pixel decoder          |     57.4 ms |        0.9 ms |
| **Total**              | **3194 ms** |   **35.4 ms** |

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

## Performance work: VAE convolutions, attention pipelining, demo prefetch and previews

Profiling a full-cache LingBot chunk and a FastFrames-S DiT step on the M4 Max found three wins;
the last two items make demo turns feel faster without changing a frame.
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
- **Frames before the cache write** (`step(..., defer_cache=True)` / `finish_chunk`). A chunk's
  frames don't depend on its KV-cache write (KV refinement, or the first chunk's clean pass), which
  only the next chunk reads. The demo worker replies with the frames first and writes the cache
  between turns, with the prefetch: 0.7 s per full-cache chunk off each turn (2.4 s on the first),
  bit-identical frames. Same switch as the prefetch.
- **Rough previews while a chunk generates** (`step(..., preview=callback)`). The pixel decoder
  turns the DiT features of the first and second of the four denoising passes into a hazy but
  recognizable version of the chunk (20-21 dB and 22-23 dB PSNR against the final frames, against
  11-18 dB for holding the previous frame). The demo shows the middle frame as a still, marked
  `PREVIEW 1/4` / `2/4`, until the chunk's frames arrive: about 5 s and 9.5 s into a ~20 s
  full-cache turn, 3.5 s and 6.3 s into an early one. The decoder's history is restored after each
  preview, so the frames are byte-identical with previews off (141/141 demo frames). Each preview
  costs one pixel decode (~0.25 s), below the run-to-run noise of a turn. `LINGBOT_DEMO_PREVIEW=0`
  disables them (MPS/CPU, one device).
- **The image condition stops costing anything after chunk 8** (`LazyImageCondition`). After the
  first image the condition encodes black frames, and the causal VAE encoder converges: latents are
  bit-identical from latent 30 on at all three sizes. Once a new latent _and_ the encoder's whole
  feature cache repeat the previous step bit for bit, the encoder is at a fixed point, and later
  latents are reused without encoding (exactly the same values: 15 chunks with and without the
  shortcut are bit-identical at 832x464, 576x320 and 416x240). From chunk 8 the condition costs
  0 s instead of 4.19 / 2.45 / 1.32 s per chunk. That was the work a quick key press waited for
  between turns: in a quarter-size session with no pause between turns, the input-to-frame time
  outside the model went from 0.6-1.0 s (turns 1-7) to 8-10 ms (turns 8-10).

What's left in a full-cache LingBot chunk is mostly attention (~60%) and GEMMs, which run at
12.4-14.2 TFLOPS (MPS's own matmul), close to what this GPU does.

## Splitting a chunk with an iPad

Could an iPad Pro share the DiT's work over a cable? Splitting _layers_ between devices (exo's
remote stage) can't help: the passes are sequential, so one device always waits. Splitting a
chunk's _tokens_ (upstream LingBot's 8-GPU layout: each rank runs every layer on its share and
the ranks exchange K/V once per layer) runs both at once. `scripts/bench_dit_block.py` times one
DiT block at the demo's shapes; the same benchmark runs on the iPad as `DiTBench.swift` in the
[exo-touch](https://github.com/eaglstun/exo-touch) app.

| per block | M4 Max, PyTorch MPS (today) | M4 Max, MLX | iPad Pro 11" M5, MLX |
| --------- | --------------------------- | ----------- | -------------------- |
| 832x464   | 174.8 ms                    | 124.4 ms    | 160.4 ms             |
| 576x320   | 39.5 ms                     | 39.2 ms     | 50.2 ms              |
| 416x240   | 17.7 ms                     | 17.7 ms     | 22.9 ms              |

In a short burst the M5's GPU matrix units put it within ~10-25% of the M4 Max; under sustained
load it settles at about half the Mac (below). The link adds its own cost: an iPad
joins a Mac as a USB network device at 10 Gb/s (858 MB/s measured, 0.29 ms round trip), not over
Thunderbolt networking. Each layer of each pass exchanges the chunk's K/V (~37 MB at 832x464).

The synthetic blocks above concatenate the cache every block; the real DiT writes it in place.
`scripts/lingbot_block_capture.py` saves the real block 0's inputs, weights and output during a
full-cache chunk, and `scripts/lingbot_block_mlx.py` runs the same block in MLX (same dtype rules:
FP32 residual and norm statistics, BF16 matmuls, FP32 RoPE rotations). At 832x464 the MLX block
matches PyTorch to 8.4e-4 relative L2 (largest difference 0.047 on values up to 33, under half a
BF16 step there) and takes:

| real block 0, 832x464, full cache                                  | per block        | DiT per chunk |
| ------------------------------------------------------------------ | ---------------- | ------------- |
| PyTorch MPS (the demo today)                                       | 144.1 ms         | 17.3 s        |
| MLX eager                                                          | 136.7 ms (1.05x) | 16.4 s        |
| MLX `mx.compile`                                                   | 127.4 ms (1.13x) | 15.3 s        |
| iPad Pro 11" M5, MLX-Swift compiled (`DiTBlockCheck.swift`), burst | 156.8 ms (0.92x) | 18.8 s        |
| the same iPad after ~50 s of GPU load                              | ~270 ms (~0.53x) | ~32 s         |

The iPad's block matches PyTorch to 1.2e-3 relative L2 (max 0.089 on values up to 33, under one
BF16 step there); eager it takes 209.7 ms, so compiling matters more there than on the Mac.

So moving the Mac to MLX would buy ~13%, not the ~1.4x the synthetic blocks suggested. The
projection that follows used the burst figure, which the iPad doesn't sustain: at 0.92x the Mac it
would take ~48% of the tokens; a full-size chunk's DiT would then take
roughly 120 x (75 ms compute + ~22 ms transfer) = ~11.6 s instead of ~17.3 s (1.5x), up to ~1.9x
if transfers overlap compute. At 576x320 the gain shrinks to ~1.2-1.5x and at 416x240 the transfers cancel it. Building
it needs the block in MLX-Swift (`lingbot_block_mlx.py` is the reference), a K/V exchange per layer,
and the refined cache sent to the iPad between turns.

**Built: the iPad shares LingBot's DiT** (`apps/lingbot/ipad.py` here, `SplitStage.swift` in
exo-touch, `demo.py --ipad <its USB address>`). Every denoising pass's chunk tokens are split;
in each block both sides swap their tokens' canonical (pre-RoPE) keys and values, rotate the other's,
and attend over the whole cache, so this Mac also holds every token's canonical K/V for the KV
refiner. The iPad mirrors the cache: the weights once (2.6 GB in ~3.7 s, footprint 8.2 GB), the
prompt's text K/V, resets, eviction, each chunk's RoPE table and camera tokens (it computes its own
camera rows), and the refined rows after each chunk (1.1 GB, sent in `finish_chunk`, i.e. between
turns in the demo). The split moves every chunk towards where both sides finish a pass together.

One pass on its own (`lingbot_ipad_split.py`'s first version): 4.53 s alone vs 3.10-3.25 s split
(~1.4x), 4.9e-3 relative L2 against the Mac alone; its own split block with every token on the
Mac matches the pipeline's bit for bit. A whole world (`scripts/lingbot_ipad_split.py`: Mac alone,
split, Mac alone again; the two Mac-alone runs are bit-identical):

| 10-chunk world, 832x464                                        | full-cache chunk (median, chunks 5-9) |
| -------------------------------------------------------------- | ------------------------------------- |
| Mac alone (both runs, Photos analysing media part of the time) | 27.3 s                                |
| split with the iPad, rebalanced every chunk (its share 38-55%) | 21.6 s (1.26x)                        |

The split frames show the same scene and stay clean; their details drift from the Mac's as an
autoregressive model's do from any small difference (PSNR 43 dB at chunk 0, ~20 dB by chunk 9). In
the browser demo full-size turns took 10.7 / 12.0 s of model time against 11.6-12.2 / 15.6 s
alone; at 576x320 and 416x240 the exchange costs more than the iPad saves (4.6 vs 4.4 s, 3.4 vs
2.6 s), so the demo shares only 832x464 unless `--ipad-all-sizes`.

**Why 1.26x and not ~1.5x: the iPad's GPU throttles under sustained load.** It holds benchmark
speed for ~20-27 s of continuous GPU work, then drops to ~0.5-0.6x and stays there; ~30 s idle
buys back ~10 s of full speed (hence the faster first turns in the demo, which idles between
turns). exo-touch's `mac/split_fake_mac.py` drives the iPad's side with random weights and no
Mac model: with no traffic at all (`--solo`) a 2960-token pass over 27144 cache rows goes from 2.5 s
to 4.0-4.4 s after the eighth pass, at 6032 or 27144 rows and at 5.4 or 9.3 GB footprint alike;
the exchange adds ~20% on top only once throttled (4.9 s; a real session 5.0-5.3 s). The iPad's
`thermalState` stays nominal, thermal pressure 0, a fixed CPU loop steady and the battery full
throughout, so iOS doesn't report it; nothing in software moved it (zero-copy sockets, idling the
GPU between passes). The adaptive split absorbs it by moving tokens back to the Mac (the 38-55%
shares above). What's left: cooling the iPad, or cheaper attention on it.

**Hiding the K/V exchange (built; neutral).** Both sides now attend over the cache's older rows while
the other side's K/V are in flight, then over this chunk's rows, and merge the two by their
log-sum-exp (on by default; `lingbot_ipad_split.py --no-overlap` for the old order). Here that's
`flash_attention_lse` (the Metal kernel writing each row's log-sum-exp and an FP32 output) and
`merge_attention`: as close to an FP32 reference as one call (1.66e-3) at the same cost. On the iPad,
MLX computes the log-sum-exp in its fused attention for training only; exo-touch patches mlx-swift to
expose it (1.34e-3 against PyTorch on the real block, 1.24e-3 in one call). Full-cache chunks: 19.62 s
split with it, 19.77 s without (1.25x / 1.17x against the Mac alone in each run: within the
run-to-run noise). Attention over the older rows was never the critical path. Each side's next K/V
need the other's previous ones, so every block waits on the transfer both ways plus the rest of the
other side's block (projections, cross-attention, FFN, next QKV); the overlap just turns the iPad's
waiting into attention.

That exchange takes ~60-70 ms per block both ways (each side's send-to-receive time minus the
other's turnaround; `lingbot_ipad_split.py` prints it), and it's mostly bytes: a ~24 MB Mac K/V frame
spends 28.6 ms in `sendall`, the link's 858 MB/s, and the iPad's ~13 MB about 23 ms. Sending the
tensors' bytes in place instead of joined copies changed nothing (72 -> 70.5 ms). `--kv-bits 8`
sends int8 per token and head with an FP32 scale (51.6% of the bytes; each side keeps its own K/V
exact): the frames match the 16-bit split's PSNR chunk for chunk, but its one world run was
dominated by the iPad throttling (the iPad never waited) and showed no gain, so it stays opt-in.
Split timings swing ~10% with how hot the iPad is, more than 8-bit could win; a cooled iPad is the
way to measure it.

Two PyTorch 2.9 MPS traps found on the way (both handled in `apps/lingbot/ipad.py`):

- A non-blocking copy of a **non-contiguous** MPS tensor to the CPU comes back wrong (10 of 10
  tries). V, a slice of the fused QKV output, reached the iPad as garbage and every frame after
  chunk 0 went black. `.contiguous()` first.
- `torch.mps.Event.synchronize()` and `torch.mps.synchronize()` **hold the GIL** while they wait (a
  second thread drops from 54M to 2M iterations/s; `.cpu()` releases it), stalling a sender thread.
  Poll `event.query()` with short sleeps instead. Also: PyTorch MPS commits queued kernels only at a
  sync, so work queued before Python blocks on a socket doesn't start (`Event().record()` commits
  without waiting), and `.to("mps")` from the CPU waits for every queued kernel (111.6 ms behind a
  busy queue, 0.4 ms idle).

## What was changed for FastFrames-S (see `git diff`)

| Problem on Mac                                                                                                                                                          | Fix                                                                                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `device="cuda"` hardcoded in the loader, pipeline, and autocast                                                                                                         | `fastframes/device.py`: `$FASTFRAMES_DEVICE`, else cuda > mps > cpu                                                                        |
| `sdpa_kernel(CUDNN_ATTENTION)`                                                                                                                                          | Used only on CUDA                                                                                                                          |
| MPS has no float64/complex128 (timestep sinusoids, scheduler x0, RoPE)                                                                                                  | float64 math runs on CPU (bit-exact); RoPE uses a cached complex128 table applied as FP32 rotations (within 1 bf16 ulp, 99.998% identical) |
| MPS SDPA caches one MPSGraph **plus its FP32 score buffer** per shape; the KV prefix grows every frame, so 21 shapes retained ~56-68 GB that `empty_cache()` can't free | Default: the Metal flash-attention kernel above (no per-shape state). SDPA fallback: one head per call, ~6 GB retained                     |
| Allocator hoards odd-sized buffers                                                                                                                                      | `synchronize()` + `empty_cache()` after each frame on MPS                                                                                  |
| umT5 (11 GB) stays resident during the rollout                                                                                                                          | `generate.py` encodes all prompts first, then moves T5 to CPU                                                                              |
| T5 built with random FP32 init (48 s, 22 GB RAM) before loading bf16 weights                                                                                            | Built on the meta device and assigned the checkpoint (same result)                                                                         |
| OpenH264 binary is linux64-only (ctypes)                                                                                                                                | Non-Linux `save_video` uses ffmpeg/libx264, BT.601 limited range, faststart                                                                |

Not ported: training (NCCL, Triton flex_attention, CUDA RNG state), FastFrames-L (14B: ~28 GB DiT

- ~26 GB KV cache + T5 does not fit in 64 GB without offloading), multi-device LingBot
  (`--gpus 8`, torchrun), and the multi-GPU and baseline latency rows.
