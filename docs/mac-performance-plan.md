# Further performance work on Apple Silicon

This review targets the M4 Max 64 GB Mac documented in `MAC.md`, using the existing
PyTorch 2.9.1 MPS environment. The baseline is fork commit `974f473`, including the
three-pass LingBot schedule and first-chunk image-condition reuse. MPS availability
was checked locally. The timings below are prior measurements from `MAC.md`, not
new benchmark results.

The next useful work is reducing repeated memory movement and eager operations
around the DiT. Attention and large matrix products already approach this machine's
measured throughput. Expect incremental gains; no evidence supports another
large speedup from tuning those kernels alone.

## Current costs

The [existing profile](../MAC.md#performance-work-vae-convolutions-attention-pipelining-demo-prefetch-and-previews)
puts the DiT at roughly 80% of a full-size LingBot chunk. Inside a full-cache block,
self-attention accounts for approximately 60%, the FFN 18%, other projections 13%,
and elementwise operations and cache writes 12%. These approximate measurements
are not an exact partition. They predate the latest schedule and conditioning
defaults and must be remeasured for end-to-end comparisons.

The existing real-block MLX experiment improved 144.1 ms to 127.4 ms when compiled,
about 13% less block time. That is useful evidence of remaining overhead, but not
proof that a framework migration would give the same whole-video improvement.

FastFrames-L's int8 KV and DiT options primarily improve memory capacity. They
already make 81-frame inference possible. The VAE condition shortcut, reduced
denoising schedule, lazy conditioning, contiguous convolution taps, and text-encoder
release are implemented and should remain in the baseline.

## Sequential implementation queue

Each item gets an Astra agent, its own branch based on the same baseline, and a
separate PR if it produces a defensible improvement. Only one agent and one GPU
benchmark workload run at a time. Failed experiments are recorded without shipping
a slower default. These candidates are hypotheses until measured.

### 1 Batch rolling KV eviction

`FastFramesLingBot._evict` currently loops over K and V, every layer, and safe
non-overlapping token bands. The ordering is necessary because the source and
destination overlap across the whole shift, but it does not require a separate
Python copy call per layer. A strided slice across all layers could reduce eager
dispatches while retaining the existing band order and bounded scratch memory.

Own `apps/lingbot/pipeline.py`, a focused cache test, and a reproducible eviction
benchmark. Preserve sink frames, retained-frame order, iPad eviction notification,
and the CUDA path. Check whether MPS strided copies secretly allocate a large
temporary; reject a memory regression. Benchmark all three demo resolutions and
validate repeated eviction with exact tensor equality. Compare an actual clip
long enough to cross the first eviction boundary.

### 2 Fuse real RoPE rotations on MPS

`apps/lingbot/dit.py:rope_real` widens input values, performs several elementwise
products and sums, stacks real and imaginary components, and converts back to the
input dtype. It is called for queries and keys in every block, and by refinement.
A small runtime-compiled Metal kernel could do this in one dispatch without those
intermediate tensors. Use the existing attention module as the repository's
shader-loading pattern.

Own a new MPS RoPE helper, its narrowly gated integration, tests and benchmark.
Handle actual strides and storage offsets, or explicitly fall back for unsupported
layouts. Preserve FP32 intermediate rounding and the final BF16 cast; fused
multiply-add contraction can change results. Keep CPU/CUDA behavior unchanged and
retain a reference switch for A/B comparisons. Require tensor parity, repeated
determinism, actual block timing, and fixed-seed frame comparison before enabling
the fast path by default.

### 3 Reuse timestep modulation

`LingBotDiT._block` recomputes `block.modulation + time_e0`, followed by `1 + e[1]`
and `1 + e[4]`, every time a block runs. The model weights and supported timesteps
are fixed across chunks. The existing `time_bank` provides a natural place to
precompute these small tensors for eager MPS execution.

Own the timestep bank and its use in `apps/lingbot/dit.py`, tests, and a benchmark.
Keep the callable block interface usable by existing capture/profiling scripts,
and keep CUDA graph behavior and the iPad implementation unchanged. Account for
the clean timestep zero as well as every denoising timestep, dtype transitions,
size rebuilds and prompt changes. Require exact arithmetic parity and a measured
improvement; this is the lowest-payoff candidate and may be rejected as noise.

## Further candidates and reasons to defer

| Candidate | Assessment |
| --- | --- |
| Fused normalization, modulation and residual kernels | Broader opportunity within the measured glue cost, but reduction order and BF16 rounding require careful validation. Start with the simpler RoPE case. |
| VAE layout-copy reduction | Inspect `mps_conv3d.py` if first-frame or structure encoding remains dominant. It transposes and materializes inputs and outputs per convolution. Reusing layouts across layers is a larger change and non-contiguous weights have previously produced wrong results. |
| Decode fewer preview frames | Full preview decoding costs roughly 0.25 s in the existing measurements. Temporal decoder history makes extracting a middle frame more complex than slicing the input. |
| MPS int8 matrix kernels or a PyTorch upgrade | The current packed-int8 route has documented intermittent corruption. Test in a separate environment; do not replace the working environment or assume newer versions fix it. |
| Full MLX DiT | Existing block measurements suggest a modest gain, with substantial integration and numerical-validation work. Keep as a separate project after localized improvements. |
| Attention tile tuning | Several tile, layout and shared-memory variants have already failed. Reopen only with a new profile or a specific untested mechanism. |
| Shorter KV window or fewer steps | The current experiments already expose quality costs or gains below noise. These are quality tradeoffs, not transparent implementation fixes. |
| More iPad work | Sustained thermal throttling limits the measured benefit. It also requires another device, outside the immediate Mac-only pass. |

## Acceptance and measurement

Use the baseline's Python environment without `uv sync`. Read `AGENTS.md` before
running a model. Apply `limit_mps_memory` before GPU allocation and retain the
existing synchronized cache releases. Do not start a second model while one is
resident. If other GPU activity cannot be inspected, report that limitation and
use interleaved A/B timings to reduce bias.

Warm shaders and shapes before timing. Synchronize at timing boundaries, record
individual A/B samples and medians, and distinguish isolated operation, real-block,
whole-clip, and interactive-turn results. Use 832x464, 576x320 and 416x240 where
applicable. Include both cache growth and steady state. Record driver allocation
as well as tensor allocation; a faster operation that retains unacceptable graph
or scratch memory is not a win.

Numerically neutral changes must preserve fixed-seed frame hashes on the same
device. If arithmetic differs, inspect representative early, middle and late
frames and describe the tradeoff rather than claiming equivalence from PSNR alone.
Only claim whole-video speedups when a whole-video comparison was run. A useful
isolated optimization may still have an end-to-end gain below timing noise.

Each fix PR should include its baseline commit, reproducible commands, measured
timing and memory results, correctness checks, and remaining limitations. Open PRs
against `eaglstun/ml-fastframes:main`; do not merge automatically.

## API references

PyTorch documents [runtime Metal shader compilation](https://docs.pytorch.org/docs/main/generated/torch.mps.compile_shader.html)
and [MPS synchronization and memory counters](https://docs.pytorch.org/docs/stable/mps.html).
Those pages describe newer releases; verify signatures against the installed
2.9.1 implementation before using them. Existing runtime-compiled attention in
this repository is the tested integration reference.

## Results

### Batched cache eviction

[PR 28](https://github.com/eaglstun/ml-fastframes/pull/28), commit `bf3c3ac`, replaces
the per-layer loop on supported MPS buffers with a single Metal dispatch. The
initial PyTorch strided-copy experiment was rejected: it increased driver memory
by about 1 GiB and took 79 ms instead of roughly 10 ms at full size.

Nine-sample interleaved medians for the Metal implementation:

| Resolution | First eviction baseline to Metal | Steady eviction baseline to Metal |
| --- | --- | --- |
| 416x240 | 3.649 to 2.604 ms | 4.343 to 2.698 ms |
| 576x320 | 5.794 to 4.589 ms | 5.152 to 4.595 ms |
| 832x464 | 10.608 to 9.424 ms | 10.143 to 9.510 ms |

All 35 LingBot tests passed and all 93 raw RGB frames matched for example 00,
seed 0 at 416x240, crossing both eviction boundaries. Isolated allocation counters
stayed flat. The capped whole-clip comparison, including hashing, was 12.89 s
baseline versus 14.02 s candidate. This does **not** establish a whole-video
improvement. The PR is draft pending repeated end-to-end measurements to distinguish
noise from a regression; the isolated gain is small enough that it may not justify
shipping the added kernel. Two preliminary clip checks lacked the memory cap;
validation was repeated with a 45% cap before the PR was opened.

### Fused real RoPE

[PR 29](https://github.com/eaglstun/ml-fastframes/pull/29), commit `9b876d9`, adds
a single-dispatch Metal rotation for supported contiguous BF16 MPS tensors.
Unsupported layouts, dtypes and autograd use the original operations. The shader
preserves separate FP32 multiplication and addition rounding. Set
`FASTFRAMES_MPS_ROPE=eager` to select the reference path.

At 832x464, isolated DiT rotation fell from 1.900 to 0.165 ms; batch-30 refiner
rotation fell from 68.910 to 2.726 ms. Driver allocation sampled after the latter
operation fell from 6.82 to 2.92 GB, with 1.11 GB live tensors in both cases.
These are operation measurements, not whole-process peak measurements.

| Real block resolution | Four-frame cache baseline to Metal | Full cache baseline to Metal |
| --- | --- | --- |
| 416x240 | 15.324 to 14.523 ms | 19.457 to 18.627 ms |
| 576x320 | 34.602 to 33.120 ms | 60.075 to 60.003 ms |
| 832x464 | 73.208 to 63.455 ms | 184.153 to 176.072 ms |

Full-size full-cache block time improved about 4.4%; the medium full-cache change
is below noise. All tested block outputs matched. All 37 LingBot tests passed,
including BF16 rounding, offset slices, batch-30 tensors, signed zeros, range
boundaries, layout fallback and autograd fallback. Four 93-frame rollouts at
416x240 produced identical hashes for every frame. Their eager/Metal/Metal/eager
times were 12.369/13.211/14.575/14.643 s: upward drift prevents a whole-video
speedup claim. Full-size model checks used the standard 60% cap after T5 loading
hit the initial 45% cap safely. CUDA and physical iPad validation remain untested.
