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

### Timestep modulation reuse rejected

The third Astra experiment tested an optimistic frozen-weight prototype before
adding mutation tracking or production caching. It precomputed all 30 blocks at
all five timesteps, preserving the bits of the six FP32 modulation vectors.
Building the bank took 2.95 ms and retained 7.03 MiB of live tensors and 8.25 MiB
of driver allocation at every resolution.

Alternating measurements of real block 0 gave these medians:

| Resolution | Four-frame cache baseline to cached | Full cache baseline to cached |
| --- | --- | --- |
| 416x240 | 15.267 to 15.270 ms | 19.498 to 19.487 ms |
| 576x320 | 28.326 to 28.406 ms | 52.137 to 51.908 ms |
| 832x464 | 66.992 to 66.964 ms | 220.762 to 231.775 ms |

Five cases changed by less than 0.5% with mixed signs; the last was approximately
5% slower with substantial spread. Even this optimistic prototype showed no
consistent improvement, so the added state and invalidation logic are not justified.
No production code was changed and no fix PR was opened. Whole-clip and rebuild
checks were omitted after rejecting the performance hypothesis. Full-size runs
used the standard 60% memory cap; other runs used 45%.

## Recommended disposition

Review PR 29: the follow-up below found a 4.12% median paired whole-clip reduction
at 416x240 (six of six wins) and a more variable 7.65% at 832x464 (four of six),
with exact frame parity. Leave PR 28 in draft and unmerged: the follow-up did not
reproduce its earlier roughly 9% regression, but found no repeatable whole-clip
benefit to justify the added kernel.
Do not implement timestep modulation caching based on the current measurements.
All three experiments ran sequentially, on independent branches from `974f473`;
their speedups cannot be added together and combined performance is unmeasured.

## Follow-up whole-clip validation

The follow-up RoPE test uses the same resident pipeline and fixed scene, seed and
93-frame horizon for every run. Both variants first generate a full warm-up clip.
Six measured pairs at 832x464 alternate eager-first and Metal-first order, fixed
before results are observed. Every clip recreates image conditioning and resets
KV caches, decoder history and seeded noise.

Generation time is the sum of synchronized per-chunk wall times, including camera
rays, lazy condition encoding, DiT, refinement, decoding and cache releases.
Model loading, prompt/start setup, CPU output copies and hashing are outside this
metric. Separate wall times include the copies and hashes. Every frame is checked
against the eager warm-up, and every chunk records live and driver allocation.
Chunk-end allocation readings are not peak-memory measurements. Full-size runs
use a 60% memory cap before allocation; smaller runs use 45%.

System thermal and performance counters were unavailable in this session, so
timing drift cannot be attributed to a specific cause. Adjacent balanced pairs
reduce order bias but cannot eliminate all variation from other system activity.

### Trained resolution RoPE results

All 14 clips at 832x464 (two warm-ups and six measured pairs) matched all 93 frames
against the same eager reference. Measured synchronized generation times:

| Pair | Order | Eager | Metal | Reduction with Metal |
| --- | --- | --- | --- | --- |
| 1 | Eager then Metal | 125.295 s | 109.977 s | 12.23% |
| 2 | Metal then eager | 113.136 s | 103.814 s | 8.24% |
| 3 | Eager then Metal | 117.199 s | 117.688 s | -0.42% |
| 4 | Metal then eager | 122.773 s | 111.006 s | 9.58% |
| 5 | Eager then Metal | 119.153 s | 110.744 s | 7.06% |
| 6 | Metal then eager | 86.085 s | 90.993 s | -5.70% |

The median paired reduction is 7.65%, with Metal winning four of six pairs.
The separate medians are 118.176 s eager and 110.360 s Metal. The eager-first and
Metal-first groups have median paired reductions of 7.06% and 8.24%, respectively.
The final two full-cache chunks show a median paired reduction of 7.65%, similar
to the first four chunks' 7.66%.

This supports a typical benefit in this run, but the range crosses zero and does
not establish a guaranteed or statistically significant whole-clip improvement.
Maximum sampled chunk-end driver allocation per clip ranged 18.20–19.86 GB for
Metal and 17.60–18.20 GB for eager. There is no whole-clip memory-reduction claim.
The single scene, seed and 93-frame horizon also limit generalization.

### Small resolution RoPE results

The same two warm-ups and six measured pairs at 416x240 matched all 93 frames in
every clip. Metal won all six pairs, with a median paired reduction of 4.12%
(0.569 s) and a range of 0.27–7.34%. The separate generation-time medians were
13.757 s eager and 13.077 s Metal. Per-clip maximum sampled chunk-end driver
allocation ranged 11.96–13.03 GB for both paths.

| Pair | Eager | Metal | Reduction with Metal |
| --- | --- | --- | --- |
| 1 | 13.710 s | 13.011 s | 5.10% |
| 2 | 13.804 s | 13.011 s | 5.74% |
| 3 | 13.943 s | 13.504 s | 3.15% |
| 4 | 13.223 s | 13.188 s | 0.27% |
| 5 | 13.441 s | 13.021 s | 3.13% |
| 6 | 14.174 s | 13.133 s | 7.34% |

The follow-up therefore provides consistent whole-clip evidence at small
resolution and a positive but more variable result at trained resolution.
PR 29 includes the improved benchmark and raw JSONL records in `docs/benchmarks/`.
These measurements supersede the initial four-clip test's inconclusive result
without implying equal gains for other scenes, sizes or interactive preview mode.

### Cache eviction follow-up

PR 28 uses the same full warm-ups, six balanced measured pairs and separate
generation/hash timing at 416x240. Its baseline method was checked against the
parsed Python AST of the original `_evict` at `974f473`. RoPE remains unchanged
in this independent branch. All 14 clips matched all 93 reference frame hashes.

| Pair | Baseline | Metal eviction | Reduction with Metal |
| --- | --- | --- | --- |
| 1 | 12.791 s | 12.816 s | -0.20% |
| 2 | 13.196 s | 12.851 s | 2.61% |
| 3 | 13.112 s | 13.257 s | -1.11% |
| 4 | 13.358 s | 13.169 s | 1.42% |
| 5 | 13.808 s | 14.390 s | -4.21% |
| 6 | 15.332 s | 14.864 s | 3.05% |

The candidate won three of six pairs. Its median paired reduction was 0.61%
(82 ms), but **the first run in every pair won regardless of implementation**.
The run-order effect prevents interpreting that median as a kernel benefit.
The final two chunks, which actually evict cache entries, also split three wins
each: the median saving was 19 ms and the range crossed zero widely.

Both implementations reached the same maximum sampled chunk-end driver allocation
of 13.029 GB. Final driver readings were 8.736–8.745 GB. The earlier roughly 9%
whole-clip slowdown did not reproduce, and no repeatable whole-clip gain was
established. Keep the PR draft and unmerged. A full-resolution repeat was not
run: the isolated millisecond saving is far below the observed clip timing
variation and does not justify another long run. The PR contains the complete
JSONL samples at `docs/benchmarks/lingbot-eviction-416x240.jsonl` and a reusable
clip benchmark.
