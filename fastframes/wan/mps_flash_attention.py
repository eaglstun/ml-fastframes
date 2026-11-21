# For licensing see accompanying LICENSE file.
"""FlashAttention-style scaled-dot-product attention as a single Metal compute kernel (MPS only).

Why: ``F.scaled_dot_product_attention`` on MPS materialises the full ``[Lq, Lk]`` FP32 score matrix
and caches one MPSGraph (plus that buffer) per input shape.  This kernel never materialises the
scores (online softmax over key blocks, as in FlashAttention), takes all shapes as runtime
arguments (one pipeline for every length, nothing retained per shape), and reads/writes the model's
native ``[B, L, H, D]`` layout directly, so there are no transpose copies either.

Design, per SIMD-group (32 threads = one threadgroup):
  * owns ``8*RT`` query rows and walks the keys in blocks of ``8*NK``;
  * ``S = Q K^T`` and ``O += P V`` run on the GPU's 8x8 ``simdgroup_matrix`` multiply-accumulate
    units with BF16 operand tiles and FP32 accumulators (the mixed form widens exactly, so the dot
    products and the 32k-key accumulation are done in FP32; only the final result is rounded to
    BF16);
  * the softmax (scale, row max, exp, rescale of the running output) is done in registers on the
    per-thread fragments of those matrices, so the scores never touch memory;
  * operand tiles are loaded ``_LOOKAHEAD`` steps before the multiply that uses them (software
    pipelining).  A load placed right before its use stalls the SIMD-group for the load latency;
    this alone took the kernel from ~10.6 to ~12.5 TFLOPS on an M4 Max.

Credits for the borrowed ideas:
  * MLX "steel" kernels (ml-explore/mlx, mlx/backend/metal/kernels/steel/, MIT): treating a
    ``simdgroup_matrix<T,8,8>`` as a per-thread two-element fragment (lane -> (row, col) mapping
    below), which allows typed loads/stores and elementwise math without ``simdgroup_load``.
  * philipturner/metal-flash-attention (MIT): tiling a GEMM/attention as register-resident 8x8
    accumulators per SIMD-group, loading operands straight from device memory.
  * Dao et al., FlashAttention: the online-softmax recurrence.
The lane mapping is not guaranteed by the MSL spec, so it is verified at first use and the caller
falls back to SDPA if it does not hold.
"""
import math

import torch

__all__ = ["flash_attention", "flash_attention_lse", "merge_attention", "is_available"]

_SRC = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

#define D   %(D)d          // head dim (multiple of 8)
#define H   %(H)d          // heads; tokens are H*D elements apart
#define NEG (-1.0e30f)

typedef simdgroup_float8x8 mat;     // accumulators (scores, output)
typedef simdgroup_bfloat8x8 bmat;   // operand tiles, as stored
#define FRAG(m)  (reinterpret_cast<thread float2&>(m))
#define BFRAG(m) (reinterpret_cast<thread bfloat2&>(m))

// Lane -> element mapping of an 8x8 simdgroup matrix (2 elements per thread), as used by MLX steel:
// row = 4*(lane>>4) + ((lane>>1)&3), cols = 4*((lane>>3)&1) + 2*(lane&1) + {0,1}.
// Lanes holding the same row are lane^1 and lane^8 (and lane^9).

kernel void probe_mapping(device float* out, const device float* in,
                          ushort lane [[thread_index_in_simdgroup]]) {
    mat m;
    simdgroup_load(m, in, 8);
    float2 f = FRAG(m);
    const uint fr = ((lane >> 4) << 2) + ((lane >> 1) & 3);
    const uint fc = (((lane >> 3) & 1) << 2) + ((lane & 1) << 1);
    out[lane] = (f.x == in[fr * 8 + fc] && f.y == in[fr * 8 + fc + 1]) ? 1.0f : 0.0f;
}

// q: [B, Lq, H, D], k/v: [B, Lk, H, D] bfloat, contiguous; out: [B, Lq, H, D] (OT);
// flash_attn_lse also writes each row's natural-log sum of exp(scores), lse: [B, Lq, H] float.
// Grid: x = 32 * ceil(Lq / (8*RT)) threads (threadgroup = 32 = one SIMD-group), y = B*H.
kernel void %(NAME)s(device %(OT)s* out,
                       const device bfloat* q,
                       const device bfloat* k,
                       const device bfloat* v,
                       constant long& Lq_,
                       constant long& Lk_,
                       constant float& scale,   // includes log2(e): the softmax uses exp2
%(LSE_ARG)s                       uint3 tg [[threadgroup_position_in_grid]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    const uint Lq = uint(Lq_), Lk = uint(Lk_);
    const uint fr = ((lane >> 4) << 2) + ((lane >> 1) & 3);
    const uint fc = (((lane >> 3) & 1) << 2) + ((lane & 1) << 1);
    const uint b = tg.y / H, h = tg.y %% H;
    const device bfloat* qh = q + (ulong(b) * Lq * H + h) * D;
    const device bfloat* kh = k + (ulong(b) * Lk * H + h) * D;
    const device bfloat* vh = v + (ulong(b) * Lk * H + h) * D;
    device %(OT)s* oh = out + (ulong(b) * Lq * H + h) * D;

%(BODY)s}
"""

def _emit(w, steps, lookahead):
    """Write ``steps`` (pairs of load statements and the multiplies that consume them) with every
    load issued ``lookahead`` steps before its multiplies."""
    for loads, _ in steps[:lookahead]:
        for x in loads:
            w(x)
    for n, (_, macs) in enumerate(steps):
        if n + lookahead < len(steps):
            for x in steps[n + lookahead][0]:
                w(x)
        for x in macs:
            w(x)


def _block(D, H, RT, NK, tail):
    """One online-softmax step over 8*NK keys starting at ``kb``.  ``tail`` adds index clamps and a
    key mask for the last, partial block; the fast path has only constant offsets."""
    DT, ST, LA = D // 8, H * D, _LOOKAHEAD
    R, J, DC = range(RT), range(NK), range(DT)
    Z = "make_filled_simdgroup_matrix<float, 8, 8>(0.0f)"
    o = []
    w = o.append
    # K^T tile element (row = dim, col = key): this thread reads dim fr of keys fc and fc+1.
    # V tile element (row = key, col = dim): this thread reads dims fc, fc+1 of key fr.
    if tail:
        for j in J:
            w(f"const device bfloat* ka{j} = kh + ulong(min(kb + {j * 8} + fc, Lk - 1)) * {ST} + fr;")
            w(f"const device bfloat* kb{j} = kh + ulong(min(kb + {j * 8 + 1} + fc, Lk - 1)) * {ST} + fr;")
            w(f"const device bfloat* v{j} = vh + ulong(min(kb + {j * 8} + fr, Lk - 1)) * {ST} + fc;")
        kload = lambda j, dc: f"bfloat2(ka{j}[{dc * 8}], kb{j}[{dc * 8}])"  # noqa: E731
        vload = lambda j, dc: f"bfloat2(v{j}[{dc * 8}], v{j}[{dc * 8 + 1}])"  # noqa: E731
    else:
        w(f"const device bfloat* kt = kh + ulong(kb + fc) * {ST} + fr;")
        w(f"const device bfloat* vt = vh + ulong(kb + fr) * {ST} + fc;")
        kload = lambda j, dc: f"bfloat2(kt[{j * 8 * ST + dc * 8}], kt[{(j * 8 + 1) * ST + dc * 8}])"  # noqa: E731
        vload = lambda j, dc: f"bfloat2(vt[{j * 8 * ST + dc * 8}], vt[{j * 8 * ST + dc * 8 + 1}])"  # noqa: E731
    # Enough operand registers that a tile loaded LA steps ahead never overwrites one still in use.
    NA, NKS, NV = LA + 1, -(-LA // RT) + 1, LA + 1
    w("bmat " + ", ".join([f"A{p}" for p in range(NA)] + [f"K{j}_{p}" for j in J for p in range(NKS)]
                          + [f"V{j}_{p}" for j in J for p in range(NV)]) + ";")
    w("mat " + ", ".join(f"S{i}_{j} = {Z}" for i in R for j in J) + ";")
    # S = Q K^T, FP32 accumulate.  One step per (dim tile, row tile).
    steps = []
    for dc in DC:
        for i in R:
            pa, pk = (dc * RT + i) % NA, dc % NKS
            loads = [f"BFRAG(K{j}_{pk}) = {kload(j, dc)};" for j in J] if i == 0 else []
            loads.append(f"BFRAG(A{pa}) = bfloat2(qt[{i * 8 * ST + dc * 8}], qt[{i * 8 * ST + dc * 8 + 1}]);")
            steps.append((loads, [f"simdgroup_multiply_accumulate(S{i}_{j}, A{pa}, K{j}_{pk}, S{i}_{j});" for j in J]))
    _emit(w, steps, LA)
    # Online softmax on the per-thread fragments (the scores never touch memory).  Base 2: log2(e)
    # is folded into ``scale``, and exp2 is cheaper than exp.
    w("float mx, mn, alpha; float2 s;")
    for i in R:
        w("mx = NEG;")
        for j in J:
            w(f"s = FRAG(S{i}_{j}) * scale;")
            if tail:
                w(f"s.x = (kb + {j * 8} + fc < Lk) ? s.x : NEG;")
                w(f"s.y = (kb + {j * 8 + 1} + fc < Lk) ? s.y : NEG;")
            w(f"FRAG(S{i}_{j}) = s; mx = max(mx, max(s.x, s.y));")
        w("mx = max(mx, simd_shuffle_xor(mx, (ushort)1));")
        w("mx = max(mx, simd_shuffle_xor(mx, (ushort)8));")
        w(f"mn = max(m{i}, mx); alpha = exp2(m{i} - mn); m{i} = mn; l{i} *= alpha;")
        for dc in DC:
            w(f"FRAG(O{i}_{dc}) *= alpha;")
        for j in J:
            w(f"s = exp2(FRAG(S{i}_{j}) - mn); l{i} += s.x + s.y; FRAG(S{i}_{j}) = s;")
    # O += P V.  One step per dim tile.
    steps = []
    for dc in DC:
        pv = dc % NV
        steps.append(([f"BFRAG(V{j}_{pv}) = {vload(j, dc)};" for j in J],
                      [f"simdgroup_multiply_accumulate(O{i}_{dc}, S{i}_{j}, V{j}_{pv}, O{i}_{dc});"
                       for i in R for j in J]))
    _emit(w, steps, LA)
    return "".join(f"        {x}\n" for x in o)


def _body(D, H, RT, NK, lse=False, out="bfloat"):
    """Straight-line kernel body: every 8x8 tile is its own named variable and every offset is a
    literal.  (Indexing arrays of matrices in loops kept them out of registers: 2-4x slower.)
    ``lse`` also writes each row's log-sum-exp; ``out`` is the output's element type."""
    DT, ST = D // 8, H * D
    o = []
    w = o.append
    # The last group is shifted back to end at Lq: it recomputes (and rewrites, identically) a few
    # rows of its neighbour instead of needing clamps or guards.  Requires Lq >= 8*RT.
    w(f"    const uint r0 = min(tg.x * {8 * RT}, Lq - {8 * RT}) + fr;")
    w(f"    const device bfloat* qt = qh + ulong(r0) * {ST} + fc;")
    for i in range(RT):
        w(f"    float m{i} = NEG, l{i} = 0.0f;")
        w("    mat " + ", ".join(f"O{i}_{dc} = make_filled_simdgroup_matrix<float, 8, 8>(0.0f)"
                              for dc in range(DT)) + ";")
    w("    uint kb = 0;")
    w(f"    for (; kb + {8 * NK} <= Lk; kb += {8 * NK}) {{")
    w(_block(D, H, RT, NK, False) + "    }")
    w("    if (kb < Lk) {")
    w(_block(D, H, RT, NK, True) + "    }")
    w("    float lt, inv; float2 ov;")
    for i in range(RT):
        w(f"    lt = l{i} + simd_shuffle_xor(l{i}, (ushort)1);")
        w("    lt += simd_shuffle_xor(lt, (ushort)8);")
        w("    inv = 1.0f / lt;")
        w("    {")
        w(f"        device {out}* op = oh + ulong(r0 + {i * 8}) * {ST} + fc;")
        for dc in range(DT):
            w(f"        ov = FRAG(O{i}_{dc}) * inv; op[{dc * 8}] = {out}(ov.x); op[{dc * 8 + 1}] = {out}(ov.y);")
        if lse:  # m and lt are in base 2 (log2(e) is in ``scale``); one of the row's four lanes writes
            w(f"        if ((lane & 9) == 0) lse[(ulong(b) * Lq + r0 + {i * 8}) * H + h] = "
              f"M_LN2_F * (m{i} + log2(lt));")
        w("    }")
    return "\n".join(o) + "\n"


# (row tiles, key tiles): 16 query rows per SIMD-group, 24 keys per online-softmax step.  Measured on
# an M4 Max: RT=2 is the most that stays in registers (RT>=3 is ~1.6x slower), NK=3 beats 1, 2, 4.
# RT=1 has a higher multiply rate but twice the SIMD-groups streaming K/V, and is memory-bound.
MIN_QUERY_ROWS = 16
_DEFAULT_TILING = (2, 3)
# Steps between loading an operand tile and multiplying with it.  0 (load at use) is ~17% slower;
# 2..5 are within 2% of each other.
_LOOKAHEAD = 3
_LOG2E = 1.4426950408889634
_libs = {}
_mapping_ok = None


def _lib(head_dim, heads, tiling, lse=False):
    key = (head_dim, heads, tiling, lse)
    if key not in _libs:
        out = "float" if lse else "bfloat"  # the LSE variant's outputs get merged: keep FP32 until then
        _libs[key] = torch.mps.compile_shader(_SRC % {
            "D": head_dim, "H": heads, "NAME": "flash_attn_lse" if lse else "flash_attn", "OT": out,
            "LSE_ARG": "                       device float* lse,\n" if lse else "",
            "BODY": _body(head_dim, heads, *tiling, lse=lse, out=out)})
    return _libs[key]


def is_available(head_dim=128):
    """True if the kernel compiles here and the simdgroup lane mapping it relies on holds."""
    global _mapping_ok
    if _mapping_ok is None:
        try:
            lib = _lib(head_dim, 1, _DEFAULT_TILING)
            src = torch.arange(64, dtype=torch.float32, device="mps")
            out = torch.zeros(32, dtype=torch.float32, device="mps")
            lib.probe_mapping(out, src, threads=(32, 1, 1), group_size=(32, 1, 1))
            _mapping_ok = bool(out.sum().item() == 32)
        except Exception:  # no MPS, old Metal, compile failure
            _mapping_ok = False
    return _mapping_ok and head_dim % 8 == 0


def flash_attention(q, k, v, scale=None, tiling=None):
    """Non-causal, unmasked attention on BF16 ``[B, L, H, D]`` MPS tensors; returns ``[B, Lq, H, D]``."""
    b, lq, h, d = q.shape
    lk = k.shape[1]
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16 and q.device.type == "mps"
    assert k.shape == v.shape and k.shape[0] == b and k.shape[2:] == (h, d) and d % 8 == 0
    tiling = tiling or _DEFAULT_TILING
    assert lq >= 8 * tiling[0] and lk >= 1
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    groups = -(-lq // (8 * tiling[0]))
    scale2 = float(scale or 1.0 / math.sqrt(d)) * _LOG2E  # the kernel's softmax is base 2
    _lib(d, h, tiling).flash_attn(out, q, k, v, lq, lk, scale2,
                                  threads=(32 * groups, b * h, 1), group_size=(32, 1, 1))
    return out


def flash_attention_lse(q, k, v, scale=None, tiling=None):
    """``flash_attention`` over one part of the keys, for merging with another part's
    (``merge_attention``): returns the FP32 output ``[B, Lq, H, D]`` and each row's natural-log sum
    of ``exp(scale * q.k)`` over these keys, ``[B, Lq, H]`` FP32."""
    b, lq, h, d = q.shape
    lk = k.shape[1]
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16 and q.device.type == "mps"
    assert k.shape == v.shape and k.shape[0] == b and k.shape[2:] == (h, d) and d % 8 == 0
    tiling = tiling or _DEFAULT_TILING
    assert lq >= 8 * tiling[0] and lk >= 1
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.empty((b, lq, h), dtype=torch.float32, device=q.device)
    groups = -(-lq // (8 * tiling[0]))
    scale2 = float(scale or 1.0 / math.sqrt(d)) * _LOG2E
    _lib(d, h, tiling, lse=True).flash_attn_lse(out, q, k, v, lq, lk, scale2, lse,
                                                threads=(32 * groups, b * h, 1), group_size=(32, 1, 1))
    return out, lse


def merge_attention(out_a, lse_a, out_b, lse_b, dtype=torch.bfloat16):
    """Attention over keys A and B together from each part's output and log-sum-exp: each output
    weighted by its share of the softmax denominator."""
    lse = torch.logaddexp(lse_a, lse_b)
    merged = out_a * torch.exp(lse_a - lse).unsqueeze(-1) + out_b * torch.exp(lse_b - lse).unsqueeze(-1)
    return merged.to(dtype)
