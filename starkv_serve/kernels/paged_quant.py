"""Paged decode over the 4-bit STAR-KV cache.

STAR-KV's ``kv_quant.py`` moved onto vLLM's paged cache. The format and the
quantizer are unchanged: every K/V latent is quantized per token, each KV head's
leading ``int(0.2 * rank)`` channels (the latent follows VS's descending singular
values, so the leading ones carry the range) at 4 bits and the rest at 3 bits,
each block with one symmetric bf16 scale, after a block Hadamard rotation that
the checkpoint loader folds into VS and U. Codes are stored as 4-bit nibbles
either way, so every field stays byte aligned.

What changes relative to the contiguous kernels:

* One row per token holds all of a layer's fields (``QuantRow``), so a slot of
  vLLM's page is the whole token and the block table locates it, as in the
  bf16 cache. The row is padded to 16 bytes: Triton specializes integer
  arguments divisible by 16, which lets it vectorize the per-token loads.
* RoPE comes from the rotary embedding's own ``cos_sin_cache`` rows, which are
  already the ``[cos | sin]`` table ``abx_z`` builds, so K is rotated exactly as
  vLLM rotated Q, including LongChat's linear position scaling.
* Requests have different lengths: tiles past a request's ``seq_len`` do no work
  and their logits stay at -inf.
* The new tokens are quantized straight into their slots, prefill included.
  Prefill itself attends over its exact codes, as in STAR-KV; only decode reads
  the 4-bit cache.

Per decode step and layer: ``fold_q`` folds the query into U, ``_paged_abx_z``
computes the logits from the packed K, a softmax, and ``_paged_pv_fold`` applies
the probabilities to the packed V split over the token axis. See ``kv_quant.py``
for why each kernel is shaped the way it is. MHA and head_dim 128 only.
"""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

OUTLIER_RATIO = 0.2
HADAMARD_SEED = 1234
EPS = 1e-5                       # amax is clamped here so an all-zero block has a scale
EPS_ = tl.constexpr(EPS)
HEAD_DIM = 128


def n_outlier(rank: int) -> int:
    return int(OUTLIER_RATIO * rank)


def block_hadamard(rank: int, n_out: int, generator: torch.Generator) -> torch.Tensor:
    """rank x rank orthonormal T = diag(H_outlier, H_inlier), float64.

    STAR-KV's kv_quant.py, so the same seed gives the same rotation, but pinned to
    the CPU: vLLM builds the model under a CUDA default device. Each block is a
    Hadamard on its largest power-of-two leading part (identity on the remainder)
    with random sign flips on the rows.
    """
    T = torch.eye(rank, dtype=torch.float64, device="cpu")
    for a, n in ((0, n_out), (n_out, rank - n_out)):
        if n <= 1:
            continue
        m = 1 << (n.bit_length() - 1)
        H = torch.ones(1, 1, dtype=torch.float64, device="cpu")
        while H.shape[0] < m:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        signs = torch.where(torch.rand(m, generator=generator, device="cpu") < 0.5, -1.0, 1.0).to(torch.float64)
        T[a:a + m, a:a + m] = signs[:, None] * H / math.sqrt(m)
    return T


@dataclass(frozen=True)
class QuantRow:
    """Byte layout of one token in a 4-bit layer's cache page.

      [0, ks_off)        K codes, lanes 2k and 2k+1 in byte k (k_width / 2 bytes)
      [ks_off, vs_off)   K scales, (outlier, inlier) bf16 per KV head
      [vs_off, vq_off)   V scales, (outlier, inlier) bf16
      [vq_off, ...)      V codes: outlier channels first, then inliers, so no
                         byte straddles the two blocks
    """
    k_width: int
    num_kv_heads: int
    v_rank: int

    @property
    def v_nout(self) -> int:
        return n_outlier(self.v_rank)

    @property
    def ks_off(self) -> int:
        return self.k_width // 2

    @property
    def vs_off(self) -> int:
        return self.ks_off + 4 * self.num_kv_heads

    @property
    def vq_off(self) -> int:
        return self.vs_off + 4

    @property
    def v_nbo(self) -> int:
        return (self.v_nout + 1) // 2

    @property
    def v_bytes(self) -> int:
        return self.v_nbo + (self.v_rank - self.v_nout + 1) // 2

    @property
    def nbytes(self) -> int:
        return -(-(self.vq_off + self.v_bytes) // 16) * 16


def k_tables(ranks: list[int], device) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """(outlier lanes per head, head of every K lane, per-16-lane-tile metadata).

    A head's width is its rank rounded up to 16, so no 16-lane tile straddles two
    heads and ``_paged_abx_z`` walks the whole rank axis as one loop. Per tile:
    its head, how many of its lanes are outliers, and whether it is its head's last.
    """
    nouts = [n_outlier(r) for r in ranks]
    lanes, head, t_out, last = [], [], [], []
    for h, (r, no) in enumerate(zip(ranks, nouts)):
        tiles = -(-r // 16)
        lanes += [h] * (16 * tiles)
        for j in range(tiles):
            head.append(h)
            t_out.append(max(0, min(16, no - 16 * j)))
            last.append(int(j == tiles - 1))
    t = lambda v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    return t(nouts), t(lanes), (t(head), t(t_out), t(last))


# ---------------------------------------------------------------------------
# Write side: quantize and pack the step's tokens into their slots. Bit for bit
# the quantizer of kv_quant._q: IEEE division (div_rn) and round-half-even.
# ---------------------------------------------------------------------------

@triton.jit
def _slot_row(cache_ptr, slot_ptr, i, c_s0, c_s1, block_size):
    # CUDA-graph padding tokens carry slot -1; send them to the reserved null block.
    slot = tl.maximum(tl.load(slot_ptr + i), 0).to(tl.int64)
    return cache_ptr + (slot // block_size) * c_s0 + (slot % block_size) * c_s1


@triton.jit
def _pack_k(x_ptr, cache_ptr, slot_ptr, r_ptr, off_ptr, nout_ptr,
            x_s0, c_s0, c_s1, block_size, KS_OFF, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    row = _slot_row(cache_ptr, slot_ptr, i, c_s0, c_s1, block_size)
    R = tl.load(r_ptr + h)
    NOUT = tl.load(nout_ptr + h)
    off = tl.load(off_ptr + h)
    W_H = ((R + 15) // 16) * 16               # stored width; lanes past R are zero
    lane = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i * x_s0 + off + lane, mask=lane < W_H, other=0.0).to(tl.float32)
    is_out = lane < NOUT
    a_out = tl.maximum(tl.max(tl.where(is_out, tl.abs(x), 0.0), 0), EPS_)
    a_in = tl.maximum(tl.max(tl.where((lane >= NOUT) & (lane < W_H), tl.abs(x), 0.0), 0), EPS_)
    s_out = tl.extra.cuda.libdevice.div_rn(a_out, 7.0)
    s_in = tl.extra.cuda.libdevice.div_rn(a_in, 3.0)
    q = tl.where(is_out,
                 tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(x, s_out)), -8.0), 7.0),
                 tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(x, s_in)), -4.0), 3.0))
    n = (q + 8.0).to(tl.int32)
    even, odd = tl.split(tl.reshape(n, (BLOCK // 2, 2)))
    k = tl.arange(0, BLOCK // 2)
    tl.store(row + off // 2 + k, (even | (odd << 4)).to(tl.uint8), mask=2 * k < W_H)
    ks = (row + KS_OFF).to(tl.pointer_type(tl.bfloat16)) + 2 * h
    tl.store(ks, tl.where(NOUT > 0, s_out, 1.0).to(tl.bfloat16))
    tl.store(ks + 1, s_in.to(tl.bfloat16))


@triton.jit
def _pack_v(x_ptr, cache_ptr, slot_ptr, x_s0, c_s0, c_s1, block_size, VS_OFF, VQ_OFF,
            R, NOUT, NBO, NB, BLOCK: tl.constexpr, BLOCK_B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    row = _slot_row(cache_ptr, slot_ptr, i, c_s0, c_s1, block_size)
    c = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i * x_s0 + c, mask=c < R, other=0.0).to(tl.float32)
    s_out = tl.extra.cuda.libdevice.div_rn(tl.maximum(tl.max(tl.where(c < NOUT, tl.abs(x), 0.0), 0), EPS_), 7.0)
    s_in = tl.extra.cuda.libdevice.div_rn(tl.maximum(tl.max(tl.where((c >= NOUT) & (c < R), tl.abs(x), 0.0), 0), EPS_), 3.0)
    # Each byte takes its two channels from its own block; an odd block ends in a
    # padding nibble.
    j = tl.arange(0, BLOCK_B)
    in_out = j < NBO
    c_lo = tl.where(in_out, 2 * j, NOUT + 2 * (j - NBO))
    c_end = tl.where(in_out, NOUT, R)
    ok_lo = (j < NB) & (c_lo < c_end)
    ok_hi = (j < NB) & (c_lo + 1 < c_end)
    xl = tl.load(x_ptr + i * x_s0 + c_lo, mask=ok_lo, other=0.0).to(tl.float32)
    xh = tl.load(x_ptr + i * x_s0 + c_lo + 1, mask=ok_hi, other=0.0).to(tl.float32)
    s = tl.where(in_out, s_out, s_in)
    hi_q = tl.where(in_out, 7.0, 3.0)
    lo_q = tl.where(in_out, -8.0, -4.0)
    nl = tl.where(ok_lo, tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(xl, s)), lo_q), hi_q) + 8.0, 8.0)
    nh = tl.where(ok_hi, tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(xh, s)), lo_q), hi_q) + 8.0, 8.0)
    tl.store(row + VQ_OFF + j, (nl.to(tl.int32) | (nh.to(tl.int32) << 4)).to(tl.uint8), mask=j < NB)
    vs = (row + VS_OFF).to(tl.pointer_type(tl.bfloat16))
    tl.store(vs, tl.where(NOUT > 0, s_out, 1.0).to(tl.bfloat16))
    tl.store(vs + 1, s_in.to(tl.bfloat16))


def pack_kv(k_codes, v_codes, cache, slot_mapping, row: QuantRow, k_ranks, k_offsets, k_nouts):
    """Quantize the step's codes into their slots of ``cache`` [num_blocks, block_size, row bytes]."""
    n = k_codes.shape[0]
    assert k_codes.stride(-1) == 1 and v_codes.stride(-1) == 1
    _pack_k[(n, row.num_kv_heads)](
        k_codes, cache, slot_mapping, k_ranks, k_offsets, k_nouts,
        k_codes.stride(0), cache.stride(0), cache.stride(1), cache.shape[1], row.ks_off,
        BLOCK=HEAD_DIM)
    _pack_v[(n,)](
        v_codes, cache, slot_mapping, v_codes.stride(0), cache.stride(0), cache.stride(1),
        cache.shape[1], row.vs_off, row.vq_off, row.v_rank, row.v_nout, row.v_nbo, row.v_bytes,
        BLOCK=triton.next_power_of_2(row.v_rank), BLOCK_B=triton.next_power_of_2(row.v_bytes))


# ---------------------------------------------------------------------------
# K: fold the query into U, then logits straight from the 4-bit codes
# ---------------------------------------------------------------------------

@triton.jit
def _fold_q(q_ptr, u_ptr, hol_ptr, ab_ptr, q_s0, q_s1, u_s0, ab_s0, ab_s1,
            W, SCALE, BLOCK_W: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    lane = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    ok = lane < W
    i = tl.arange(0, 64)
    head = tl.load(hol_ptr + lane, mask=ok, other=0).to(tl.int64)
    qb = q_ptr + b * q_s0 + head[:, None] * q_s1
    a0 = tl.load(qb + i[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    a1 = tl.load(qb + (i + 64)[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    u0 = tl.load(u_ptr + lane[:, None] * u_s0 + i[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    u1 = tl.load(u_ptr + lane[:, None] * u_s0 + (i + 64)[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    base = ab_ptr + b * ab_s0 + lane[:, None] * ab_s1
    tl.store(base + i[None, :], ((a0 * u0 + a1 * u1) * SCALE).to(tl.float16), mask=ok[:, None])
    tl.store(base + (i + 64)[None, :], ((a1 * u0 - a0 * u1) * SCALE).to(tl.float16), mask=ok[:, None])


def fold_q(q, k_up, head_of_lane, scale):
    """q [n, H, 128] (RoPE applied), U^T [W, 128] -> A|B [n, W, 128] fp16, with

      A[r,i] = q_i U[r,i] + q_{i+64} U[r,i+64]      B[r,i] = q_{i+64} U[r,i] - q_i U[r,i+64]

    so that q . RoPE_l(U^T x) == sum_r x[r] * ([cos_l | sin_l] . AB[r]). The logit
    scale folds in here too.
    """
    n, W = q.shape[0], k_up.shape[0]
    assert q.stride(-1) == 1 and k_up.stride(-1) == 1
    ab = torch.empty(n, W, HEAD_DIM, device=q.device, dtype=torch.float16)
    _fold_q[(n, triton.cdiv(W, 64))](q, k_up, head_of_lane, ab, q.stride(0), q.stride(1),
                                     k_up.stride(0), ab.stride(0), ab.stride(1), W,
                                     float(scale), BLOCK_W=64)
    return ab


@triton.jit
def _paged_abx_z(ab_ptr, cache_ptr, bt_ptr, sl_ptr, cs_ptr, out_ptr, th_ptr, tn_ptr, tl_ptr,
                 ab_s0, ab_s1, c_s0, c_s1, bt_s0, bt_s1, cs_s0, cs_s1, o_s0, o_s1,
                 block_size, KS_OFF, NT, BLOCK_L: tl.constexpr, ACC16: tl.constexpr):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_l = tl.program_id(1)
    seq_len = tl.load(sl_ptr + pid_b)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = offs_l < seq_len
    # Tiles past this request's length do no work. Inside a CUDA graph the grid is
    # sized for max_model_len, so most tiles of a short request land here.
    nt = tl.where(pid_l * BLOCK_L < seq_len, NT, 0)
    block_ids = tl.load(bt_ptr + pid_b * bt_s0 + (offs_l // block_size) * bt_s1,
                        mask=l_mask, other=0).to(tl.int64)
    row = cache_ptr + block_ids * c_s0 + (offs_l % block_size).to(tl.int64) * c_s1
    ks_row = (row + KS_OFF).to(tl.pointer_type(tl.bfloat16))
    offs_c = tl.arange(0, 128)
    offs_k = tl.arange(0, 8)
    offs_r = tl.arange(0, 16)
    # [cos | sin] for this token tile, once, shared by every head.
    cs = tl.load(cs_ptr + offs_l[:, None] * cs_s0 + offs_c[None, :] * cs_s1,
                 mask=l_mask[:, None], other=0.0).to(tl.float16)
    ab_b = ab_ptr + pid_b * ab_s0
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)
    for t in range(0, nt):
        h = tl.load(th_ptr + t)
        n_out = tl.load(tn_ptr + t)
        is_last = tl.load(tl_ptr + t)
        ab = tl.load(ab_b + (t * 16 + offs_r)[:, None] * ab_s1 + offs_c[None, :])        # [16, 128]
        if ACC16:
            # fp16 accumulation over K=128 of |cos|,|sin| <= 1 times A|B: ~0.1% error,
            # well under the 4-bit step, at twice the fp32-accumulate MMA rate.
            z = tl.dot(cs, tl.trans(ab), out_dtype=tl.float16).to(tl.float32)
        else:
            z = tl.dot(cs, tl.trans(ab))
        raw = tl.load(row[:, None] + (t * 8 + offs_k)[None, :], mask=l_mask[:, None], other=0x88)
        # fp32 bits 0x4B000000 | n == 2^23 + n: a nibble to a float with an OR.
        lo = ((raw & 0x0F).to(tl.int32) | 0x4B000000).to(tl.float32, bitcast=True)
        hi = (((raw >> 4) & 0x0F).to(tl.int32) | 0x4B000000).to(tl.float32, bitcast=True)
        v = tl.reshape(tl.join(lo, hi), (BLOCK_L, 16)) - 8388616.0                            # n - 8
        so = tl.load(ks_row + 2 * h, mask=l_mask, other=0.0).to(tl.float32)
        si = tl.load(ks_row + 2 * h + 1, mask=l_mask, other=0.0).to(tl.float32)
        sc = tl.where((offs_r < n_out)[None, :], so[:, None], si[:, None])
        acc += tl.sum(v * sc * z, 1)
        # A head's logit is complete at its last tile: store it, start the next.
        tl.store(out_ptr + pid_b * o_s0 + h * o_s1 + offs_l, acc, mask=l_mask & (is_last != 0))
        acc = tl.where(is_last != 0, 0.0, acc)


# ---------------------------------------------------------------------------
# V: split over tokens, the per-token scale folded into the probabilities
# ---------------------------------------------------------------------------

@triton.jit
def _paged_pv_fold(p_ptr, cache_ptr, bt_ptr, sl_ptr, out_ptr,
                   p_s0, p_s1, c_s0, c_s1, bt_s0, bt_s1, o_s0, o_s1,
                   block_size, VS_OFF, VQ_OFF, H, R, NOUT, NBO, T_OUT, SEG_LEN,
                   BLOCK_H: tl.constexpr, BLOCK_VB: tl.constexpr, BLOCK_L: tl.constexpr):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_t = tl.program_id(1)
    pid_s = tl.program_id(2)
    seq_len = tl.load(sl_ptr + pid_b)
    # A tile sits wholly inside one block, so one scale per token covers all of it
    # and folds into p instead of multiplying every V element.
    is_out = pid_t < T_OUT
    t = tl.where(is_out, pid_t, pid_t - T_OUT)
    byte0 = tl.where(is_out, 0, NBO)
    ch0 = tl.where(is_out, 0, NOUT)
    ch_end = tl.where(is_out, NOUT, R)
    blk = tl.where(is_out, 0, 1)
    offs_h = tl.arange(0, BLOCK_H)
    offs_bt = t * BLOCK_VB + tl.arange(0, BLOCK_VB)
    offs_c = ch0 + t * 2 * BLOCK_VB + tl.arange(0, 2 * BLOCK_VB)
    h_mask = offs_h < H
    c_mask = offs_c < ch_end
    b_mask = offs_bt * 2 < ch_end - ch0
    acc = tl.zeros((BLOCK_H, 2 * BLOCK_VB), dtype=tl.float32)
    l_beg = pid_s * SEG_LEN
    l_end = tl.minimum(l_beg + SEG_LEN, seq_len)
    for l0 in range(l_beg, l_end, BLOCK_L):
        offs_l = l0 + tl.arange(0, BLOCK_L)
        l_mask = offs_l < l_end
        block_ids = tl.load(bt_ptr + pid_b * bt_s0 + (offs_l // block_size) * bt_s1,
                            mask=l_mask, other=0).to(tl.int64)
        row = cache_ptr + block_ids * c_s0 + (offs_l % block_size).to(tl.int64) * c_s1
        p = tl.load(p_ptr + pid_b * p_s0 + offs_h[:, None] * p_s1 + offs_l[None, :],
                    mask=h_mask[:, None] & l_mask[None, :], other=0.0)
        s = tl.load((row + VS_OFF).to(tl.pointer_type(tl.bfloat16)) + blk, mask=l_mask, other=0.0)
        ps = (p.to(tl.float32) * s.to(tl.float32)[None, :]).to(p_ptr.dtype.element_ty)
        raw = tl.load(row[:, None] + VQ_OFF + (byte0 + offs_bt)[None, :],
                      mask=l_mask[:, None] & b_mask[None, :], other=0x88)
        lo = ((raw & 0x0F).to(tl.int32) | 0x4B000000).to(tl.float32, bitcast=True)
        hi = (((raw >> 4) & 0x0F).to(tl.int32) | 0x4B000000).to(tl.float32, bitcast=True)
        x = (tl.reshape(tl.join(lo, hi), (BLOCK_L, 2 * BLOCK_VB)) - 8388616.0).to(p_ptr.dtype.element_ty)
        acc = tl.dot(ps, x, acc)
    tl.atomic_add(out_ptr + pid_b * o_s0 + offs_h[:, None] * o_s1 + offs_c[None, :], acc,
                  mask=h_mask[:, None] & c_mask[None, :] & (l_beg < seq_len), sem="relaxed")


def paged_quant_decode(
    query: torch.Tensor,          # [n, H, 128], RoPE already applied
    kv_cache: torch.Tensor,       # [num_blocks, block_size, row.nbytes] uint8 (any strides)
    block_table: torch.Tensor,    # [n, max_blocks] int32
    seq_lens: torch.Tensor,       # [n] int32, including the token just written
    max_seq_len: int,             # host-side bound on seq_lens (max_model_len in a CUDA graph)
    row: QuantRow,
    k_up: torch.Tensor,           # [k_width, 128]  U^T, heads side by side
    head_of_lane: torch.Tensor,   # [k_width] int32
    k_tiles: tuple,               # per-16-lane-tile (head, outlier lanes, is last) from k_tables
    v_up_by_q: torch.Tensor,      # [H, v_rank, 128]
    cos_sin_cache: torch.Tensor,  # [max_positions, 128] = cat(cos, sin)
    scale: float,
    target_v_programs: int = 2048,
) -> torch.Tensor:
    """Returns the attention output per query head, ``[n, H, 128]``."""
    n, H, D = query.shape
    assert D == HEAD_DIM and H == row.num_kv_heads, "4-bit decode: MHA, head_dim 128"
    assert kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] == row.nbytes
    assert cos_sin_cache.shape[-1] == D and cos_sin_cache.shape[0] >= max_seq_len

    ab = fold_q(query, k_up, head_of_lane, scale)
    logits = torch.full((n, H, max_seq_len), float("-inf"), dtype=query.dtype, device=query.device)
    th, tn, tlast = k_tiles
    # Swept in kv_quant.py on an RTX 4090: 128-token tiles with 4 stages at long
    # context, 64 and 3 below 16k where there are too few programs otherwise.
    block_l, stages = (128, 4) if max_seq_len >= 16384 else (64, 3)
    _paged_abx_z[(n, triton.cdiv(max_seq_len, block_l))](
        ab, kv_cache, block_table, seq_lens, cos_sin_cache, logits, th, tn, tlast,
        ab.stride(0), ab.stride(1), kv_cache.stride(0), kv_cache.stride(1),
        block_table.stride(0), block_table.stride(1), cos_sin_cache.stride(0), cos_sin_cache.stride(1),
        logits.stride(0), logits.stride(1), kv_cache.shape[1], row.ks_off, th.numel(),
        BLOCK_L=block_l, ACC16=True, num_warps=4, num_stages=stages)
    probs = torch.softmax(logits, dim=-1)

    R, nout = row.v_rank, row.v_nout
    block_vb, block_l = 64, 32
    t_out = triton.cdiv(nout, 2 * block_vb) if nout else 0
    nt = t_out + triton.cdiv(R - nout, 2 * block_vb)
    # Segments per request so requests x tiles x segments ~ target_v_programs. Depends
    # only on (n, max_seq_len), so it is fixed for a captured CUDA graph.
    nseg = max(1, min(triton.cdiv(target_v_programs, n * nt), triton.cdiv(max_seq_len, block_l)))
    seg = triton.cdiv(triton.cdiv(max_seq_len, nseg), block_l) * block_l
    nseg = triton.cdiv(max_seq_len, seg)
    pv = torch.zeros(n, H, R, device=query.device, dtype=torch.float32)
    _paged_pv_fold[(n, nt, nseg)](
        probs, kv_cache, block_table, seq_lens, pv,
        probs.stride(0), probs.stride(1), kv_cache.stride(0), kv_cache.stride(1),
        block_table.stride(0), block_table.stride(1), pv.stride(0), pv.stride(1),
        kv_cache.shape[1], row.vs_off, row.vq_off, H, R, nout, row.v_nbo, t_out, seg,
        BLOCK_H=max(16, triton.next_power_of_2(H)), BLOCK_VB=block_vb, BLOCK_L=block_l, num_warps=4)

    # Lift V codes to head_dim per query head, with the head as the bmm batch axis.
    lifted = torch.bmm(pv.to(v_up_by_q.dtype).transpose(0, 1), v_up_by_q)   # [H, n, 128]
    return lifted.transpose(0, 1)
