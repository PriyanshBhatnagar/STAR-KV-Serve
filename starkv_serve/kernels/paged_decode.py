"""Paged STAR-KV decode attention with per-head rank, over vLLM's paged cache.

Two kernels per decode step:

* ``_paged_abx_fwd``: gathers each KV head's K codes through the block table
  (found via ``k_head_offsets``), rebuilds K with that head's own rank in
  ``cdiv(rank, BLOCK_R)`` GEMM trips, rotates it with vLLM's ``cos_sin_cache``
  (so K is rotated exactly as Q) and writes the logits.
* ``_paged_pv_fwd``: applies the probabilities of all query heads to the
  head-shared V codes in low-rank space (``P[H_q, L] @ V'[L, r_v]``). Each
  request's sequence is split into segments run as parallel programs and summed;
  the result is lifted to ``head_dim`` per query head at the end.

The cache view ``[num_blocks, block_size, width]`` may have arbitrary strides:
vLLM packs layers with different page sizes side by side inside a block.
"""

import torch
import triton
import triton.language as tl

BLOCK_L = 64
BLOCK_R = 16
NUM_WARPS = 4
# The float32 accumulator acc[H_q, BLOCK_V] must fit in shared memory.
MAX_BLOCK_V = 512
# Target number of value-pass programs per call (requests x segments).
TARGET_V_PROGRAMS = 256


@triton.jit
def _paged_abx_fwd(
    q_ptr, kc_ptr, bt_ptr, sl_ptr, b_ptr, r_ptr, off_ptr, cs_ptr, out_ptr,
    q_s0, q_s1, q_s2,
    kc_s0, kc_s1, kc_s2,
    bt_s0, bt_s1,
    b_s1, b_s2,
    cs_s0, cs_s1,
    o_s0, o_s1, o_s2,
    block_size, half_d,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_l = tl.program_id(axis=2)

    # Query head pid_h reads the K codes of its KV group; ranks are per KV head.
    kv_head = pid_h // GROUP_SIZE
    seq_len = tl.load(sl_ptr + pid_b)
    R = tl.load(r_ptr + kv_head)
    # Where this head's codes start, in both the cache row and U.
    off_h = tl.load(off_ptr + kv_head)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    l_mask = offs_l < seq_len
    offs_r = tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < half_d

    block_ids = tl.load(bt_ptr + pid_b * bt_s0 + (offs_l // block_size) * bt_s1,
                        mask=l_mask, other=0).to(tl.int64)
    # int64 before multiplying: block_id * block stride overflows int32 at
    # realistic cache sizes.
    slot_ptrs = kc_ptr + block_ids * kc_s0 + (offs_l % block_size).to(tl.int64) * kc_s1

    k_ptrs = slot_ptrs[:, None] + (off_h + offs_r[None, :]) * kc_s2
    b_ptrs = b_ptr + (off_h + offs_r[:, None]) * b_s1 + offs_d[None, :] * b_s2

    # Tiles past this request's length do no GEMM trips. Inside a CUDA graph the
    # grid is sized for max_model_len, so most tiles of a short request land here.
    n_trips = tl.where(pid_l * BLOCK_L < seq_len, tl.cdiv(R, BLOCK_R), 0)

    xb_0 = tl.zeros((BLOCK_L, BLOCK_D), dtype=tl.float32)
    xb_1 = tl.zeros((BLOCK_L, BLOCK_D), dtype=tl.float32)
    for r in range(0, n_trips):
        r_mask = offs_r < R - r * BLOCK_R
        x = tl.load(k_ptrs, mask=l_mask[:, None] & r_mask[None, :], other=0.0).to(tl.float32)
        b_0 = tl.load(b_ptrs, mask=r_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        b_1 = tl.load(b_ptrs + half_d * b_s2, mask=r_mask[:, None] & d_mask[None, :],
                      other=0.0).to(tl.float32)
        xb_0 = tl.dot(x, b_0, xb_0)
        xb_1 = tl.dot(x, b_1, xb_1)
        k_ptrs += BLOCK_R * kc_s2
        b_ptrs += BLOCK_R * b_s1

    cs_row = cs_ptr + offs_l[:, None] * cs_s0
    cos = tl.load(cs_row + offs_d[None, :] * cs_s1,
                  mask=l_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    sin = tl.load(cs_row + (half_d + offs_d[None, :]) * cs_s1,
                  mask=l_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    k_0 = xb_0 * cos - xb_1 * sin
    k_1 = xb_1 * cos + xb_0 * sin

    q_row = q_ptr + pid_b * q_s0 + pid_h * q_s1
    q_0 = tl.load(q_row + offs_d * q_s2, mask=d_mask, other=0.0).to(tl.float32)
    q_1 = tl.load(q_row + (half_d + offs_d) * q_s2, mask=d_mask, other=0.0).to(tl.float32)

    logits = tl.sum(q_0[None, :] * k_0, 1) + tl.sum(q_1[None, :] * k_1, 1)
    tl.store(out_ptr + pid_b * o_s0 + pid_h * o_s1 + offs_l * o_s2, logits, mask=l_mask)


@triton.jit
def _paged_pv_fwd(
    p_ptr, kc_ptr, bt_ptr, sl_ptr, out_ptr,
    p_s0, p_s1, p_s2,
    kc_s0, kc_s1, kc_s2,
    bt_s0, bt_s1,
    o_s0, o_s1, o_s2, o_s3,
    block_size, k_width, v_rank, num_heads, tiles_per_seg,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)
    seq_len = tl.load(sl_ptr + pid_b)

    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < num_heads
    # This program owns V-code columns [pid_c * BLOCK_V, pid_c * BLOCK_V + BLOCK_V).
    offs_v = pid_c * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < v_rank

    # ...and tiles [t_start, t_end) of the request; empty past the request's length.
    t_start = pid_s * tiles_per_seg
    t_end = tl.minimum(t_start + tiles_per_seg, tl.cdiv(seq_len, BLOCK_L))

    acc = tl.zeros((BLOCK_H, BLOCK_V), dtype=tl.float32)
    for t in range(t_start, t_end):
        offs_l = t * BLOCK_L + tl.arange(0, BLOCK_L)
        l_mask = offs_l < seq_len
        block_ids = tl.load(bt_ptr + pid_b * bt_s0 + (offs_l // block_size) * bt_s1,
                            mask=l_mask, other=0).to(tl.int64)
        slot_ptrs = kc_ptr + block_ids * kc_s0 + (offs_l % block_size).to(tl.int64) * kc_s1
        v = tl.load(slot_ptrs[:, None] + (k_width + offs_v[None, :]) * kc_s2,
                    mask=l_mask[:, None] & v_mask[None, :], other=0.0)
        p = tl.load(p_ptr + pid_b * p_s0 + offs_h[:, None] * p_s1 + offs_l[None, :] * p_s2,
                    mask=h_mask[:, None] & l_mask[None, :], other=0.0)
        # One GEMM per tile over all query heads against the head-shared V codes.
        acc = tl.dot(p, v, acc)

    tl.store(out_ptr + pid_b * o_s0 + pid_s * o_s1 + offs_h[:, None] * o_s2 + offs_v[None, :] * o_s3,
             acc, mask=h_mask[:, None] & v_mask[None, :])


def paged_starkv_decode(
    query: torch.Tensor,          # [n, H_q, head_dim], RoPE already applied
    kv_cache: torch.Tensor,       # [num_blocks, block_size, width] (any strides)
    block_table: torch.Tensor,    # [n, max_blocks] int32
    seq_lens: torch.Tensor,       # [n] int32, including the token just written
    max_seq_len: int,             # host-side bound on seq_lens: the step's max, or
                                  # max_model_len inside a CUDA graph
    k_up: torch.Tensor,           # [k_width, head_dim]  U^T, heads side by side
    k_ranks: torch.Tensor,        # [H_kv] int32, real rank per KV head
    k_head_offsets: torch.Tensor, # [H_kv] int32, where each head's codes start
    k_width: int,                 # K codes per token: the sum of the head widths
    v_up_by_q: torch.Tensor,      # [H_q, v_rank, head_dim]
    cos_sin_cache: torch.Tensor,  # [max_positions, head_dim] = cat(cos, sin)
    scale: float,
    v_programs: int = TARGET_V_PROGRAMS,  # value-pass programs to aim for; <= n means no split
) -> torch.Tensor:
    """Returns the attention output per query head, ``[n, H_q, head_dim]``."""
    n, num_heads, head_dim = query.shape
    num_kv_heads = k_ranks.numel()
    v_rank = v_up_by_q.shape[1]
    # k_width is a host int: reading it off k_head_offsets would sync the device
    # on every decode step and cannot be captured in a CUDA graph.
    assert head_dim % 2 == 0 and num_heads % num_kv_heads == 0
    assert kv_cache.shape[-1] == k_width + v_rank, "cache width != K codes + V codes"
    assert cos_sin_cache.shape[-1] == head_dim and cos_sin_cache.shape[0] >= max_seq_len
    device = query.device

    # Logits in the model's dtype: half the memory traffic of fp32, and the
    # softmax still reduces in fp32 internally.
    logits = torch.full((n, num_heads, max_seq_len), float("-inf"),
                        dtype=query.dtype, device=device)
    half_d = head_dim // 2
    grid = (n, num_heads, triton.cdiv(max_seq_len, BLOCK_L))
    _paged_abx_fwd[grid](
        query, kv_cache, block_table, seq_lens, k_up, k_ranks, k_head_offsets, cos_sin_cache, logits,
        query.stride(0), query.stride(1), query.stride(2),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0), block_table.stride(1),
        k_up.stride(0), k_up.stride(1),
        cos_sin_cache.stride(0), cos_sin_cache.stride(1),
        logits.stride(0), logits.stride(1), logits.stride(2),
        kv_cache.shape[1], half_d,
        GROUP_SIZE=num_heads // num_kv_heads,
        BLOCK_D=triton.next_power_of_2(half_d),
        BLOCK_R=BLOCK_R,
        BLOCK_L=BLOCK_L,
        num_warps=NUM_WARPS,
    )

    probs = torch.softmax(logits.mul_(scale), dim=-1)

    # Segment each request's tiles so requests x segments ~ v_programs. The count depends
    # only on (n, max_seq_len), so it is fixed for a captured CUDA graph.
    n_tiles = triton.cdiv(max_seq_len, BLOCK_L)
    tiles_per_seg = max(1, triton.cdiv(n * n_tiles, max(1, v_programs)))
    n_segs = triton.cdiv(n_tiles, tiles_per_seg)
    seg_acc = torch.empty((n, n_segs, num_heads, v_rank), dtype=torch.float32, device=device)
    block_v = min(MAX_BLOCK_V, max(16, triton.next_power_of_2(v_rank)))
    _paged_pv_fwd[(n, n_segs, triton.cdiv(v_rank, block_v))](
        probs, kv_cache, block_table, seq_lens, seg_acc,
        probs.stride(0), probs.stride(1), probs.stride(2),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0), block_table.stride(1),
        seg_acc.stride(0), seg_acc.stride(1), seg_acc.stride(2), seg_acc.stride(3),
        kv_cache.shape[1], k_width, v_rank, num_heads, tiles_per_seg,
        BLOCK_H=max(16, triton.next_power_of_2(num_heads)),
        BLOCK_V=block_v,
        BLOCK_L=BLOCK_L,
        num_warps=NUM_WARPS,
    )
    acc = seg_acc.sum(dim=1)

    # Lift V codes to head_dim per query head. bmm with the head as the batch axis:
    # a broadcasting matmul against [H_q, v_rank, head_dim] would first expand the
    # weight to [n, H_q, v_rank, head_dim], gigabytes at serving batch sizes.
    lifted = torch.bmm(acc.to(v_up_by_q.dtype).transpose(0, 1), v_up_by_q)  # [H_q, n, head_dim]
    return lifted.transpose(0, 1)
