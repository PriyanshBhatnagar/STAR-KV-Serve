"""Load the fused K/V factors from ``fused_weights.pt`` (written by ``train.py``) for serving.

Per compressed layer the checkpoint holds::

    k_proj.VS.weight   [sum(head_ranks), hidden]   per-head rows back to back
    k_proj.U.weight    [H*Hd, sum(head_ranks)]     block-diagonal by head
    k_proj.head_ranks  [H]  int32
    v_proj.VS.weight   [rv, hidden]
    v_proj.U.weight    [H*Hd, rv]

K is repacked per head at its own width (its rank rounded up to ``K_ALIGN``), with
the heads side by side and ``k_head_offsets`` marking where each starts; entries
between a head's rank and its width are zero. V is zero-padded to a multiple of
``V_ALIGN``, since Triton specializes integer arguments divisible by 16.

With ``kv_quant`` the K and V rows are rotated by STAR-KV's block Hadamard (same
seed and order as ``fold_kv_hadamard``), folded into VS and U, and V is not padded.
"""

from dataclasses import dataclass

import torch

from starkv_serve.kernels.paged_quant import HADAMARD_SEED, QuantRow, block_hadamard, n_outlier

V_ALIGN = 16
# K code widths are rounded to the decode kernel's BLOCK_R: the rank axis is
# walked in tiles of this, and it keeps every head's segment an exact number of
# 32-byte memory sectors.
K_ALIGN = 16


def _align_up(n: int, multiple: int) -> int:
    return -(-int(n) // multiple) * multiple


@dataclass
class StarKVLayerWeights:
    layer_idx: int
    k_head_ranks: torch.Tensor    # [H] int32, real rank per KV head
    k_head_widths: torch.Tensor   # [H] int32, stored width: rank rounded to K_ALIGN
    k_head_offsets: torch.Tensor  # [H] int32, where each head's codes start
    k_width: int                  # sum of the widths: K codes per token
    k_max_rank: int               # widest head width (prefill padding only)
    k_down: torch.Tensor          # [k_width, hidden]  hidden -> K codes
    k_up: torch.Tensor            # [k_width, Hd]      K codes -> pre-RoPE K
    v_rank: int                 # stored V width: padded to V_ALIGN, or not with kv_quant
    v_rank_real: int            # V rank the checkpoint actually kept
    v_down: torch.Tensor        # [v_rank, hidden]        hidden -> V codes
    v_up: torch.Tensor          # [H*Hd, v_rank]          V codes -> V
    kv_quant: bool = False      # cache holds 4-bit codes (paged_quant.QuantRow)

    @property
    def cache_width(self) -> int:
        """Per-token codes for this layer: K codes, then V codes."""
        return self.k_width + self.v_rank

    def cache_bytes(self, elem_size: int) -> int:
        """Per-token cache bytes for this layer."""
        if self.kv_quant:
            return QuantRow(self.k_width, self.k_head_ranks.numel(), self.v_rank).nbytes
        return self.cache_width * elem_size


@dataclass
class StarKVCheckpoint:
    layers: dict[int, StarKVLayerWeights]
    skip_layers: tuple[int, ...]

    def is_compressed(self, layer_idx: int) -> bool:
        return layer_idx in self.layers


def load_fused_checkpoint(path: str, num_layers: int, num_kv_heads: int, head_dim: int,
                          dtype: torch.dtype = torch.float16, kv_quant: bool = False) -> StarKVCheckpoint:
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)

    layers: dict[int, StarKVLayerWeights] = {}
    for li in range(num_layers):
        pre = f"model.layers.{li}.self_attn."
        if f"{pre}k_proj.head_ranks" not in sd:
            if f"{pre}k_proj.Sigma_blocks.0.diag" in sd:
                raise ValueError(
                    f"{path}: layer {li} is in the legacy U/Sigma/V format. Serving "
                    "needs the fused format that train.py writes after fuse_and_prune."
                )
            continue  # uncompressed layer
        layers[li] = _load_layer(sd, pre, li, num_kv_heads, head_dim, dtype, kv_quant)

    if not layers:
        raise ValueError(f"{path}: no fused STAR-KV layers found")
    skip = tuple(li for li in range(num_layers) if li not in layers)
    return StarKVCheckpoint(layers=layers, skip_layers=skip)


def _rotate(vs, u, ranks, generator):
    """VS <- T^T VS and U <- U T per block of rows, in float64: U T T^T VS == U VS."""
    col = 0
    for r in ranks:
        T = block_hadamard(r, n_outlier(r), generator)
        vs[col:col + r] = (T.T @ vs[col:col + r].double()).float()
        u[:, col:col + r] = (u[:, col:col + r].double() @ T).float()
        col += r


def _load_layer(sd, pre, li, H, Hd, dtype, kv_quant) -> StarKVLayerWeights:
    ranks = [int(r) for r in sd[f"{pre}k_proj.head_ranks"].tolist()]
    if len(ranks) != H:
        raise ValueError(f"layer {li}: {len(ranks)} head ranks, model has {H} KV heads")

    k_vs = sd[f"{pre}k_proj.VS.weight"].float()
    k_u = sd[f"{pre}k_proj.U.weight"].float()
    if k_vs.shape[0] != sum(ranks) or k_u.shape != (H * Hd, sum(ranks)):
        raise ValueError(
            f"layer {li}: k_proj shapes VS{tuple(k_vs.shape)} U{tuple(k_u.shape)} "
            f"do not match head_ranks sum {sum(ranks)} with H={H}, Hd={Hd}"
        )
    if kv_quant:
        # A fresh generator per projection, as fold_kv_hadamard does. Rotate copies:
        # the state dict is memory-mapped and cached.
        k_vs, k_u = k_vs.clone(), k_u.clone()
        _rotate(k_vs, k_u, ranks, torch.Generator().manual_seed(HADAMARD_SEED))

    hidden = k_vs.shape[1]
    widths = [_align_up(r, K_ALIGN) for r in ranks]
    offsets = [sum(widths[:h]) for h in range(H)]
    k_width = sum(widths)
    max_r = max(widths)
    k_down = k_vs.new_zeros(k_width, hidden)
    k_up = k_vs.new_zeros(k_width, Hd)
    col = 0
    for h, r in enumerate(ranks):
        off = offsets[h]
        k_down[off:off + r] = k_vs[col:col + r]
        # U is block-diagonal by head; keep it rank-major so the kernel reads a
        # head's rows as one contiguous run.
        k_up[off:off + r] = k_u[h * Hd:(h + 1) * Hd, col:col + r].T
        col += r

    v_vs = sd[f"{pre}v_proj.VS.weight"].float()
    v_u = sd[f"{pre}v_proj.U.weight"].float()
    if v_u.shape != (H * Hd, v_vs.shape[0]):
        raise ValueError(f"layer {li}: v_proj shapes VS{tuple(v_vs.shape)} U{tuple(v_u.shape)}")

    rv = v_vs.shape[0]
    if kv_quant:
        v_vs, v_u = v_vs.clone(), v_u.clone()
        _rotate(v_vs, v_u, [rv], torch.Generator().manual_seed(HADAMARD_SEED))
    rv_pad = rv if kv_quant else _align_up(rv, V_ALIGN)
    v_down = v_vs.new_zeros(rv_pad, hidden)
    v_down[:rv] = v_vs
    v_up = v_u.new_zeros(H * Hd, rv_pad)
    v_up[:, :rv] = v_u

    return StarKVLayerWeights(
        layer_idx=li,
        k_head_ranks=torch.tensor(ranks, dtype=torch.int32),
        k_head_widths=torch.tensor(widths, dtype=torch.int32),
        k_head_offsets=torch.tensor(offsets, dtype=torch.int32),
        k_width=k_width,
        k_max_rank=max_r,
        k_down=k_down.to(dtype),
        k_up=k_up.to(dtype),
        v_rank=rv_pad,
        v_rank_real=rv,
        v_down=v_down.to(dtype),
        v_up=v_up.to(dtype),
        kv_quant=kv_quant,
    )
