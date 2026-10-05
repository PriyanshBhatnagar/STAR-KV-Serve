"""Load STAR-KV's fused checkpoint (``train.py --output fused_weights.pt``) for serving.

The checkpoint is the hand-off between the two halves of this repo: training
and HF-side evaluation stay exactly as in STAR-KV, and serving reads the same
file. Only the fused decomposition tensors are consumed; base-model weights come
from the HF model vLLM already loads.

Per compressed layer the fused format holds (see STAR-KV ``model.py``,
``FusedDecomposeLinear_headwise`` / ``FusedDecomposeLinear``)::

    k_proj.VS.weight   [sum(head_ranks), hidden]   per-head rows back to back
    k_proj.U.weight    [H*Hd, sum(head_ranks)]     block-diagonal by head
    k_proj.head_ranks  [H]  int32
    v_proj.VS.weight   [rv, hidden]
    v_proj.U.weight    [H*Hd, rv]

K is repacked here exactly the way ``export_kproj_for_triton`` does it: each
head padded with zero rows/columns up to the layer's max rank, so storage has
uniform strides while ``k_head_ranks`` carries each head's real rank for the
dynamic-rank kernel. Padding is exact because the padded entries are zero.
"""

from dataclasses import dataclass

import torch


@dataclass
class StarKVLayerWeights:
    layer_idx: int
    k_head_ranks: torch.Tensor  # [H] int32, real rank per KV head
    k_max_rank: int             # padded per-head width stored in the cache
    k_down: torch.Tensor        # [H*k_max_rank, hidden]  hidden -> K codes
    k_up: torch.Tensor          # [H, Hd, k_max_rank]     K codes -> pre-RoPE K
    v_rank: int
    v_down: torch.Tensor        # [v_rank, hidden]        hidden -> V codes
    v_up: torch.Tensor          # [H*Hd, v_rank]          V codes -> V

    @property
    def cache_width(self) -> int:
        """Per-token cache elements for this layer: padded K codes, then V codes."""
        return self.k_head_ranks.numel() * self.k_max_rank + self.v_rank


@dataclass
class StarKVCheckpoint:
    layers: dict[int, StarKVLayerWeights]
    skip_layers: tuple[int, ...]

    def is_compressed(self, layer_idx: int) -> bool:
        return layer_idx in self.layers


def load_fused_checkpoint(path: str, num_layers: int, num_kv_heads: int,
                          head_dim: int, dtype: torch.dtype = torch.float16) -> StarKVCheckpoint:
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
        layers[li] = _load_layer(sd, pre, li, num_kv_heads, head_dim, dtype)

    if not layers:
        raise ValueError(f"{path}: no fused STAR-KV layers found")
    skip = tuple(li for li in range(num_layers) if li not in layers)
    return StarKVCheckpoint(layers=layers, skip_layers=skip)


def _load_layer(sd, pre, li, H, Hd, dtype) -> StarKVLayerWeights:
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

    max_r = max(ranks)
    hidden = k_vs.shape[1]
    k_down = k_vs.new_zeros(H * max_r, hidden)
    k_up = k_vs.new_zeros(H, Hd, max_r)
    col = 0
    for h, r in enumerate(ranks):
        k_down[h * max_r:h * max_r + r] = k_vs[col:col + r]
        k_up[h, :, :r] = k_u[h * Hd:(h + 1) * Hd, col:col + r]
        col += r

    v_vs = sd[f"{pre}v_proj.VS.weight"].float()
    v_u = sd[f"{pre}v_proj.U.weight"].float()
    if v_u.shape != (H * Hd, v_vs.shape[0]):
        raise ValueError(f"layer {li}: v_proj shapes VS{tuple(v_vs.shape)} U{tuple(v_u.shape)}")

    return StarKVLayerWeights(
        layer_idx=li,
        k_head_ranks=torch.tensor(ranks, dtype=torch.int32),
        k_max_rank=max_r,
        k_down=k_down.to(dtype),
        k_up=k_up.to(dtype),
        v_rank=v_vs.shape[0],
        v_down=v_vs.to(dtype),
        v_up=v_u.to(dtype),
    )
