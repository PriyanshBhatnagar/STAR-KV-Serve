"""vLLM attention backend for STAR-KV's compressed layers.

Cache layout, per token: ``[K codes, each KV head at its own width | V codes]``,
or with ``kv_quant`` the same codes at 4 bits plus their scales
(``paged_quant.QuantRow``). The layer declares that as a plain
``FullAttentionSpec`` with one head slot and its row's bytes per token, so vLLM
groups it with the model's uncompressed layers and sums their page sizes instead
of padding every layer to the widest one.

Decode runs the low-rank paged attention kernel (``kernels/paged_decode.py``,
or ``kernels/paged_quant.py`` for the 4-bit cache).
Prefill is dense, as in STAR-KV's HF path: this step's codes are lifted back to
full K and V and attended with FlashAttention. That needs every prompt in one
step with no cached context, so the plugin turns prefix caching and chunked
prefill off (``config.StarKVEngineDefaults``).

CUDA graphs: pure single-token decode batches can be captured whole. Inside a
captured graph nothing may depend on the step's lengths, so decode sizes its
logits buffer and kernel grid from ``max_model_len`` there.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from vllm.config import CUDAGraphMode, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    Attention,
    _encode_layer_name,
    unified_attention_with_output,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import FullAttentionSpec

from starkv_serve.checkpoint import StarKVLayerWeights
from starkv_serve.kernels.paged_decode import paged_starkv_decode
from starkv_serve.kernels.paged_quant import HEAD_DIM, QuantRow, k_tables, pack_kv, paged_quant_decode

# Cap on the decode logits buffer (n * H_q * span float32 elements). vLLM's
# memory profiling runs a prefill, not decode, so this buffer is not accounted
# for when the KV cache is sized; larger decode batches are split into chunks.
_MAX_DECODE_LOGIT_ELEMS = 1 << 26


@dataclass
class StarKVMetadata:
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    # Host-side prefill bookkeeping. Prefill never runs inside a CUDA graph.
    prefill_seq_lens_cpu: list[int] | None
    prefill_query_start_loc_cpu: list[int] | None


class StarKVMetadataBuilder(AttentionMetadataBuilder[StarKVMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Decodes first, so decode and prefill each see a contiguous slice.
        self._init_reorder_batch_threshold(1)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> StarKVMetadata:
        m = common_attn_metadata
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            m, decode_threshold=1)

        prefill_seq_lens_cpu = prefill_qsl_cpu = None
        if num_prefills:
            qsl = m.query_start_loc_cpu[num_decodes:num_decodes + num_prefills + 1]
            prefill_qsl_cpu = (qsl - qsl[0]).tolist()
            if m.seq_lens_cpu_upper_bound is not None:
                prefill_seq_lens_cpu = (
                    m.seq_lens_cpu_upper_bound[num_decodes:num_decodes + num_prefills].tolist())

        return StarKVMetadata(
            num_actual_tokens=m.num_actual_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            max_seq_len=m.max_seq_len,
            seq_lens=m.seq_lens,
            block_table=m.block_table_tensor,
            slot_mapping=m.slot_mapping,
            prefill_seq_lens_cpu=prefill_seq_lens_cpu,
            prefill_query_start_loc_cpu=prefill_qsl_cpu,
        )


class StarKVAttentionImpl(AttentionImpl[StarKVMetadata]):
    def __init__(self, num_heads, head_size, scale, num_kv_heads=None, alibi_slopes=None,
                 sliding_window=None, kv_cache_dtype="auto", logits_soft_cap=None,
                 attn_type=AttentionType.DECODER, kv_sharing_target_layer_name=None,
                 *, starkv_layer: StarKVLayerWeights, rotary_emb):
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap:
            raise NotImplementedError("STAR-KV attention: no alibi, sliding window or softcap")
        if attn_type != AttentionType.DECODER or kv_sharing_target_layer_name is not None:
            raise NotImplementedError("STAR-KV attention: decoder self-attention only")
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(f"STAR-KV attention: kv_cache_dtype {kv_cache_dtype!r}")
        if starkv_layer.kv_quant and ((num_kv_heads or num_heads) != num_heads or head_size != HEAD_DIM):
            raise NotImplementedError("STAR-KV 4-bit cache: MHA with head_dim 128 only")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.w = starkv_layer
        self.rotary_emb = rotary_emb
        self.max_model_len = get_current_vllm_config().model_config.max_model_len
        try:
            self.fa_version = get_flash_attn_version()
        except (TypeError, AssertionError, ValueError):
            self.fa_version = 2

    def to_device(self, device: torch.device, dtype: torch.dtype) -> None:
        """Called once from the layer's post-load hook, before any graph capture."""
        w, H, Hd = self.w, self.num_kv_heads, self.head_size
        self.k_ranks = w.k_head_ranks.to(device)
        self.k_head_offsets = w.k_head_offsets.to(device)
        self.k_width = w.k_width
        # Decode reads U through the same per-head offsets as the cache.
        self.k_up = w.k_up.to(device=device, dtype=dtype).contiguous()

        # Prefill lifts every head in one bmm, which needs a uniform width, so the
        # codes are gathered into a padded view for that call only. Indices past a
        # head's width point at its first column; the U rows they meet are zero.
        max_w = w.k_max_rank
        widths = w.k_head_widths.tolist()
        offsets = w.k_head_offsets.tolist()
        idx = torch.empty(H * max_w, dtype=torch.long)
        k_up_padded = w.k_up.new_zeros(H, max_w, Hd)
        for h, (off, width) in enumerate(zip(offsets, widths)):
            idx[h * max_w:h * max_w + width] = torch.arange(off, off + width)
            idx[h * max_w + width:(h + 1) * max_w] = off
            k_up_padded[h, :width] = w.k_up[off:off + width]
        self.k_prefill_gather = idx.to(device)
        self.k_up_T = k_up_padded.to(device=device, dtype=dtype).contiguous()
        self.v_up = w.v_up.to(device=device, dtype=dtype).contiguous()
        self.v_up_by_q = (self.v_up.view(H, Hd, w.v_rank)
                          .repeat_interleave(self.num_heads // H, dim=0)
                          .transpose(-1, -2).contiguous())
        self.kv_down = torch.cat([w.k_down, w.v_down], dim=0).to(device=device, dtype=dtype).contiguous()

        self.row = None
        if w.kv_quant:
            self.row = QuantRow(w.k_width, H, w.v_rank)
            self.k_nouts, self.head_of_lane, self.k_tiles = k_tables(w.k_head_ranks.tolist(), device)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("STAR-KV attention: fused output quantization")
        if attn_metadata is None:  # profiling run
            return output.fill_(0)

        md: StarKVMetadata = attn_metadata
        n = md.num_actual_tokens
        Hq, Hd = self.num_heads, self.head_size
        cache = kv_cache.squeeze(1)  # [num_blocks, block_size, cache_width]
        if self.row is None:
            # CUDA-graph padding tokens carry slot -1; send them to the reserved null block.
            slots = md.slot_mapping[:n].clamp(min=0)
            block_size = cache.shape[1]
            cache[slots // block_size, slots % block_size] = torch.cat(
                [key[:n], value[:n]], dim=-1).to(cache.dtype)
        else:
            cache = cache.view(torch.uint8)  # [num_blocks, block_size, row bytes]
            pack_kv(key[:n], value[:n], cache, md.slot_mapping[:n], self.row,
                    self.k_ranks, self.k_head_offsets, self.k_nouts)

        q = query[:n].view(n, Hq, Hd)
        out = output[:n].view(n, Hq, Hd)
        cos_sin_cache = self.rotary_emb.cos_sin_cache

        nd = md.num_decode_tokens
        if nd:
            in_full_graph = get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
            span = self.max_model_len if in_full_graph else max(md.max_seq_len, 1)
            # Padding rows have length 0; give them one (null-block) position to attend to.
            seq_lens = md.seq_lens[:nd].clamp(min=1)
            step = max(1, _MAX_DECODE_LOGIT_ELEMS // (Hq * span))
            for s in range(0, nd, step):
                e = min(nd, s + step)
                if self.row is None:
                    o = paged_starkv_decode(
                        q[s:e], cache, md.block_table[s:e], seq_lens[s:e], span,
                        self.k_up, self.k_ranks, self.k_head_offsets, self.k_width,
                        self.v_up_by_q, cos_sin_cache, self.scale)
                else:
                    o = paged_quant_decode(
                        q[s:e], cache, md.block_table[s:e], seq_lens[s:e], span, self.row,
                        self.k_up, self.head_of_lane, self.k_tiles, self.v_up_by_q,
                        cos_sin_cache, self.scale)
                out[s:e] = o.to(out.dtype)

        if md.num_prefills:
            self._dense_prefill(md, q[nd:n], key[nd:n], value[nd:n], out[nd:n], cos_sin_cache)
        return output

    def _dense_prefill(self, md, q, k_codes, v_codes, out, cos_sin_cache):
        """STAR-KV's HF prefill: K/V lifted from this step's codes, no cached context."""
        p0, p1 = md.num_decodes, md.num_decodes + md.num_prefills
        qsl_cpu = md.prefill_query_start_loc_cpu
        seq_lens_cpu = md.prefill_seq_lens_cpu or md.seq_lens[p0:p1].tolist()
        query_lens = [qsl_cpu[i + 1] - qsl_cpu[i] for i in range(len(seq_lens_cpu))]
        if seq_lens_cpu != query_lens:
            raise NotImplementedError(
                "STAR-KV prefill needs every prompt in one step with no cached context; "
                "StarKVEngineDefaults should have disabled prefix caching and chunked "
                "prefill and raised max_num_batched_tokens to max_model_len."
            )
        T = q.shape[0]
        H, R, Hd = self.num_kv_heads, self.w.k_max_rank, self.head_size
        qsl = torch.tensor(qsl_cpu, dtype=torch.int32).to(q.device, non_blocking=True)
        lens = torch.tensor(query_lens).to(q.device, non_blocking=True)
        pos = (torch.arange(T, device=q.device)
               - torch.repeat_interleave(qsl[:-1].long(), lens, output_size=T))
        # Gather the ragged per-head segments into a padded [T, H, R] view, then one
        # bmm over KV heads; a broadcasting matmul would expand U^T to
        # [T, H, R, head_dim].
        k_padded = k_codes[:, self.k_prefill_gather].view(T, H, R)
        k = torch.bmm(k_padded.transpose(0, 1), self.k_up_T).transpose(0, 1)
        cos, sin = cos_sin_cache[pos].to(k.dtype).chunk(2, dim=-1)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        k1, k2 = k.chunk(2, dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1).contiguous()
        v = F.linear(v_codes, self.v_up).view(T, H, Hd).contiguous()
        flash_attn_varlen_func(
            q.contiguous(), k, v, max_seqlen_q=max(query_lens), cu_seqlens_q=qsl,
            max_seqlen_k=max(query_lens), cu_seqlens_k=qsl, softmax_scale=self.scale,
            causal=True, out=out, fa_version=self.fa_version,
        )


class StarKVAttentionBackend(AttentionBackend):
    forward_includes_kv_cache_update: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"  # registered as AttentionBackendEnum.CUSTOM by the plugin

    @staticmethod
    def get_impl_cls():
        return StarKVAttentionImpl

    @staticmethod
    def get_builder_cls():
        return StarKVMetadataBuilder


class StarKVAttention(Attention):
    """An ``Attention`` whose K and V inputs are STAR-KV codes, not head-sized tensors."""

    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int,
                 starkv_layer: StarKVLayerWeights, rotary_emb, qkv_proj,
                 cache_config=None, prefix: str = ""):
        super().__init__(num_heads, head_size, scale, num_kv_heads=num_kv_heads,
                         cache_config=cache_config, prefix=prefix,
                         attn_backend=StarKVAttentionBackend,
                         starkv_layer=starkv_layer, rotary_emb=rotary_emb)
        self.starkv_layer = starkv_layer
        # A tuple, so the projection is not registered a second time as a child of
        # this module (weights must load into the attention module's own qkv_proj).
        self._qkv_proj = (qkv_proj,)

    def get_kv_cache_spec(self, vllm_config) -> FullAttentionSpec:
        dtype = self.kv_cache_torch_dtype
        elem = torch.empty((), dtype=dtype).element_size()
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=dtype,
            num_head_slots=1,
            state_content_bytes=self.starkv_layer.cache_bytes(elem),
        )

    def forward(self, query, key, value, output_shape=None, output_dtype=None):
        output = torch.empty((query.shape[0], self.num_heads * self.head_size),
                             dtype=output_dtype or query.dtype, device=query.device)
        if self.use_direct_call:
            unified_attention_with_output(query, key, value, output, self.layer_name)
        else:
            torch.ops.vllm.unified_attention_with_output(
                query, key, value, output, _encode_layer_name(self.layer_name))
        return output

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        qkv = self._qkv_proj[0]
        q_size = self.num_heads * self.head_size
        # K/V rows of the base projection are dead: codes come from kv_down instead.
        if qkv.weight.shape[0] != q_size:
            qkv.weight = torch.nn.Parameter(qkv.weight.data[:q_size].contiguous(),
                                            requires_grad=False)
        self.impl.to_device(qkv.weight.device, act_dtype)
