"""STAR-KV Llama for vLLM: stock Llama with compressed attention in the layers
the checkpoint compressed, and stock attention everywhere else."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import AttentionType

from starkv_serve.attention import StarKVAttention
from starkv_serve.config import get_starkv_config, starkv_checkpoint_for


class StarKVLlamaAttention(nn.Module):
    """Mirrors vLLM's ``LlamaAttention`` for one compressed layer.

    ``qkv_proj`` keeps the stock name and shape so ``q_proj`` loads through the
    stock stacked loader. The checkpoint has no ``k/v_proj`` weights for this
    layer, so the K/V rows are never filled; they are dropped after loading and
    the layer computes the cache codes with the checkpoint's down-projection.
    """

    def __init__(self, config, hidden_size: int, num_heads: int, num_kv_heads: int,
                 max_position_embeddings: int = 8192, quant_config=None, bias: bool = False,
                 bias_o_proj: bool = False, cache_config=None, prefix: str = "",
                 attn_type: str = AttentionType.DECODER):
        super().__init__()
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError("STAR-KV serving supports tensor_parallel_size=1")
        if quant_config is not None:
            raise NotImplementedError("STAR-KV serving does not support weight quantization")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("STAR-KV serving supports decoder attention only")

        vllm_config = get_current_vllm_config()
        layer = starkv_checkpoint_for(vllm_config).layers[extract_layer_index(prefix)]

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = getattr(config, "head_dim", None) or hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size, head_size=self.head_dim,
            total_num_heads=num_heads, total_num_kv_heads=num_kv_heads,
            bias=bias, quant_config=quant_config, prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=num_heads * self.head_dim, output_size=hidden_size,
            bias=bias_o_proj, quant_config=quant_config, prefix=f"{prefix}.o_proj",
        )
        LlamaAttention._init_rotary_emb(self, config, quant_config=quant_config)

        self.k_code_size = layer.k_width
        self.v_code_size = layer.v_rank
        self.attn = StarKVAttention(
            num_heads, self.head_dim, self.scaling, num_kv_heads,
            starkv_layer=layer, rotary_emb=self.rotary_emb, qkv_proj=self.qkv_proj,
            cache_config=cache_config, prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q, _ = self.qkv_proj(hidden_states)  # Q only, once loading is done
        q, _ = self.rotary_emb.forward_native(positions, q, None)
        codes = F.linear(hidden_states, self.attn.impl.kv_down)
        k_codes, v_codes = codes.split([self.k_code_size, self.v_code_size], dim=-1)
        out, _ = self.o_proj(self.attn(q, k_codes, v_codes))
        return out


class StarKVLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, vllm_config, prefix: str = "", config=None):
        compressed = starkv_checkpoint_for(vllm_config).is_compressed(extract_layer_index(prefix))
        super().__init__(vllm_config=vllm_config, prefix=prefix, config=config,
                         attn_layer_type=StarKVLlamaAttention if compressed else LlamaAttention)


# The K/V factors, read by checkpoint.py when the layers are built.
_FACTOR_SUFFIXES = (".VS.weight", ".U.weight", ".head_ranks")


class StarKVLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        # Fail before building anything if the path is missing.
        weights = get_starkv_config(vllm_config).weights
        super().__init__(vllm_config=vllm_config, prefix=prefix,
                         layer_type=StarKVLlamaDecoderLayer)
        self.starkv_weights = weights

    def load_weights(self, weights):
        """Every weight comes from the fused checkpoint, none from the HF model.

        train.py fine-tunes the whole model with the factors, so the HF model's
        own weights are not the trained ones. vLLM's iterator over the HF files
        is never started; the HF model supplies only the config and tokenizer.
        """
        del weights
        sd = torch.load(self.starkv_weights, map_location="cpu", mmap=True, weights_only=True)
        return super().load_weights(
            (name, t) for name, t in sd.items() if not name.endswith(_FACTOR_SUFFIXES))
