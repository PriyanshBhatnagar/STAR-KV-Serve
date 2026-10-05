"""Serving-time configuration, read from vLLM's ``--additional-config``.

  vllm serve lmsys/longchat-7b-v1.5-32k \
      --hf-overrides '{"architectures": ["StarKVLlamaForCausalLM"]}' \
      --additional-config '{"starkv": {"weights": "/path/to/fused_weights.pt"}}'

Add ``"kv_quant": true`` to store the compressed layers' cache as 4-bit codes
(``kernels/paged_quant.py``).

Prefix caching and chunked prefill are turned off by the plugin
(``StarKVEngineDefaults``), so they need no flags.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig

from starkv_serve.checkpoint import StarKVCheckpoint, load_fused_checkpoint

logger = init_logger(__name__)


class StarKVEngineDefaults(VerifyAndUpdateConfig):
    """Engine settings STAR-KV's prefill needs, applied by vLLM to this architecture.

    Prefill lifts the step's own codes to full K/V and attends over them; it
    never reads the cache. So every prompt must arrive whole in one step with
    nothing cached before it: prefix caching and chunked prefill are off, and
    the step's token budget is at least ``max_model_len`` so the longest prompt
    fits.
    """

    @staticmethod
    def verify_and_update_config(vllm_config) -> None:
        cache, sched = vllm_config.cache_config, vllm_config.scheduler_config
        max_len = vllm_config.model_config.max_model_len
        cache.enable_prefix_caching = False
        sched.enable_chunked_prefill = False
        if sched.max_num_batched_tokens < max_len:
            sched.max_num_batched_tokens = max_len
            # SchedulerConfig.__post_init__ derives these from the old budget.
            sched.max_num_encoder_input_tokens = sched.encoder_cache_size = max_len
        sched.verify_max_model_len(max_len)
        logger.info("STAR-KV: prefix caching and chunked prefill disabled, "
                    "max_num_batched_tokens=%d", sched.max_num_batched_tokens)


@dataclass(frozen=True)
class StarKVServeConfig:
    weights: str
    kv_quant: bool = False


def get_starkv_config(vllm_config) -> StarKVServeConfig:
    raw = (vllm_config.additional_config or {}).get("starkv") or {}
    if not raw.get("weights"):
        raise ValueError(
            "StarKVLlamaForCausalLM needs the fused checkpoint written by train.py: "
            '--additional-config \'{"starkv": {"weights": "/path/to/fused_weights.pt"}}\''
        )
    return StarKVServeConfig(weights=raw["weights"], kv_quant=bool(raw.get("kv_quant", False)))


@lru_cache(maxsize=4)
def _cached_checkpoint(path: str, num_layers: int, num_kv_heads: int, head_dim: int,
                       dtype: torch.dtype, kv_quant: bool) -> StarKVCheckpoint:
    return load_fused_checkpoint(path, num_layers, num_kv_heads, head_dim, dtype=dtype,
                                 kv_quant=kv_quant)


def starkv_checkpoint_for(vllm_config) -> StarKVCheckpoint:
    """Every decoder layer asks for its own entry; the file is read once."""
    cfg = get_starkv_config(vllm_config)
    hf = vllm_config.model_config.hf_config
    head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
    num_kv_heads = getattr(hf, "num_key_value_heads", None) or hf.num_attention_heads
    return _cached_checkpoint(cfg.weights, hf.num_hidden_layers, num_kv_heads, head_dim,
                              vllm_config.model_config.dtype, cfg.kv_quant)
