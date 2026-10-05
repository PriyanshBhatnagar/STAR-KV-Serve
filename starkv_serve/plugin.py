"""vLLM general plugin entry point. vLLM may call this more than once per process."""


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    from starkv_serve.config import StarKVEngineDefaults

    if "StarKVLlamaForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("StarKVLlamaForCausalLM",
                                     "starkv_serve.model:StarKVLlamaForCausalLM")
    # vLLM applies this when it builds the engine config for the architecture,
    # before the scheduler and KV cache manager read it.
    MODELS_CONFIG_MAP["StarKVLlamaForCausalLM"] = StarKVEngineDefaults
    register_backend(AttentionBackendEnum.CUSTOM, "starkv_serve.attention.StarKVAttentionBackend")
