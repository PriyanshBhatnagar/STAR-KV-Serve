# [ICML 2026] STAR-KV-Serve: Serving STAR-KV with vLLM

vLLM integration of the **ICML 2026 Spotlight** paper:

**STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control**

[Paper](https://arxiv.org/abs/2606.08382) | [Project Page](https://icml.cc/virtual/2026/poster/61958) | [OpenReview](https://openreview.net/forum?id=lJjH1q6RwY&noteId=0YBoTCnGTc) | [STAR-KV code](https://github.com/PriyanshBhatnagar/STAR-KV)

## Authors

Priyansh Bhatnagar<sup>1\*</sup>, Ashkan Moradifirouzabadi<sup>1\*</sup>, Se-Hyun Yang<sup>2</sup>, SeungJae Lee<sup>2</sup>, Jungwook Choi<sup>3</sup>, Mingu Kang<sup>1</sup>

<sup>1</sup>University of California San Diego &nbsp; <sup>2</sup>Dnotitia &nbsp; <sup>3</sup>Hanyang University

<sup>\*</sup>Equal contribution

## Overview

STAR-KV compresses the KV cache by caching low-rank intermediate activations, with a learnable soft-threshold that selects the rank of every attention head and layer. This repository brings that cache to [vLLM](https://github.com/vllm-project/vllm), and covers the full pipeline:

- **Training** — STAR-KV's training code: produces the checkpoint (`fused_weights.pt`).
- **Serving** — a vLLM plugin that loads `fused_weights.pt` directly, with no conversion step, and serves the KV cache as bf16 low-rank codes or quantized to 3.2 bits.
- **Benchmarking** — serving against stock vLLM: evaluates concurrent sessions, decode latency, attention time and throughput.

## Repository Structure

```
├── train.py                       # Training (soft-threshold mechanism), as in STAR-KV
├── model.py                       # Decomposed modules, fusion and checkpoint loading, as in STAR-KV
├── soft_thres_layer.py            # Learnable soft-threshold function, as in STAR-KV
├── LlamaLoRaAttention_headwise.py # Low-rank attention module, as in STAR-KV
├── kv_quant.py                    # 3.2-bit KV quantization format, as in STAR-KV
├── check_compression.py           # Report the KV compression encoded in a checkpoint
├── bench_serving.py               # Serving benchmark: stock vLLM vs STAR-KV vs STAR-KV 3.2-bit
└── starkv_serve/                  # vLLM plugin
    ├── kernels/paged_decode.py    #   low-rank paged decode attention (Triton)
    ├── kernels/paged_quant.py     #   paged decode attention over the 3.2-bit cache (Triton)
    ├── attention.py               #   attention backend, metadata and KV-cache spec
    ├── model.py                   #   StarKVLlamaForCausalLM
    ├── checkpoint.py              #   reads fused_weights.pt
    └── config.py, plugin.py       #   configuration and vLLM plugin registration
```

## Installation

1. Clone the repository

```
git clone https://github.com/PriyanshBhatnagar/STAR-KV-Serve.git
cd STAR-KV-Serve
```

2. Create the training environment, used by `train.py` and `check_compression.py`

```
conda create -n StarKV python=3.12.7
conda activate StarKV
pip install -r requirements.txt
```

3. Create the serving environment, used by the vLLM plugin and `bench_serving.py`. It targets **vLLM 0.29.0**; on drivers that support CUDA 12.x but not 13 (e.g. driver 550) the default PyPI wheels do not run, so use the CUDA 12.9 builds:

```
conda create -n StarKV-Serve python=3.12
conda activate StarKV-Serve
pip install torch==2.13.0+cu129 --index-url https://download.pytorch.org/whl/cu129
pip install "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
pip install --no-deps -e .
```

With CUDA 12.4, FlashInfer's JIT fails (`nvcc fatal: Unknown option '--compress-mode=size'`). Set `VLLM_USE_FLASHINFER_SAMPLER=0` for anything that samples in-process; `vllm serve` is unaffected.

## Usage

### Training

```
export HF_TOKEN="your_huggingface_token"   # for gated models (e.g. Llama-3)
export WANDB_API_KEY="your_wandb_key"      # optional

python train.py \
--model lmsys/longchat-7b-v1.5-32k \
--output fused_weights.pt \
--epochs 1 --lr 2e-5 --seq-len 8192 --num-samples 5000 \
--alpha-lr 1e-2 --alpha-samples 3000 --comp-weight-k 0.1 --comp-weight-v 0.1 --kd-weight 1.0 \
--comp-ratio 0.75 --skip-layers 0 1 2 31 \
--rank-multiple-k 16 --rank-multiple-v 32
```

Trains with knowledge distillation from the uncompressed teacher, in two phases over `--num-samples` blocks:

1. **Phase 1:** a learnable soft-threshold truncates the singular values of the K and V projections until the compression budget is met (or `--alpha-samples` blocks have passed).
2. **Phase 2:** each head's K rank is rounded up to a multiple of `--rank-multiple-k` and the V rank to a multiple of `--rank-multiple-v`, Sigma is fused into V, and the remaining blocks recover the fused model with KD alone.

The fused model is written to `--output` — this checkpoint is the only artifact, and is what vLLM serves.

To check ranks and compression across layers:

```
python check_compression.py --weights fused_weights.pt --per-layer --compressed-only
```

###
### Serving Benchmark

```
python bench_serving.py \
  --weights fused_weights.pt \
  --out results/serving \
  --contexts 4096 8192 16384 32768 \
  --gpu 0 --match-starkv-config
```

For each context and each system (stock vLLM, STAR-KV, STAR-KV 3.2-bit), it starts a server, reads the maximum number of concurrent sessions its KV cache holds, and runs `vllm bench serve` at concurrency 1 and at that maximum. Each session is a random prompt of `context − output_len` tokens followed by `--output-len` (default 32) decode tokens. At the end it prints, per context, each system's sessions, TPOT and attention time at concurrency 1 and total throughput at maximum concurrency, together with STAR-KV's decode speedup, attention speedup and throughput gain over vLLM. All metrics (TTFT, TPOT, ITL and E2E latency at mean / median / p99, throughput and attention time) are written to `<out>/summary.jsonl`, and an interrupted run resumes where it stopped.

**Attention time** is the GPU time per decode step spent in vLLM's attention operations over all layers: writing the new token's K/V into the paged cache and attending over the cache. It therefore includes both the computation and the KV-cache traffic — where compression pays off — but not the Q/K/V and output projections. It is measured with the PyTorch profiler on one additional request at concurrency 1, after the timed runs, as the median over its decode steps. Since attention inside a CUDA graph is invisible to the profiler, it is measured for systems that run eager; `--match-starkv-config` runs the vLLM baseline eager as well, with the same engine settings as STAR-KV.

Run benchmarks on an idle GPU: another process holding memory either prevents the server from starting at the requested `--gpu-memory-utilization` or makes it run out of memory mid-run.

## Implementation

**Low-rank paged KV cache.** Each compressed layer stores, per token, its K codes followed by one V code shared by all heads, instead of full-width K and V. Every KV head is stored at its own width — its rank rounded up to the decode kernel's tile (16) — with the heads laid side by side and `k_head_offsets` recording where each begins, so a layer holds the sum of its heads' ranks. The layer declares its row to vLLM as a `FullAttentionSpec` with one head slot and `cache_width × dtype_size` bytes per token, so compressed and uncompressed layers share one KV-cache group at their real page sizes. Layers the checkpoint leaves uncompressed use vLLM's stock attention.

**Low-rank paged decode attention.** `paged_starkv_decode` gathers codes through the block table and reconstructs K per KV head from that head's own rank, applying RoPE from vLLM's `cos_sin_cache` so that K is rotated exactly as Q. The attention probabilities are applied to the shared V codes in low-rank space (`P[H_q, L] @ V'[L, r_v]`) and lifted to head_dim only at the end. The value pass splits each request's sequence into segments that run as parallel GPU programs and are summed; since the probabilities are already normalized over the whole sequence, the sum is exact.

**Prefill** is dense, as in STAR-KV: the step's codes are lifted to full K and V and attended with FlashAttention.

**3.2-bit cache.** STAR-KV's `kv_quant.py` format on vLLM's pages. Each KV head's K latent and the shared V latent are quantized per token: the leading 20% of channels (the largest singular directions) at 4 bits and the rest at 3 bits, each group with its own scale, after a block Hadamard rotation that the loader folds into the low-rank factors. A token's page slot is one row, `[K codes | K scales | V scales | V codes]`, with the codes packed as 4-bit nibbles. New tokens are quantized directly into their slots. Decode folds the query into the K up-projection once per step, so the K logits are computed from the quantized codes, and applies the probabilities to V split over the token axis, with each token's scale folded into the probabilities.

**Weights.** Every weight loads from `fused_weights.pt`: `train.py` fine-tunes the whole model along with the factors, so the HF model supplies only the config and tokenizer. Compressed layers have no K/V projection; their cache codes come from the fused down-projection.

## Results

LongChat-7B on one RTX 4090 (24 GB), with a checkpoint at ~75% KV compression over 28 compressed layers. `bench_serving.py --match-starkv-config` at `--gpu-memory-utilization 0.90`: all three systems run eager, with prefix caching and chunked prefill disabled, so the comparison isolates the KV cache.

| Context | Sessions (vLLM / STAR-KV / 3.2-bit) | Attention per step, µs (vLLM / STAR-KV / 3.2-bit) | Attention speedup (STAR-KV / 3.2-bit) | Decode speedup (STAR-KV / 3.2-bit) | Throughput gain (STAR-KV / 3.2-bit) |
|---|---|---|---|---|---|
| 1K  | 16 / 40 / 76      | 1,095 / 1,600 / 1,973  | 0.68x / 0.55x | 0.77x / 0.82x | 1.11x / 1.33x |
| 2K  | 8 / 20 / 37       | 2,020 / 2,060 / 2,337  | 0.98x / 0.86x | 0.87x / 0.88x | 1.24x / 1.36x |
| 4K  | 4 / 9 / 18        | 3,898 / 2,797 / 2,782  | **1.39x / 1.40x** | 0.97x / 0.94x | 1.26x / 1.40x |
| 8K  | 1 / 4 / 8         | 7,811 / 4,564 / 3,561  | **1.71x / 2.19x** | **1.14x / 1.20x** | **1.54x / 1.71x** |
| 16K | 1 / 2 / 3         | 15,424 / 8,504 / 5,285 | **1.81x / 2.92x** | **1.34x / 1.59x** | **1.29x / 1.40x** |
| 32K | refused / 1 / 1   | – / – / 9,015          | – | – | – |

Speedups and gains are against stock vLLM: attention and decode (TPOT) speedup at concurrency 1, throughput gain in total tokens per second at each system's maximum concurrency.

- **Capacity.** STAR-KV holds 2–2.5× the concurrent sessions of vLLM, and the 3.2-bit cache 3–8×.
- **Attention.** vLLM's attention time doubles with the context, while STAR-KV's grows more slowly: below ~2K reconstructing K from the codes costs more than the smaller cache saves, and from 4K on STAR-KV is faster, reaching 1.81× (bf16) and 2.92× (3.2-bit) at 16K.
- **Decode.** The decode speedup is smaller than the attention speedup because attention is only part of each step — about 7% of vLLM's step at 1K and half of it at 16K.
- **32K.** vLLM cannot fit a single 32K session. The 3.2-bit cache serves one (22.3 ms TPOT). STAR-KV bf16 fits one session in its KV cache, but its dense prefill of a 32K prompt then exceeds the remaining memory on a 24 GB card.

## Reference

If you find this work useful, please consider citing our paper:

```
@misc{starkv2026,
      title={STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding for Adaptive Rank Control}, 
      author={Priyansh Bhatnagar and Ashkan Moradifirouzabadi and Se-Hyun Yang and SeungJae Lee and Jungwook Choi and Mingu Kang},
      year={2026},
      eprint={2606.08382},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.08382}, 
}
```
