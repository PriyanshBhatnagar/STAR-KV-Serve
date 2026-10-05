"""Serving benchmark on one GPU: stock vLLM vs STAR-KV (bf16 and 4-bit cache), over
context lengths, each at concurrency 1 and at the maximum concurrency its KV cache
supports.

Per (system, context):
  1. start `vllm serve` for that system;
  2. read "Maximum concurrency for N tokens per request" from the server log. A boot that
     cannot fit one session is retried one --gpu-memory-utilization step higher, an OOM one
     step lower (same rule for both systems), then recorded as refused;
  3. run `vllm bench serve` with random prompts of (context - output_len) tokens:
     a discarded warm-up request, then concurrency 1 (2 requests), then one wave at max
     concurrency. Each run uses its own seed so no prompt hits the prefix cache;
  4. profile one more request at concurrency 1 and record the attention time per decode
     step (see attention_us_per_step);
  5. stop the server and append a record to <out>/summary.jsonl.

Existing (system, context) records are skipped, so an interrupted run resumes.
Contexts above the model's RoPE positions override linear rope scaling for both systems.

At the end it prints, per context, each system's max sessions, TPOT and attention time at
concurrency 1 and total throughput at max concurrency, with STAR-KV's decode speedup,
attention speedup and throughput gain against vLLM. Every metric is in <out>/summary.jsonl.

Attention time is measured only for systems that run eager (vLLM with
--match-starkv-config, STAR-KV without --starkv-cudagraphs): inside a CUDA graph the
attention op is not visible to the profiler.

  python bench_serving.py --weights fused_weights.pt --out results/serving \
      --contexts 4096 8192 16384 32768 --gpu 0
"""

import argparse
import bisect
import gzip
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

METRICS = ["completed", "duration", "request_throughput", "output_throughput", "total_token_throughput",
           "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
           "p99_tpot_ms", "mean_itl_ms", "p99_itl_ms", "mean_e2el_ms", "median_e2el_ms", "p99_e2el_ms"]


def eager(args, system: str) -> bool:
    if system == "vllm":
        return args.match_starkv_config
    return not args.starkv_cudagraphs


def trace_dir(args, system: str, ctx: int, util: float) -> Path:
    return Path(args.out).resolve() / "traces" / f"{system}_{ctx}_u{util}"


# Everything a decode step spends on attention: writing the new token's K/V into the
# paged cache (stock vLLM does it in its own op, STAR-KV inside attention) and the
# attention itself, which reads the cache. The projections are outside both ops.
ATTN_OPS = {"vllm::unified_attention_with_output", "vllm::unified_kv_cache_update"}
# vLLM labels every engine step "execute_context_<n>(<tokens>)_generation_<n>(<tokens>)";
# a decode step schedules no context (prefill) tokens.
STEP_SCOPE = re.compile(r"execute_context_(\d+)\(\d+\)_generation_(\d+)\(\d+\)")


def attention_us_per_step(trace: Path) -> dict | None:
    """GPU time of the attention ops per decode step, from one worker trace.

    Each attention op's time is the summed duration of every kernel and copy it
    launched -- compute and KV-cache traffic, without the CPU launch gaps between
    them. Ops are grouped into engine steps by vLLM's step label, and only pure decode
    steps are kept. Returns the median over those steps.
    """
    with gzip.open(trace, "rt") if trace.suffix == ".gz" else open(trace) as f:
        events = json.load(f).get("traceEvents", [])
    ops, steps, launches, device = defaultdict(list), [], [], defaultdict(float)
    for e in events:
        cat, args_ = e.get("cat"), e.get("args") or {}
        if cat == "cpu_op" and e.get("name") in ATTN_OPS:
            ops[e["tid"]].append([e["ts"], e["ts"] + e["dur"], 0.0])
        elif cat == "user_annotation" and (m := STEP_SCOPE.fullmatch(e.get("name", ""))):
            if m.group(1) == "0" and m.group(2) != "0":
                steps.append((e["ts"], e["ts"] + e["dur"]))
        elif cat in ("cuda_runtime", "cuda_driver") and "correlation" in args_:
            launches.append((e["ts"], e["tid"], args_["correlation"]))
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset") and "correlation" in args_:
            device[args_["correlation"]] += e["dur"]
    if not ops or not steps:
        return None
    for tid in ops:
        ops[tid].sort()
    starts = {tid: [o[0] for o in v] for tid, v in ops.items()}
    for ts, tid, corr in launches:
        if tid in ops and corr in device:
            i = bisect.bisect_right(starts[tid], ts) - 1
            if i >= 0 and ts <= ops[tid][i][1]:
                ops[tid][i][2] += device[corr]
    steps.sort()
    step_starts = [s for s, _ in steps]
    per_step = defaultdict(float)
    for v in ops.values():
        for start, end, us in v:
            i = bisect.bisect_right(step_starts, start) - 1
            if i >= 0 and end <= steps[i][1]:
                per_step[i] += us
    decode = list(per_step.values())
    if not decode:
        return None
    decode.sort()
    return {"attention_us": decode[len(decode) // 2], "decode_steps": len(decode)}


def read_attention(directory: Path, timeout_s: int = 600) -> dict | None:
    """Wait for the worker to write its trace after /stop_profile, then parse it."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for trace in sorted(directory.glob("*.json*")):
            try:
                result = attention_us_per_step(trace)
            except (OSError, EOFError, json.JSONDecodeError):
                continue  # still being written
            if result:
                return result
        time.sleep(5)
    return None


def server_command(args, system: str, ctx: int, util: float) -> tuple[list[str], dict]:
    cmd = [args.vllm, "serve", args.model, "--served-model-name", "model", "--dtype", "float16",
           "--max-model-len", str(ctx), "--gpu-memory-utilization", str(util),
           "--port", str(args.port), "--seed", "0"]
    env = {"CUDA_VISIBLE_DEVICES": args.gpu, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    if eager(args, system):
        # Idle until a client sends /start_profile.
        cmd += ["--profiler-config", json.dumps({
            "profiler": "torch", "torch_profiler_dir": str(trace_dir(args, system, ctx, util)),
            "torch_profiler_with_stack": False, "torch_profiler_dump_cuda_time_total": False})]
    hf = {}
    if ctx > args.rope_positions:
        factor = float(math.ceil(ctx * args.rope_factor / args.rope_positions))
        hf["rope_parameters"] = {"rope_type": "linear", "factor": factor, "rope_theta": 10000.0}

    if system == "vllm":
        if args.match_starkv_config:
            # Same engine features STAR-KV runs under, so the comparison measures
            # the KV cache and not vLLM's scheduler: STAR-KV's prefill needs each
            # prompt in one step with no cached context, and it runs eager.
            cmd += ["--enforce-eager", "--no-enable-prefix-caching",
                    "--no-enable-chunked-prefill", "--max-num-batched-tokens", str(ctx)]
        else:
            cmd += ["--compilation-config", json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE"}),
                    "--enable-prefix-caching", "--enable-chunked-prefill",
                    "--max-num-batched-tokens", "8192"]
        env["VLLM_PLUGINS"] = ""
    elif system in ("starkv", "starkv_4bit"):
        hf["architectures"] = ["StarKVLlamaForCausalLM"]
        starkv = {"weights": args.weights, "kv_quant": system == "starkv_4bit"}
        # The plugin turns prefix caching and chunked prefill off itself; the token
        # budget is pinned to the context as for the vLLM baseline.
        cmd += ["--additional-config", json.dumps({"starkv": starkv}),
                "--max-num-batched-tokens", str(ctx)]
        cmd += [] if args.starkv_cudagraphs else ["--enforce-eager"]
        env["VLLM_PLUGINS"] = "starkv_serve"
    else:
        raise ValueError(system)
    if hf:
        cmd += ["--hf-overrides", json.dumps(hf)]
    return cmd, env


def wait_ready(proc, port, timeout_s=2400) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def parse_server_log(text: str) -> dict:
    info = {}
    for key, pattern, cast in (
        ("max_concurrency_reported", r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", float),
        ("kv_cache_tokens", r"GPU KV cache size: ([\d,]+) tokens", lambda x: int(x.replace(",", ""))),
        ("kv_cache_gib", r"Available KV cache memory: ([\d.]+) GiB", float),
        ("model_memory_gib", r"Model loading took ([\d.]+) GiB", float),
    ):
        m = re.findall(pattern, text)
        if m:
            info[key] = cast(m[-1])
    errors = [l for l in text.splitlines()
              if re.search(r"ValueError|RuntimeError|OutOfMemory|NotImplementedError|AssertionError", l)]
    if errors:
        info["errors"] = errors[-3:]
    if re.search(r"OutOfMemoryError|CUDA out of memory", text):
        info["failure"] = "oom"
    elif re.search(r"KV cache is needed|No available memory for the cache blocks", text):
        info["failure"] = "kv"
    return info


def stop(proc, gpu):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    for _ in range(60):  # let the GPU drain before the next boot
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                               "-i", gpu], capture_output=True, text=True).stdout.strip()
        if used.isdigit() and int(used) < 1000:
            return
        time.sleep(5)


def client(args, out_dir: Path, name: str, ctx: int, concurrency: int, num_prompts: int, seed: int,
           profile: bool = False) -> dict:
    fname = f"{name}.json"
    cmd = [args.vllm, "bench", "serve", "--backend", "openai",
           "--base-url", f"http://127.0.0.1:{args.port}", "--endpoint", "/v1/completions",
           "--model", "model", "--served-model-name", "model", "--tokenizer", args.model,
           "--dataset-name", "random", "--random-input-len", str(ctx - args.output_len),
           "--random-output-len", str(args.output_len), "--random-range-ratio", "0.0",
           "--ignore-eos", "--temperature", "0", "--num-prompts", str(num_prompts),
           "--max-concurrency", str(concurrency), "--request-rate", "inf", "--num-warmups", "0",
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99",
           "--seed", str(seed), "--disable-tqdm", *(["--profile"] if profile else []),
           "--save-result", "--result-dir", str(out_dir), "--result-filename", fname]
    env = {**os.environ, "VLLM_PLUGINS": "", "CUDA_VISIBLE_DEVICES": ""}
    with open(out_dir / f"{name}.client.log", "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=4 * 3600).returncode
    path = out_dir / fname
    if rc != 0 or not path.exists():
        return {"client_returncode": rc}
    result = json.loads(path.read_text())
    return {k: result[k] for k in METRICS if k in result}


def run_one(args, out_dir: Path, system: str, ctx: int) -> dict:
    record = {"system": system, "context": ctx, "attempts": []}
    util, direction = args.start_util, 0
    while 0.5 <= util <= 0.96 + 1e-9:
        util = round(util, 2)
        tag = f"{system}_{ctx}"
        cmd, env = server_command(args, system, ctx, util)
        log_path = out_dir / f"{tag}_u{util}.server.log"
        t0 = time.time()
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(out_dir),
                                    env={**os.environ, **env}, start_new_session=True)
        ready = wait_ready(proc, args.port)
        info = parse_server_log(log_path.read_text(errors="replace"))
        attempt = {"gpu_memory_utilization": util, "boot_s": round(time.time() - t0, 1), **info}
        record["attempts"].append(attempt)
        if not ready:
            stop(proc, args.gpu)
            step = {"oom": -0.05, "kv": 0.05}.get(info.get("failure"), 0)
            if step == 0 or direction * step < 0:
                break
            direction, util = step, util + step
            continue
        try:
            record.update({k: v for k, v in attempt.items() if k not in ("errors", "failure")})
            record["status"] = "ok"
            record["max_concurrency"] = n_max = int(math.floor(info.get("max_concurrency_reported", 0)))
            client(args, out_dir, f"{tag}_warmup", ctx, 1, 1, seed=1000)
            record["c1"] = client(args, out_dir, f"{tag}_c1", ctx, 1, 2, seed=1)
            if n_max >= 1:
                record["cmax"] = client(args, out_dir, f"{tag}_cmax", ctx, n_max, n_max, seed=2)
            if eager(args, system):
                # After the timed runs, so profiling never slows a reported TPOT.
                if client(args, out_dir, f"{tag}_profile", ctx, 1, 1, seed=3, profile=True).get("completed"):
                    record["attention"] = read_attention(trace_dir(args, system, ctx, util))
        finally:
            stop(proc, args.gpu)
        return record
    last = record["attempts"][-1] if record["attempts"] else {}
    record["status"] = {"kv": "refused_kv", "oom": "failed_oom"}.get(last.get("failure"), "failed")
    record["max_concurrency"] = 0
    return record


LABELS = {"vllm": "vLLM", "starkv": "STAR-KV", "starkv_4bit": "3.2-bit"}


def print_summary(summary: Path):
    """One row per context: sessions, TPOT and attention time per decode step at
    concurrency 1 and total tok/s at max concurrency per system, and STAR-KV's decode
    speedup (vLLM TPOT / its TPOT), attention speedup (vLLM attention / its attention)
    and throughput gain (its tok/s / vLLM's), each against vLLM."""
    records = {}
    for line in summary.read_text().splitlines():
        r = json.loads(line)
        records[(r["system"], r["context"])] = r
    systems = [s for s in LABELS if any(k[0] == s for k in records)]
    others = [s for s in systems if s != "vllm"]

    def metric(system, ctx, run, key):
        # A run that completed nothing (e.g. it ran out of memory mid-wave) is missing,
        # not zero.
        m = (records.get((system, ctx)) or {}).get(run) or {}
        return m.get(key) if m.get("completed") else None

    def ratio(num, den):
        return f"{num / den:.2f}x" if num and den else "–"

    def num(v, fmt):
        return format(v, fmt) if v is not None else "–"

    def sessions(system, ctx):
        r = records.get((system, ctx))
        if r is None:
            return "–"
        if r["status"] != "ok":
            return {"refused_kv": "refused", "failed_oom": "oom"}.get(r["status"], "failed")
        return str(r["max_concurrency"])

    names = lambda group: "/".join(LABELS[s] for s in group)
    header = ["context", f"sessions {names(systems)}", f"TPOT c1 ms {names(systems)}",
              f"decode speedup {names(others)}", f"attn us/step {names(systems)}",
              f"attn speedup {names(others)}", f"tok/s @max {names(systems)}",
              f"throughput gain {names(others)}"]
    rows = []
    for ctx in sorted({c for _, c in records}):
        tpot = {s: metric(s, ctx, "c1", "mean_tpot_ms") for s in systems}
        tput = {s: metric(s, ctx, "cmax", "total_token_throughput") for s in systems}
        attn = {s: ((records.get((s, ctx)) or {}).get("attention") or {}).get("attention_us")
                for s in systems}
        rows.append([
            f"{ctx // 1024}k" if ctx % 1024 == 0 else str(ctx),
            " / ".join(sessions(s, ctx) for s in systems),
            " / ".join(num(tpot[s], ".1f") for s in systems),
            " / ".join(ratio(tpot.get("vllm"), tpot[s]) for s in others),
            " / ".join(num(attn[s], ",.0f") for s in systems),
            " / ".join(ratio(attn.get("vllm"), attn[s]) for s in others),
            " / ".join(num(tput[s], ",.0f") for s in systems),
            " / ".join(ratio(tput[s], tput.get("vllm")) for s in others),
        ])
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    print()
    for r in [header] + rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip())
    print("\ndecode speedup: vLLM TPOT / STAR-KV TPOT at concurrency 1. attn: GPU time of the "
          "attention ops (KV-cache write + attention over the cache, all layers) per decode step "
          "at concurrency 1. throughput gain: total tok/s at each system's max concurrency, "
          f"against vLLM. All metrics: {summary}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="fused_weights.pt written by train.py")
    p.add_argument("--model", default="lmsys/longchat-7b-v1.5-32k")
    p.add_argument("--out", required=True, help="results directory (summary.jsonl, logs, raw JSON)")
    p.add_argument("--systems", nargs="+", default=["vllm", "starkv", "starkv_4bit"],
                   choices=["vllm", "starkv", "starkv_4bit"])
    p.add_argument("--contexts", nargs="+", type=int, default=[4096, 8192, 16384, 32768])
    p.add_argument("--output-len", type=int, default=32, help="decode tokens per request")
    p.add_argument("--gpu", default="0", help="physical GPU index for every server")
    p.add_argument("--start-util", type=float, default=0.90, help="initial --gpu-memory-utilization")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--starkv-cudagraphs", action="store_true", help="run STAR-KV without --enforce-eager")
    p.add_argument("--match-starkv-config", action="store_true",
                   help="Run the vLLM baseline eager, with prefix caching and chunked "
                        "prefill off, as STAR-KV must. Isolates the KV cache from vLLM's "
                        "scheduler features; without it the baseline gets CUDA graphs and "
                        "STAR-KV does not.")
    p.add_argument("--rope-positions", type=int, default=32768,
                   help="positions covered by the model's own RoPE scaling")
    p.add_argument("--rope-factor", type=float, default=8.0, help="the model's own linear RoPE factor")
    p.add_argument("--vllm", default=str(Path(sys.executable).parent / "vllm"), help="vllm executable")
    args = p.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.weights = str(Path(args.weights).resolve())
    summary = out_dir / "summary.jsonl"
    done = {(r["system"], r["context"]) for r in map(json.loads, summary.read_text().splitlines())} \
        if summary.exists() else set()

    for ctx in args.contexts:
        for system in args.systems:
            if (system, ctx) in done:
                continue
            print(f"[{time.strftime('%H:%M:%S')}] {system} context={ctx}", flush=True)
            rec = run_one(args, out_dir, system, ctx)
            with open(summary, "a") as f:
                f.write(json.dumps(rec) + "\n")
            attn = (rec.get("attention") or {}).get("attention_us")
            print(f"    -> {rec['status']} max_concurrency={rec['max_concurrency']} "
                  f"kv_tokens={rec.get('kv_cache_tokens')} util={rec.get('gpu_memory_utilization')}"
                  + (f" attention={attn:.0f}us/step" if attn else ""), flush=True)

    if summary.exists():
        print_summary(summary)


if __name__ == "__main__":
    main()
