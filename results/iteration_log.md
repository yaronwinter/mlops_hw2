# Phase 6 — SLO Iteration Log

**SLO:** P95 end-to-end agent latency **< 5 s** at **≥ 10 RPS** (1 RPS = one full
agent run) sustained over a **5-minute** window, on 1× H100 80 GB serving
`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`.

**How each row is produced** — each run is a prepared, single-lever config script:
```bash
RESTART_VLLM=1 VLLM_SCRIPT=scripts/configs/01_fp8.sh scripts/start_stack.sh
uv run python load_test/driver.py --rps 10 --duration 300 --out results/run_01.json
```
(Run 0 uses the baseline `scripts/start_vllm.sh` — omit `VLLM_SCRIPT`.)

**Config scripts** (the prepared experiment menu)

| Run | Script | One lever vs. its base |
|----:|--------|------------------------|
| 0 | `scripts/start_vllm.sh` | BF16 baseline, no quantization |
| 1 | `scripts/configs/01_fp8.sh` | `+ --quantization fp8` (vs baseline) |
| 2 | `scripts/configs/02_fp8_seqs96.sh` | `--max-num-seqs 48→96` (vs Run 1) |
| 3 | `scripts/configs/03_fp8_protect_decode.sh` | `--max-num-batched-tokens 8192→4096` (vs Run 1) |
| 4 | `scripts/configs/04_fp8_kvcache_fp8.sh` | `+ --kv-cache-dtype fp8` (vs Run 1) |
| 5 | `scripts/configs/05_fp8_maxlen_4096.sh` | `--max-model-len 8192→4096` (vs Run 1) |
| 10 | `scripts/configs/10_fp8_throughput_safe.sh` | **stacked:** seqs 96 + maxlen 4096 (quality-safe) |
| 11 | `scripts/configs/11_fp8_max_headroom.sh` | **stacked:** Run 10 + kv-cache fp8 (max headroom) |

Runs 2–5 each branch off Run 1 (the expected working FP8 base), changing one
lever — so pick the next run by what Grafana shows, not strictly in order.
Runs 10–11 are **pre-stacked best-of** configs: run them only after the
single-lever runs, and only stack levers that individually helped. Compare a
stack against the best single-lever run (and, for Run 11, against Run 10 to
isolate the kv-cache delta).
Copy the numbers straight from the driver's JSON `summary` block
(`latency_p50/p95/p99/max`, `achieved_rps`, `ok/timeouts/http_errors/client_errors`).
Run the eval after any change that could affect quality (FP8 KV cache, fewer
iterations, `max_tokens`) and record the pass rate so you can tell a latency win
apart from a quality regression.

> Discipline: **change one lever per run.** If you change two, you can't attribute
> the delta — and attribution is what this phase is graded on.

---

## Results

| Run | Lever changed | P50 (s) | P95 (s) | P99 (s) | Max (s) | Achieved RPS | ok / timeout / http / client | Eval pass % | SLO met? |
|----:|---------------|--------:|--------:|--------:|--------:|-------------:|------------------------------|------------:|:--------:|
| 0 | BF16 baseline (no quant) | | | | | | / / / | | |
| 1 | + FP8 (`--quantization fp8`) | | | | | | / / / | | |
| 2 | FP8 + max-num-seqs 96 | | | | | | / / / | | |
| 3 | FP8 + batched-tokens 4096 | | | | | | / / / | | |
| 4 | FP8 + kv-cache fp8 | | | | | | / / / | | |
| 5 | FP8 + max-model-len 4096 | | | | | | / / / | | |
| 10 | FP8 stacked: seqs96 + maxlen4096 | | | | | | / / / | | |
| 11 | FP8 stacked: + kv-cache fp8 | | | | | | / / / | | |

---

## Per-run narrative

Fill one block per run. The grader wants the reasoning chain, not just the number:
**saw X → hypothesized Y → changed Z → result W**, with before/after evidence that
the *targeted* metric moved — and whether end-to-end P95 (and quality) followed.

### Run 0 — Baseline
- **Config:** _paste the exact `start_vllm.sh` flags + agent `MAX_ITERATIONS`, `max_tokens`._
- **Saw (Grafana):** _TTFT / decode time / KV-cache % / waiting-queue depth at 10 RPS._
- **Result:** P50 ▢ / P95 ▢ / achieved RPS ▢. SLO: ▢ met / ▢ missed by ▢ s.
- **Bottleneck read:** _where in the request lifecycle the time goes (prefill? decode? queue? agent threadpool?)._

### Run 1
- **Saw:** _the metric from the prior run that motivated this change._
- **Hypothesis:** _changing Z should move metric M because…_
- **Changed:** _the single lever + old → new value._
- **Grafana evidence:** _did the targeted metric (e.g. queue depth, TTFT) actually move? before → after._
- **End-to-end result:** P95 ▢ → ▢. Did SLO move with the metric, or not (and why)?
- **Quality:** eval pass % ▢ → ▢ (unchanged / regressed / improved).
- **Decision:** keep / revert / try next.

### Run 2
- **Saw:**
- **Hypothesis:**
- **Changed:**
- **Grafana evidence:**
- **End-to-end result:**
- **Quality:**
- **Decision:**

### Run 3
- **Saw:**
- **Hypothesis:**
- **Changed:**
- **Grafana evidence:**
- **End-to-end result:**
- **Quality:**
- **Decision:**

---

## Lever cheat-sheet (what to reach for, given the symptom)

| Symptom in Grafana | Likely cause | Lever to try |
|--------------------|--------------|--------------|
| Waiting/queued requests pile up; GPU underused | batch too small | raise `--max-num-seqs` |
| High TTFT, low decode time | prefill-bound | confirm prefix-cache hit rate; lower `--max-num-batched-tokens` chunk or `--max-num-seqs` to protect decodes |
| KV cache near 100%, preemptions/swaps | memory-bound | lower `--max-model-len`, or `--kv-cache-dtype fp8` (re-run eval) |
| vLLM metrics fine, but end-to-end P95 high | agent layer | check `MAX_ITERATIONS`, `max_tokens`, async handler / threadpool |
| P95 tail driven by occasional long runs | revise loop firing | lower `MAX_ITERATIONS` (re-run eval to confirm quality holds) |

## Final verdict

_SLO hit, or missed with the gap quantified. The best config and the one-line
reason it wins. The most useful thing you learned from an iteration that did **not**
work — a metric that improved while the SLO didn't is its own lesson._
