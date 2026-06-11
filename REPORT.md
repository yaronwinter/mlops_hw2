# MLOps HW2 — Inference Optimization & SLO Report

**Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` (MoE, ~30B total / ~3B active params)
**Hardware:** 1× H100 80 GB
**SLO:** P95 end-to-end agent latency **< 5 s** at **≥ 10 RPS** (1 RPS = one full
agent run) sustained over a **5-minute** window.

**Workload profile:** prompts 1.5–3K tokens (dominated by the shared DB schema),
short structured outputs (a SQL statement / a small verify JSON), and 2–3
*dependent* vLLM calls per request (`generate_sql → verify → (revise)`). This is
**prefill-heavy, decode-light** — the cost is TTFT, not generation length.

> Status: scaffold. Sections 1 and the rationale are final; numbers (eval +
> latency) get filled in from the H100 run. Detailed per-run capture lives in
> [`results/iteration_log.md`](results/iteration_log.md); this report distills it.

---

## 1. Serving configuration (Phase 1)

Flags chosen *for this workload*, with the one-line rationale for each. Baseline
is BF16 so quantization could be A/B'd as an iteration (§3). **Final chosen
config:** _TBD — the winning config script from §3._

| Flag | Value | Why this lever, for this workload |
|------|-------|-----------------------------------|
| `--quantization fp8` | on (on-the-fly) | ~60 GB BF16 weights → ~30 GB; frees ~30 GB for KV cache and uses the H100's FP8 cores. The single biggest lever on an 80 GB card. |
| `--max-model-len` | 8192 (→ 4096?) | Real need is ~3K prompt + ≤512 output; the model's 262K default over-reserves and shrinks effective batching. |
| `--enable-prefix-caching` | on | The DB schema + system prompt are identical across calls to the same DB, so the 1.5–3K-token prefix is prefilled once and reused — biggest TTFT win after FP8. |
| `--enable-chunked-prefill` | on | Interleaves big 3K-token prefills with in-flight decodes so decode requests don't stall behind a prefill — protects the P95 tail. |
| `--max-num-batched-tokens` | 8192 | Prefill chunk budget; tuned alongside chunked prefill to balance prefill throughput vs. decode latency. |
| `--max-num-seqs` | 48 (→ 96?) | Concurrency dial. ~10 RPS × ~2.5 dependent calls ≈ 25 req/s in flight; sized to admit that without over-batching and spiking TTFT. |
| `--gpu-memory-utilization` | 0.92 | Maximize KV-cache space once weights are FP8, leaving headroom for CUDA-graph capture and activation spikes. |

**Deliberately omitted:** `--tensor-parallel-size` (1 GPU → TP=1); `--enforce-eager`
(kept CUDA graphs ON for lower per-step latency).

**Agent-side latency levers** (request params, not server flags — see `agent/graph.py`):
`max_tokens` caps decode time per call; the `/answer` handler and LLM calls are
`async` so the I/O-bound agent doesn't pin a thread per request; `MAX_ITERATIONS`
bounds the worst-case dependent-call chain that drives P95.

---

## 2. Baseline eval results (Phase 5)

Execution accuracy on the 30-question eval set (`results/eval_baseline.json`),
comparing canonicalized row sets against gold SQL.

- **Overall pass rate:** ▢ / 30 (▢ %)
- **Per-iteration pass rate:** iter 1 ▢ % · iter 2 ▢ % · iter 3 ▢ %
- **Commentary:** _which question types failed, whether failures were generate-
  vs. verify-side, anything the schema rendering made hard._

---

## 3. Hitting the SLO (Phase 6)

### Baseline vs. SLO
- **Baseline (Run 0, BF16):** P50 ▢ / P95 ▢ / achieved RPS ▢. SLO ▢ met / ▢ missed by ▢ s.
- **Bottleneck read (Grafana):** _TTFT vs. decode vs. queue depth vs. KV-cache % — where the time goes._

### Iteration log
Format: **saw X → hypothesized Y → changed Z → result was W.** One line per
iteration (3–4 is normal). Full run table + config map in
[`results/iteration_log.md`](results/iteration_log.md).

1. _saw … → hypothesized … → changed … → result was …_  (screenshot: `screenshots/grafana_before.png` / `grafana_after.png`)
2. _…_
3. _…_

### Final numbers
- **Final config:** ▢ (which `scripts/configs/*.sh`)
- **P50 ▢ / P95 ▢ / P99 ▢ / achieved RPS ▢** over 5 min at target RPS.
- **Quality after tuning:** `results/eval_after_tuning.json` → ▢ % (vs. baseline ▢ %).
- **Verdict:** _SLO hit — or missed, with the gap quantified. Honest read on
  whether a metric improved without the SLO following._

---

## 4. Did the agent loop earn its keep?

_One paragraph. Did `verify → revise` actually improve answers, or just add
latency? Cite the per-iteration pass rate: how many questions that were wrong at
iteration 1 became right after a revise? Weigh that gain against the extra
dependent call it adds to the P95 tail._

---

## 5. What I'd do with more time

_Be specific — workload-grounded, not "add Kubernetes." Candidate directions to
refine or replace:_
- _Shrink the prompt: the schema dominates the 1.5–3K tokens — prune to
  question-relevant tables/columns to cut prefill cost directly._
- _Add a prefix-cache-hit-rate panel to confirm the schema prefix is actually
  being reused, and quantify the TTFT it saves._
- _A smaller/cheaper verify call (tighter `max_tokens` + guided JSON) since that
  reply is tiny — measure its share of the P95 tail._
- _Evaluate speculative decoding — likely low payoff here (short outputs, cheap
  A3B decode), but worth a measured rule-out._

---

## Appendix — deliverables checklist

- [ ] `REPORT.md` complete (≤3 pages)
- [ ] `results/eval_baseline.json`, `results/eval_after_tuning.json`
- [ ] `screenshots/vllm_manual_query.png`
- [ ] `screenshots/grafana_serving.png`, `grafana_eval_run.png`, `grafana_before.png`, `grafana_after.png`
- [ ] `screenshots/langfuse_trace.png`, `langfuse_tags.png`
- [ ] `agent/graph.py`, `agent/prompts.py`, `evals/run_eval.py` implemented
