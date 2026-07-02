# Coding-agent traces — cross-dataset analysis

Six public coding-agent trace datasets downloaded to `~/workspace/agentic_workloads/agentic_traces/` (each dataset dir now also holds its own report `.md`)
and analyzed. One detailed report per dataset in this folder. Analysis engine:
`~/workspace/agentic_workloads/analysis_engine.py`.

**What was measured per dataset:** conversation shape (steps/session), per-call **ISL/OSL** token
distributions, **prefix-cache modeling** (cISL/nISL split → hit ratio), **prefix-cache size** (peak
context → vLLM 16-tok blocks → KV-cache bytes on the real Qwen2.5 config), and **tool/action mix**.

Tokens are **measured** for AxT (provider-reported `token_usage_*`); **tiktoken `o200k_base`** estimates
elsewhere. Heavy datasets are sampled (noted in each report).

## Cross-dataset comparison

| dataset (→ report) | what it is / harness (arch) | sessions | steps/sess p50 | ISL p50 | OSL p50 | peak-ctx p50 | KV size p50 | **prefix-cache hit** |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| [`axt-swe-agent-lm-32b`](axt-swe-agent-lm-32b/axt-swe-agent-lm-32b.md) | SWE-agent-LM-32B on R2E-Gym, **append-only ReAct**; 4 ctx budgets ×2 (DeepSWE lineage); measured tokens | 4,000* | 23 | 13.7k | 68 | 17.1k | 4.3 GiB | **97.0%** |
| [`nebius-swe-agent`](nebius-swe-agent/nebius-swe-agent.md) | Nebius **SWE-agent** training corpus (~80K); classic ACI commands (`edit`/`open`/`search_dir`) | 2,500* | 17 | 8.4k | 74 | 10.0k | 0.5 GiB† | **96.0%** |
| [`swegym-openhands`](swegym-openhands/swegym-openhands.md) | **OpenHands (CodeAct)** on SWE-Gym; native tool-calling + `tool` role; can condense history | 2,500* | 14 | 11.9k | 41 | 10.7k | 2.6 GiB | **95.5%** |
| [`r2egym-sft`](r2egym-sft/r2egym-sft.md) | **R2E-Gym SFT** set (COLM'25); curated ReAct `<function=>` trajectories for fine-tuning | 3,231 | 14 | 8.1k | 63 | 11.6k | 2.8 GiB | **92.9%** |
| [`swegym-moatless`](swegym-moatless/swegym-moatless.md) | **Moatless Tools** on SWE-Gym; **state machine** (Search→Identify→Plan→Edit), rebuilds prompt per phase | 600* | 7 | 3.0k | 198 | 6.9k | 1.7 GiB | **36.4%**‡ |
| [`r2egym-verifier`](r2egym-verifier/r2egym-verifier.md) | R2E-Gym **verifier** RM data; **one-shot judge** over a whole trajectory (not an agent) | 3,000* | 1 | 14.0k | 9 | 14.0k | 3.5 GiB | 0% (single call) |
| [`sharegpt-v3`](sharegpt-v3/sharegpt-v3.md) | **ShareGPT chat baseline** — human↔ChatGPT, no tools/env (Frontier's SharedGPT source) | 3,983* | 3 | 634 | 259 | 1.7k | 0.2 GiB§ | **90.1%** |

\* sampled · † 7B model (56 KiB/tok) vs 32B (256 KiB/tok) elsewhere · ‡ non-append-only → approximate (see report) · § Llama-3.1-8B (128 KiB/tok)

**Chat vs agents:** ShareGPT is **decode-balanced** (ISL:OSL ≈ 2.4:1) and **short** (3 steps, ~1.7k peak context);
the coding agents are **prefill-heavy** (50–200:1) and **long** (14–23 steps, 10–17k peak). Same prefix-cache
mechanism, opposite request shape — which is exactly why agentic serving stresses prefill/KV far harder than chat.

## Key findings

1. **Append-only agents are extremely prefix-cacheable: 93–97% hit.** For ReAct/CodeAct agents (AxT,
   nebius, OpenHands, R2E-SFT), each turn appends a tool result to a growing history, so the vast
   majority of every prompt is a cache hit. Measured per session, **cISL dominates** — only ~3–7% of input
   tokens (nISL) ever need real prefill.

2. **Agent architecture decides cache behavior.** Moatless — a *state-machine* agent that rebuilds its
   prompt per action (Search → Identify → Plan → Edit) rather than appending — is **prefix-cache-hostile**
   at ~36% hit, with 95% of sessions shrinking context between calls. Same task, ~2.7× more prefill cost
   per cached token. Architecture, not just task, drives GPU cost.

3. **ISL ≫ OSL — the agentic signature.** Input is **50–200× larger than output at the median** (e.g.
   AxT 13.7k ISL vs 68 OSL). Agentic serving is **prefill-heavy / decode-light per call** — the opposite
   of ShareGPT chat (where OSL > ISL). This is exactly why prefix caching matters so much here.

4. **Long sessions blow up the KV footprint.** Peak context p99 reaches 60–100k tokens = **16–25 GiB of
   KV cache** on a 32B model, vs ~4 GiB at the median. A small tail of long sessions dominates memory:
   a few very long sessions account for a disproportionate share of inference requests and peak KV demand.

5. **Context budget scaling (AxT, 16K→131K).** Larger budgets → more steps (18→41), higher peak context
   (14.8k→29.9k), and higher cache-hit (91.5%→94.5%). Concretely shows how **short benchmark sessions
   underestimate cISL**.

6. **Tool mix is edit-dominated.** File editing (`file_editor`/`str_replace_editor`) is 57–66% of calls,
   then shell execution (~22–39%), then search. Consistent across harnesses.

## How this connects to the earlier context

- **vs. Frontier (arXiv 2605.21312):** Frontier *simulates* vLLM's block-hash prefix cache and reports a
  36.98% hit ratio on its SharedGPT replay. Our measured **append-only** agent hit ratios (93–97%) are far
  higher because agentic histories are near-perfectly append-only, whereas chat replay reuses less. Moatless
  (36%) lands right in Frontier's regime — because it isn't append-only.
- **On CPU-time / cache modeling:** these traces give real ISL/OSL and let us model **cISL (cache hit) and
  prefix-cache size**. What they still lack — and what only riptide-at-runtime provides — is **measured
  cache-hit % from the server** and **CPU-core-seconds** of tool execution.

## Limitations

- **OSL excludes hidden reasoning.** These are non-reasoning models; `thought` text is in OSL, but there's
  no separate reasoning-token channel (unlike o-series / extended-thinking models).
- **cISL split is append-only arithmetic** — exact for ReAct/CodeAct histories, approximate for Moatless
  (flagged in its report).
- **No real CPU-core-seconds** — only wall-time where present (AxT `env_exec_time`). Cache-hit is modeled
  from token counts, not read from a live server.
- **Sampling** — see per-report session counts; percentiles are stable but tails are sample-bounded.
