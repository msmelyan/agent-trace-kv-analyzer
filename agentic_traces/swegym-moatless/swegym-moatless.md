# swegym-moatless — trace analysis

## About this dataset

**`SWE-Gym/MoatlessTools-Sampled-Trajectories`** — rollouts of the **Moatless Tools** agent on SWE-Gym (SWE-Gym project, Qwen 7B/32B releases). Moatless is architecturally different from the others: a **structured state machine** that moves through explicit phases — `SearchCode` → `IdentifyCode` → `DecideRelevance` → `PlanToCode` → `EditCode` — **rebuilding a focused prompt for each phase** instead of appending to one growing transcript. That design is the reason its prefix-cache hit is far lower than the append-only ReAct/CodeAct agents (see below) — a clean natural experiment in how agent architecture drives GPU cost.

**Format:** zip dumps of full eval runs; I sampled 2 bench-lite 32B releases (`zero`, `it1-v2`). Each `prompt_logs/<instance>/NN_<Action>.json` = one LLM call (`messages` + `completion`), so ISL/OSL are read per call and tokenized with tiktoken `o200k_base`. Grouped by release.

**Sessions analyzed:** 600  ·  **assistant steps (LLM calls):** 5,025


## Conversation shape

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| steps / session | 8.4 | 7 | 13 | 17 | 24.0 | 25 |

## Per-step ISL / OSL (tokens)

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| **ISL** (input / call) | 4.1k | 3.0k | 8.6k | 10.6k | 14.5k | 26.2k |
| **OSL** (output / call) | 293.4 | 198 | 682 | 839.8 | 1.1k | 2.0k |

## Prefix-cache modeling

Computed per session from the exact cumulative-context token counts. cISL (cache hit) = prior cumulative prompt; nISL (new) = growth since last call; context shrink ⇒ counted as a **compaction** (cache reset).

- **Aggregate prefix-cache hit ratio** = `1 − Σ nISL / Σ ISL` = **36.4%** (Σ input 20.72M tok → only 13.17M tok actually need prefill).
- **Per-session hit ratio:** mean 33.7% · p50 33.8% · p90 46.1%
- **Sessions with a compaction / context shrink:** 569 (94.8%)

> ⚠️ **Non-append-only agent.** 95% of sessions shrink context between calls — this agent rebuilds prompts per action rather than appending, so the append-only cISL arithmetic is only approximate here. The high shrink rate is itself the finding: this architecture is **prefix-cache-hostile** vs. append-only ReAct agents (90%+ hit).

### Prefix-cache *size* (peak context per session)

KV-cache footprint uses **Qwen2.5-Coder-32B** config (layers=64, kv_heads=8, head_dim=128, fp16) → **256 KiB/token** (= 2·L·H·d·2B), vLLM block = 16 tok = 4.00 MiB.

| peak-context percentile | tokens | KV blocks (÷16) | KV cache size |
|---|--:|--:|--:|
| mean | 7.7k | 483 | 1932 MiB |
| p50 | 6.9k | 435 | 1737 MiB |
| p90 | 13.6k | 851 | 3401 MiB |
| p95 | 15.1k | 945 | 3778 MiB |
| p99 | 19.2k | 1.2k | 4806 MiB |
| max | 26.8k | 1.7k | 6708 MiB |

## Tool / action mix

| tool | calls | share |
|---|--:|--:|
| `PlanToCode` | 1,393 | 27.7% |
| `SearchCode` | 927 | 18.4% |
| `IdentifyCode` | 880 | 17.5% |
| `EditCode` | 856 | 17.0% |
| `DecideRelevance` | 831 | 16.5% |
| `ClarifyCodeChange` | 138 | 2.7% |

## By group (context budget / attempt / release)

| group | sessions | mean steps | mean peak-ctx (tok) | mean cache-hit |
|---|--:|--:|--:|--:|
| it1-v2 | 300 | 8.8 | 7.7k | 34.2% |
| zero | 300 | 7.9 | 7.8k | 33.2% |
