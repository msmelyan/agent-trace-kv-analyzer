# axt-swe-agent-lm-32b — trace analysis

## About this dataset

**`AxT-dev/swe-agent-lm-32b-r2e-gym-trajectories`** — trajectories from running **SWE-agent-LM-32B** (the open, Qwen2.5-32B-based SWE agent model from the SWE-agent/SWE-smith line) against **R2E-Gym**, a library of executable real-world software-engineering task environments (the same lineage as DeepSWE, which trains on a 4.5K-problem R2E-Gym subset). The agent runs a ReAct loop — read the issue, explore `/testbed`, edit files, run tests — until it submits a patch. This release repeats the run at **4 context budgets (16K / 32K / 98K / 131K) × 2 attempts**, which is what makes it uniquely useful for studying how context length shapes cISL and cache behavior.

**Format:** JSONL, one trajectory per line (`trajectory_steps[]` with `thought`/`action`/`observation`). **Token counts are MEASURED** here (`token_usage_prompt`/`token_usage_completion`), not tiktoken estimates. Also carries per-step timing (`llm_exec_time`, `env_exec_time`) and a final `reward`. Sampled 700 sessions/file.

**Sessions analyzed:** 4,000  ·  **assistant steps (LLM calls):** 122,540

**Resolve/reward:** mean 0.393 over 4,000 scored sessions (39.3% fully resolved)


## Conversation shape

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| steps / session | 30.6 | 23 | 61 | 79 | 150 | 200 |

## Per-step ISL / OSL (tokens)

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| **ISL** (input / call) | 18.9k | 13.7k | 39.2k | 55.0k | 91.5k | 131.0k |
| **OSL** (output / call) | 183.3 | 68 | 481 | 717 | 1.3k | 88.4k |

## Prefix-cache modeling

Computed per session from the exact cumulative-context token counts. cISL (cache hit) = prior cumulative prompt; nISL (new) = growth since last call; context shrink ⇒ counted as a **compaction** (cache reset).

- **Aggregate prefix-cache hit ratio** = `1 − Σ nISL / Σ ISL` = **97.0%** (Σ input 2.32B tok → only 70.38M tok actually need prefill).
- **Per-session hit ratio:** mean 93.6% · p50 94.4% · p90 97.7%
- **Sessions with a compaction / context shrink:** 4 (0.1%)

### Prefix-cache *size* (peak context per session)

KV-cache footprint uses **Qwen2.5-Coder-32B** config (layers=64, kv_heads=8, head_dim=128, fp16) → **256 KiB/token** (= 2·L·H·d·2B), vLLM block = 16 tok = 4.00 MiB.

| peak-context percentile | tokens | KV blocks (÷16) | KV cache size |
|---|--:|--:|--:|
| mean | 23.2k | 1.4k | 5793 MiB |
| p50 | 17.1k | 1.1k | 4284 MiB |
| p90 | 40.8k | 2.6k | 10206 MiB |
| p95 | 55.0k | 3.4k | 13744 MiB |
| p99 | 100.1k | 6.3k | 25031 MiB |
| max | 131.0k | 8.2k | 32749 MiB |

## Tool / action mix

| tool | calls | share |
|---|--:|--:|
| `file_editor` | 71,979 | 58.7% |
| `execute_bash` | 47,463 | 38.7% |
| `finish` | 3,098 | 2.5% |

## By group (context budget / attempt / release)

| group | sessions | mean steps | mean peak-ctx (tok) | mean cache-hit |
|---|--:|--:|--:|--:|
| ctx131072 | 1,000 | 41.0 | 29.9k | 94.5% |
| ctx16384 | 1,000 | 18.0 | 14.8k | 91.5% |
| ctx32768 | 1,000 | 28.0 | 21.0k | 94.1% |
| ctx98304 | 1,000 | 35.6 | 27.1k | 94.4% |
