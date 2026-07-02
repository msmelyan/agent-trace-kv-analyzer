# r2egym-sft — trace analysis

## About this dataset

**`R2E-Gym/R2EGym-SFT-Trajectories`** — the **supervised fine-tuning** set from **R2E-Gym** (COLM 2025), the project that procedurally generates executable SWE environments and trains open-weight SWE agents. These are successful/curated agent trajectories used as SFT data to teach the ReAct tool-use loop. Actions use the `<function=name>…</function>` tool-call convention (`file_editor`, `execute_bash`, `search`, `finish`).

**Format:** parquet, `messages[]` of `{role, content}` (OpenAI-style). No token counts → tiktoken `o200k_base`. All 3,231 sessions analyzed.

**Sessions analyzed:** 3,231  ·  **assistant steps (LLM calls):** 52,066


## Conversation shape

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| steps / session | 16.1 | 14 | 26 | 32 | 41 | 42 |

## Per-step ISL / OSL (tokens)

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| **ISL** (input / call) | 8.8k | 8.1k | 16.0k | 18.9k | 23.8k | 29.0k |
| **OSL** (output / call) | 160.3 | 63 | 419 | 534 | 831 | 3.6k |

## Prefix-cache modeling

Computed per session from the exact cumulative-context token counts. cISL (cache hit) = prior cumulative prompt; nISL (new) = growth since last call; context shrink ⇒ counted as a **compaction** (cache reset).

- **Aggregate prefix-cache hit ratio** = `1 − Σ nISL / Σ ISL` = **92.9%** (Σ input 459.45M tok → only 32.49M tok actually need prefill).
- **Per-session hit ratio:** mean 90.9% · p50 90.9% · p90 95.3%
- **Sessions with a compaction / context shrink:** 0 (0.0%)

### Prefix-cache *size* (peak context per session)

KV-cache footprint uses **Qwen2.5-Coder-32B** config (layers=64, kv_heads=8, head_dim=128, fp16) → **256 KiB/token** (= 2·L·H·d·2B), vLLM block = 16 tok = 4.00 MiB.

| peak-context percentile | tokens | KV blocks (÷16) | KV cache size |
|---|--:|--:|--:|
| mean | 12.6k | 790 | 3160 MiB |
| p50 | 11.6k | 728 | 2910 MiB |
| p90 | 20.5k | 1.3k | 5116 MiB |
| p95 | 23.8k | 1.5k | 5961 MiB |
| p99 | 26.8k | 1.7k | 6695 MiB |
| max | 29.4k | 1.8k | 7341 MiB |

## Tool / action mix

| tool | calls | share |
|---|--:|--:|
| `file_editor` | 29,619 | 56.9% |
| `execute_bash` | 13,912 | 26.7% |
| `search` | 5,347 | 10.3% |
| `finish` | 3,134 | 6.0% |
| `none` | 54 | 0.1% |
