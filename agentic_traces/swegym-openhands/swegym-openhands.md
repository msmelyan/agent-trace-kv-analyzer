# swegym-openhands — trace analysis

## About this dataset

**`SWE-Gym/OpenHands-Sampled-Trajectories`** — sampled rollouts of the **OpenHands** agent (the CodeAct architecture) on **SWE-Gym** (ICML 2025), the first publicly-usable environment of 2.4K real SWE tasks with executable test harnesses. Uses native OpenAI **tool-calling** (`str_replace_editor`, `execute_bash`, `finish`) rather than text-parsed actions, with explicit `tool` role messages carrying results. Each row has a `resolved` flag and full `test_result`. OpenHands can also condense/summarize long histories, which is why a few sessions show non-monotonic context.

**Format:** parquet ×3, rich `messages[]` (`role`, `content`, `tool_calls[]`, `tool_call_id`). Tokens via tiktoken `o200k_base`; **OSL includes tool-call arguments**. Sampled 2,500 sessions.

**Sessions analyzed:** 2,500  ·  **assistant steps (LLM calls):** 46,662


## Conversation shape

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| steps / session | 18.7 | 14 | 48 | 50 | 50 | 50 |

## Per-step ISL / OSL (tokens)

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| **ISL** (input / call) | 14.5k | 11.9k | 31.4k | 39.3k | 57.1k | 109.5k |
| **OSL** (output / call) | 123.1 | 41 | 315 | 413 | 648 | 16.4k |

## Prefix-cache modeling

Computed per session from the exact cumulative-context token counts. cISL (cache hit) = prior cumulative prompt; nISL (new) = growth since last call; context shrink ⇒ counted as a **compaction** (cache reset).

- **Aggregate prefix-cache hit ratio** = `1 − Σ nISL / Σ ISL` = **95.5%** (Σ input 677.52M tok → only 30.38M tok actually need prefill).
- **Per-session hit ratio:** mean 86.8% · p50 90.6% · p90 96.9%
- **Sessions with a compaction / context shrink:** 0 (0.0%)

### Prefix-cache *size* (peak context per session)

KV-cache footprint uses **Qwen2.5-Coder-32B** config (layers=64, kv_heads=8, head_dim=128, fp16) → **256 KiB/token** (= 2·L·H·d·2B), vLLM block = 16 tok = 4.00 MiB.

| peak-context percentile | tokens | KV blocks (÷16) | KV cache size |
|---|--:|--:|--:|
| mean | 14.4k | 904 | 3612 MiB |
| p50 | 10.7k | 669 | 2674 MiB |
| p90 | 34.9k | 2.2k | 8717 MiB |
| p95 | 44.2k | 2.8k | 11043 MiB |
| p99 | 64.4k | 4.0k | 16101 MiB |
| max | 109.6k | 6.8k | 27396 MiB |

## Tool / action mix

| tool | calls | share |
|---|--:|--:|
| `str_replace_editor` | 30,943 | 66.3% |
| `execute_bash` | 10,110 | 21.7% |
| `none` | 4,363 | 9.4% |
| `finish` | 1,136 | 2.4% |
| `bash` | 109 | 0.2% |
| `view` | 1 | 0.0% |
