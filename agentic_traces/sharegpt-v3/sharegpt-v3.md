# sharegpt-v3 — trace analysis

## About this dataset

**`ShareGPT_V3_unfiltered_cleaned_split`** — ~94K real human↔ChatGPT **web-chat** conversations shared to sharegpt.com; the canonical dataset vLLM-style benchmarks (and the *Frontier* paper's SharedGPT trace) replay for realistic serving workloads. Included here as a **non-agentic baseline**: general chat + coding help, **no tools, no environment**. Modeled in the multi-turn serving view — each assistant (`gpt`) turn is one LLM call whose ISL is the accumulated conversation.

**Format:** JSON list of `{id, conversations:[{from, value}]}`. Tokens via tiktoken `o200k_base`. Sampled 4,000 conversations. KV sizing uses **Llama-3.1-8B** (a typical chat-serving model; model-agnostic data). Contrast the others: chat is decode-balanced and short-session, agents are prefill-heavy and long.

**Sessions analyzed:** 3,983  ·  **assistant steps (LLM calls):** 15,813


## Conversation shape

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| steps / session | 4.0 | 3 | 7 | 9 | 14 | 68 |

## Per-step ISL / OSL (tokens)

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| **ISL** (input / call) | 717.8 | 634 | 1.6k | 1.7k | 1.9k | 9.3k |
| **OSL** (output / call) | 285.9 | 259 | 570 | 701 | 778 | 1.8k |

## Prefix-cache modeling

Computed per session from the exact cumulative-context token counts. cISL (cache hit) = prior cumulative prompt; nISL (new) = growth since last call; context shrink ⇒ counted as a **compaction** (cache reset).

- **Aggregate prefix-cache hit ratio** = `1 − Σ nISL / Σ ISL` = **90.1%** (Σ input 11.35M tok → only 1.13M tok actually need prefill).
- **Per-session hit ratio:** mean 71.2% · p50 93.1% · p90 98.8%
- **Sessions with a compaction / context shrink:** 0 (0.0%)

### Prefix-cache *size* (peak context per session)

KV-cache footprint uses **Llama-3.1-8B** config (layers=32, kv_heads=8, head_dim=128, fp16) → **128 KiB/token** (= 2·L·H·d·2B), vLLM block = 16 tok = 2.00 MiB.

| peak-context percentile | tokens | KV blocks (÷16) | KV cache size |
|---|--:|--:|--:|
| mean | 1.4k | 89 | 177 MiB |
| p50 | 1.7k | 104 | 207 MiB |
| p90 | 2.2k | 140 | 280 MiB |
| p95 | 2.4k | 149 | 296 MiB |
| p99 | 2.8k | 175 | 348 MiB |
| max | 9.6k | 599 | 1198 MiB |

## Tool / action mix

| tool | calls | share |
|---|--:|--:|
| `chat` | 15,813 | 100.0% |
