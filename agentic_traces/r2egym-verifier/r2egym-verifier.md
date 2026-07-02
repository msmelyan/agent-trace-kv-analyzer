# r2egym-verifier — trace analysis

## About this dataset

**`R2E-Gym/R2EGym-Verifier-Trajectories`** — data for training R2E-Gym's **verifier** (an outcome reward model used for best-of-n selection, one of the paper's headline results). Each row feeds a *whole* agent trajectory to a judge that emits a single short verdict, with a `rewards` label. So unlike the others this is **not a multi-turn agent** — it's a **one-shot classification** workload: exactly 1 LLM call/session, large ISL (reads the full trajectory ~15k tok), tiny OSL (~9 tok), and no prefix reuse.

**Format:** parquet, `messages[]` + `docker_images` + `rewards`. Tokens via tiktoken `o200k_base`. Sampled 3,000 sessions. Included as a deliberate *contrast* to the agent traces.

**Sessions analyzed:** 3,000  ·  **assistant steps (LLM calls):** 3,000

**Resolve/reward:** mean 0.463 over 3,000 scored sessions (46.3% fully resolved)


## Conversation shape

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| steps / session | 1.0 | 1 | 1 | 1 | 1 | 1 |

## Per-step ISL / OSL (tokens)

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| **ISL** (input / call) | 15.3k | 14.0k | 25.8k | 27.7k | 29.4k | 31.1k |
| **OSL** (output / call) | 9 | 9 | 9 | 9 | 9 | 9 |

## Prefix-cache modeling

Computed per session from the exact cumulative-context token counts. cISL (cache hit) = prior cumulative prompt; nISL (new) = growth since last call; context shrink ⇒ counted as a **compaction** (cache reset).

- **Aggregate prefix-cache hit ratio** = `1 − Σ nISL / Σ ISL` = **0.0%** (Σ input 45.82M tok → only 45.82M tok actually need prefill).
- **Per-session hit ratio:** mean 0.0% · p50 0.0% · p90 0.0%
- **Sessions with a compaction / context shrink:** 0 (0.0%)

### Prefix-cache *size* (peak context per session)

KV-cache footprint uses **Qwen2.5-Coder-32B** config (layers=64, kv_heads=8, head_dim=128, fp16) → **256 KiB/token** (= 2·L·H·d·2B), vLLM block = 16 tok = 4.00 MiB.

| peak-context percentile | tokens | KV blocks (÷16) | KV cache size |
|---|--:|--:|--:|
| mean | 15.3k | 956 | 3820 MiB |
| p50 | 14.0k | 877 | 3504 MiB |
| p90 | 25.8k | 1.6k | 6441 MiB |
| p95 | 27.7k | 1.7k | 6918 MiB |
| p99 | 29.4k | 1.8k | 7343 MiB |
| max | 31.1k | 1.9k | 7777 MiB |

## Tool / action mix

| tool | calls | share |
|---|--:|--:|
| `none` | 3,000 | 100.0% |
