# nebius-swe-agent — trace analysis

## About this dataset

**`nebius/SWE-agent-trajectories`** — a large (~80K-trajectory) corpus of **SWE-agent** runs released by Nebius, collected to train/fine-tune open SWE agents (their SWE-agent-7B/32B efforts). Uses the classic **SWE-agent ACI** (Agent-Computer Interface): the model emits a short discussion followed by a single command in a code fence — `edit`, `open`, `search_dir`, `create`, `python`, `ls` — and the environment returns the result. Each row is one issue-resolution session with a `generated_patch` and `exit_status`.

**Format:** parquet ×12, `trajectory[]` of `{role: system/user/ai, text}`. No token counts, so ISL/OSL are estimated with tiktoken `o200k_base`. Sampled 2,500 sessions. (KV sizing uses the 7B config, matching the smaller models these traces target.)

**Sessions analyzed:** 2,500  ·  **assistant steps (LLM calls):** 68,799


## Conversation shape

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| steps / session | 27.5 | 17 | 52 | 71.0 | 191.0 | 394 |

## Per-step ISL / OSL (tokens)

| metric | mean | p50 | p90 | p95 | p99 | max |
|---|--:|--:|--:|--:|--:|--:|
| **ISL** (input / call) | 10.2k | 8.4k | 22.0k | 24.2k | 26.4k | 71.6k |
| **OSL** (output / call) | 101.4 | 74 | 187 | 249 | 477.0 | 5.5k |

## Prefix-cache modeling

Computed per session from the exact cumulative-context token counts. cISL (cache hit) = prior cumulative prompt; nISL (new) = growth since last call; context shrink ⇒ counted as a **compaction** (cache reset).

- **Aggregate prefix-cache hit ratio** = `1 − Σ nISL / Σ ISL` = **96.0%** (Σ input 701.70M tok → only 27.82M tok actually need prefill).
- **Per-session hit ratio:** mean 88.8% · p50 91.0% · p90 97.0%
- **Sessions with a compaction / context shrink:** 0 (0.0%)

### Prefix-cache *size* (peak context per session)

KV-cache footprint uses **Qwen2.5-Coder-7B** config (layers=28, kv_heads=4, head_dim=128, fp16) → **56 KiB/token** (= 2·L·H·d·2B), vLLM block = 16 tok = 0.88 MiB.

| peak-context percentile | tokens | KV blocks (÷16) | KV cache size |
|---|--:|--:|--:|
| mean | 13.9k | 870 | 761 MiB |
| p50 | 10.0k | 624 | 545 MiB |
| p90 | 26.3k | 1.6k | 1441 MiB |
| p95 | 26.6k | 1.7k | 1457 MiB |
| p99 | 31.0k | 1.9k | 1693 MiB |
| max | 75.5k | 4.7k | 4129 MiB |

## Tool / action mix

| tool | calls | share |
|---|--:|--:|
| `edit` | 19,226 | 27.9% |
| `python` | 8,530 | 12.4% |
| `open` | 8,017 | 11.7% |
| `search_dir` | 4,936 | 7.2% |
| `ls` | 4,141 | 6.0% |
| `search_file` | 3,575 | 5.2% |
| `create` | 3,292 | 4.8% |
| `goto` | 2,660 | 3.9% |
| `find_file` | 2,451 | 3.6% |
| `scroll_down` | 1,787 | 2.6% |
| `submit` | 1,599 | 2.3% |
| `rm` | 1,300 | 1.9% |
