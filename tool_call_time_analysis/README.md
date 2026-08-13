# Tool-call timing in TraceLab and SWE-chat

This directory compares tool-call timing from two public datasets of real coding-agent
sessions. It answers two related questions:

1. How are tool calls distributed across latency buckets?
2. Where TraceLab has both measurements, how much end-to-end time lies outside the
   runner-reported tool execution time?

The central result is that short tools are common, but the distributions are extremely
heavy-tailed. In TraceLab's paired Codex data, the median positive residual around a call
is small (159 ms overall), yet the residual sums to 129.7 hours because a small long tail
dominates. SWE-chat independently shows a 121 ms median observable tool turnaround, but
its P99 is roughly 302 seconds.

## Metric definitions

TraceLab exposes two distinct fields:

- **Internal latency**: runner-reported duration of the tool itself.
- **End-to-end or wall latency**: time from the model emitting the call until the result
  is recorded.
- **Residual overhead**: `max(wall - internal, 0)`.
- **Effective latency**: internal latency when present, otherwise wall latency. This is
  TraceLab's standard tool-latency metric.

SWE-chat exposes timestamped `tool_use` and `tool_result` events, but no separate internal
duration. Its measurement here is therefore only:

```text
observable wall turnaround = tool_result timestamp - tool_use timestamp
```

That interval can include actual execution, harness/runtime work, scheduling, approval or
human waiting, and timestamp instrumentation effects. It is not a direct measure of CPU
execution time.

## Dataset snapshots and coverage

| Dataset | Snapshot analyzed | Timing population used here |
|---|---|---:|
| TraceLab | `syfi_coding_trace_v0.0.2.duckdb`, SHA-256 `a7bab286...e744` | 304,208 Claude and 296,064 Codex positive effective timings; 337,674 Codex calls have both wall and internal timing |
| SWE-chat | `conversations.parquet`, SHA-256 `9ee1d937...1c36`; `sessions.parquet`, `2ada6397...a95` | 330,908 positive paired wall turnarounds from 355,942 tool-use events |

TraceLab v0.0.2 describes its expanded public corpus as 665,453 agent steps across 8,058
sessions and 52 pseudonymous users, including 743,819 normalized tool calls. SWE-chat's
release has 5,851 sessions; the downloaded
sessions table contains 189 distinct developer IDs and 201 repository IDs, while the
complete release contains 205 repository rows.

### Provenance

- **TraceLab** normalizes locally emitted Claude Code and Codex histories contributed by
  users, then pseudonymizes identifiers and strips paths and tool inputs for the public
  release. The v0.0.2 release is an opt-in, sanitized trace corpus rather than controlled
  benchmark execution.
- **SWE-chat** discovers public GitHub repositories whose developers enabled Entire CLI
  checkpoint logging. Entire stores agent transcripts on `entire/checkpoints/v1` branches;
  SWE-chat parses those logs into structured sessions, tool calls, results, and code
  attribution. It therefore selects for developers willing to push these checkpoints to
  public repositories.

## TraceLab: effective tool-latency distribution

These percentages use TraceLab's standard `internal if present, otherwise wall` metric
and strictly positive latency. The final rows extend the original sub-second table to all
tool calls.

| Effective latency | Claude calls | Claude | Codex calls | Codex |
|---|---:|---:|---:|---:|
| <10 ms | 18,397 | 6.05% | 8,319 | 2.81% |
| 10-25 ms | 27,178 | 8.93% | 11,899 | 4.02% |
| 25-50 ms | 43,914 | 14.44% | 14,443 | 4.88% |
| 50-100 ms | 51,070 | 16.79% | 32,345 | 10.93% |
| 100-250 ms | 50,975 | 16.76% | 31,428 | 10.62% |
| 250-500 ms | 15,920 | 5.23% | 27,484 | 9.28% |
| 500 ms-1 s | 9,455 | 3.11% | 24,856 | 8.40% |
| **Total <1 s** | **216,909** | **71.30%** | **150,774** | **50.93%** |
| 1-10 s | 56,686 | 18.63% | 95,696 | 32.32% |
| 10 s-1 min | 17,687 | 5.81% | 42,196 | 14.25% |
| 1-10 min | 11,681 | 3.84% | 7,335 | 2.48% |
| 10 min-1 h | 1,011 | 0.33% | 52 | 0.02% |
| >=1 h | 234 | 0.08% | 11 | <0.01% |
| **All** | **304,208** | **100%** | **296,064** | **100%** |

The exact long-form counts are in
[`results/tracelab_effective_latency_by_provider.csv`](results/tracelab_effective_latency_by_provider.csv).

## TraceLab: Codex residual overhead within internal-latency buckets

This is the apples-to-apples paired population: every row has both wall and internal
timing. Calls are bucketed by runner-reported internal execution time, then the residual
is summarized inside each bucket.

| Internal tool time | Calls | Share | Median residual | P90 residual | Median E2E |
|---|---:|---:|---:|---:|---:|
| <10 ms | 138,369 | 40.98% | 88 ms | 303 ms | 88 ms |
| 10-25 ms | 5,814 | 1.72% | 170 ms | 370 ms | 188 ms |
| 25-50 ms | 8,009 | 2.37% | 169 ms | 415 ms | 209 ms |
| 50-100 ms | 21,269 | 6.30% | 163 ms | 449 ms | 226 ms |
| 100-250 ms | 23,809 | 7.05% | 190 ms | 389 ms | 382 ms |
| 250-500 ms | 22,715 | 6.73% | 192 ms | 395 ms | 545 ms |
| 500 ms-1 s | 15,704 | 4.65% | 198 ms | 349 ms | 914 ms |
| 1-10 s | 71,282 | 21.11% | 163 ms | 263 ms | 1,301 ms |
| 10 s-1 min | 24,090 | 7.13% | 6 ms | 191 ms | 30,002 ms |
| 1-10 min | 6,611 | 1.96% | 1 ms | 16 ms | 119,954 ms |
| 10 min-1 h | 2 | <0.01% | 38 ms | 56 ms | 1,669,338 ms |
| **All paired** | **337,674** | **100%** | **159 ms** | **307 ms** | — |

This answers the placement question directly: for an internally measured 25-50 ms tool
call, the median observed residual is 169 ms and the median end-to-end latency is 209 ms.
Reducing a CPU/GPU network hop by one or two milliseconds would be small relative to that
median residual in this trace.

Across all paired calls, wall time totals 598.6 hours, internal duration totals 470.1
hours, and the positive residual totals 129.7 hours, or 21.67% of wall time. The residual
mean is 1.383 seconds, much greater than its 159 ms median, demonstrating the heavy tail.

### Residual by high-volume paired tool

| Tool | Calls | E2E | Internal | Residual | Residual P50 / P90 / P99 |
|---|---:|---:|---:|---:|---:|
| `exec_command` | 247,647 | 162.2 h | 53.6 h | 108.7 h | 0.167 / 0.378 / 9.426 s |
| `write_stdin` | 73,997 | 429.3 h | 411.5 h | 18.7 h | 0.009 / 0.102 / 29.957 s |
| `apply_patch` | 9,635 | 1.69 h | 1.16 h | 0.53 h | 0.018 / 0.389 / 2.547 s |
| `shell_command` | 6,131 | 5.33 h | 3.82 h | 1.84 h | 0.041 / 0.156 / 4.015 s |

The launch path (`exec_command`) owns most of the aggregate residual. Long-running
continuations (`write_stdin`) spend nearly all their wall time in internally reported
execution, although they still have a long residual tail.

## SWE-chat: observable wall-turnaround distribution

Tool calls are paired by `(session_id, tool_call_id, occurrence)` because some sessions
reuse call IDs. Eight negative timestamp differences and 889 zero differences are
excluded from the positive distribution; 799 timestamped tool uses have no paired result.

| Wall turnaround | Calls | Share |
|---|---:|---:|
| <10 ms | 43,658 | 13.19% |
| 10-25 ms | 37,653 | 11.38% |
| 25-50 ms | 35,348 | 10.68% |
| 50-100 ms | 39,944 | 12.07% |
| 100-250 ms | 36,003 | 10.88% |
| 250-500 ms | 24,695 | 7.46% |
| 500 ms-1 s | 20,895 | 6.31% |
| **Total <1 s** | **238,196** | **71.98%** |
| 1-10 s | 55,954 | 16.91% |
| 10 s-1 min | 21,928 | 6.63% |
| 1-10 min | 13,091 | 3.96% |
| 10 min-1 h | 1,516 | 0.46% |
| >=1 h | 223 | 0.07% |
| **All** | **330,908** | **100%** |

Summary: P50 121 ms, P90 12.555 s, P99 302.393 s; 88.89% are below 10
seconds and 95.52% below one minute.

### Highest-volume SWE-chat tools

| Tool | Calls | Share | P50 | P90 | P99 |
|---|---:|---:|---:|---:|---:|
| Bash | 120,717 | 36.48% | 1.476 s | 27.074 s | 300.236 s |
| Read | 64,446 | 19.48% | 27 ms | 244 ms | 1.378 s |
| Edit | 64,353 | 19.45% | 37 ms | 918 ms | 82.850 s |
| Grep | 28,858 | 8.72% | 205 ms | 628 ms | 1.686 s |
| TaskUpdate | 8,741 | 2.64% | 8 ms | 81 ms | 2.080 s |
| Write | 8,668 | 2.62% | 35 ms | 877 ms | 180.891 s |
| Agent | 5,078 | 1.53% | 82.883 s | 380.639 s | 1,329.208 s |
| Glob | 3,943 | 1.19% | 419 ms | 1.964 s | 19.138 s |
| TaskCreate | 3,285 | 0.99% | 7 ms | 71 ms | 799 ms |
| ToolSearch | 3,104 | 0.94% | 3 ms | 112 ms | 414 ms |
| Task | 2,681 | 0.81% | 64.412 s | 264.072 s | 2,190.934 s |
| TodoWrite | 2,397 | 0.72% | 25 ms | 215 ms | 947 ms |

The long tail is partly semantic: `Agent`, `Task`, `TaskOutput`,
`AskUserQuestion`, and planning transitions can legitimately wait for nested work or a
human. They should not be interpreted as CPU execution benchmarks.

## Cross-dataset interpretation

| Observation | What the data supports |
|---|---|
| Most tool calls are sub-second | 71.30% of TraceLab Claude effective timings, 50.93% of TraceLab Codex effective timings, and 71.98% of SWE-chat wall turnarounds are under one second. |
| Fast execution does not imply fast E2E | In paired TraceLab Codex calls with 25-50 ms internal execution, median E2E is 209 ms because median residual is 169 ms. |
| The tail dominates aggregate time | TraceLab's paired residual mean is 1.383 s versus a 159 ms median; SWE-chat's P99 wall turnaround is about five minutes. |
| Tool type matters | Reads and edits have millisecond medians, while shell, nested-agent, and human-interaction tools reach seconds or minutes. |
| CPU/GPU placement cannot be decided from SWE-chat alone | SWE-chat provides no internal execution duration, so execution and harness/network overhead cannot be separated. |

Do not directly compare TraceLab's **effective** latency percentages with SWE-chat's
**wall** percentages as if they were the same metric. TraceLab can preferentially use
internal duration; SWE-chat cannot.

## Caveats

- TraceLab residual is an **upper-bound proxy** for everything around tool execution, not
  a direct measurement of network, scheduling, shell startup, or approval time.
- TraceLab has no explicit approval-start/end markers. Human approval overhead is not
  independently identifiable.
- The residual is clamped at zero. In 2,081 paired rows (0.616%), reported internal time
  exceeds observed wall time, presumably from measurement resolution or clock/instrumentation
  differences; for those rows `internal + residual = wall` does not hold exactly.
- TraceLab's `<10 ms` internal bucket includes 138,369 calls and has a zero-millisecond
  internal median. Treat very fine internal buckets cautiously because runner-reported
  duration can be quantized or rounded.
- SWE-chat timing coverage is 98.61% Claude Code among positive pairs. The sessions table
  contains 213 Codex sessions, but the structured conversation table contains no Codex
  `tool_use` events, so this analysis produces no SWE-chat Codex timing result.
- Both sources are opt-in/public developer populations, not representative samples of all
  developers or private enterprise production systems.
- Per-call additive totals can double-count elapsed time when tools overlap or run in
  parallel.

## Reproduction

Install this repository's normal requirements and download the datasets under their
respective access terms. Then run:

```bash
python tool_call_time_analysis/analyze.py \
  --tracelab-db /path/to/syfi_coding_trace_v0.0.2.duckdb \
  --swe-conversations /path/to/conversations.parquet \
  --swe-sessions /path/to/sessions.parquet \
  --output-dir tool_call_time_analysis/results
```

The script writes aggregate CSVs only. It never exports commands, transcript text, paths,
identifiers, or per-call examples.

## References

1. Kan Zhu et al., [TraceLab: Characterizing Coding Agent Workloads for LLM
   Serving](https://arxiv.org/abs/2606.30560), 2026.
2. [TraceLab v0.0.2 release and dataset checksums](https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.2).
3. [TraceLab repository and methodology](https://github.com/uw-syfi/TraceLab).
4. Joachim Baumann et al., [SWE-chat: Coding Agent Interactions From Real Users in the
   Wild](https://arxiv.org/abs/2604.20779), 2026.
5. [SWE-chat dataset card](https://huggingface.co/datasets/SALT-NLP/SWE-chat).
6. [SWE-chat repository](https://github.com/SALT-NLP/SWE-chat).

The aggregate CSV snapshots in this directory were generated locally on 2026-08-12.
