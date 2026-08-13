# AgentX Weka corpus — schema and interpretation notes

Inspected before implementing the quantitative analysis.

- **Dataset:** `semianalysisai/cc-traces-weka-062126`
- **Dataset revision:** `23f152f6f0f9399a85901b89a6458def0ef16729`
- **AIPerf reference implementation:** `ai-dynamo/aiperf` at
  `c26fe88bd88d32740c83b4637c1e573e06df5ecc`
- **Reference files:** `weka_trace_models.py`, `weka_trace.py`,
  `weka_agent_chains.py`, `weka_metric_prepass.py`, and
  `replay_dependencies.py`.

## Raw trace object

Every JSONL row is one captured Claude Code session:

| field | observed value / interpretation |
|---|---|
| `id` | Opaque trace/session ID. |
| `models` | Models observed in the trace. AIPerf does not trust this list for routing; it builds model mappings from the per-request `model` fields. |
| `block_size` | `64` in all 393 rows. |
| `hash_id_scope` | `local` in all rows. Hash identifiers are meaningful only within one trace; equal integers in different traces do not imply shared content. |
| `requests` | Time-ordered outer entries: streaming calls (`type: "s"`), normal calls (`type: "n"`), and explicit subagent blocks (`type: "subagent"`). |

The corpus contains 56,432 streaming outer calls, 366 normal outer calls,
1,697 explicit subagent blocks, and 42,029 model requests nested inside those
blocks. All 98,827 model requests satisfy `in == len(hash_ids) * 64`.

## Normal and streaming requests

| field | interpretation followed by this analysis |
|---|---|
| `t` | Request-start timestamp in seconds relative to the root trace start. Current subagent inner timestamps are also root-relative. AIPerf retains a compatibility fallback: an inner timestamp earlier than its subagent marker is treated as subagent-relative. |
| `type` | `s` is streaming and includes `ttft`; `n` is non-streaming. This is transport behavior, not parent/subagent identity. |
| `model` | Recorded model identifier for that request. |
| `in` | Input length represented by Weka hash blocks. In this corpus it is always an exact multiple of 64 and exactly equals `len(hash_ids) * block_size`. The dataset card warns that this is block-count-derived, not an exact tokenizer count, and may overcount true prompts in the heavy cache-write tail. |
| `out` | Recorded generated output length. |
| `hash_ids` | Ordered identifiers for the prompt's 64-token KV blocks. Only a *leading* common run is reusable as a prefix. Set overlap or repeated IDs after the first mismatch are not prefix reuse. |
| `api_time` | Recorded server-processing duration in seconds; request interval is `[t, t + api_time)`. Missing, negative, or non-finite values are treated as zero-width by AIPerf. |
| `think_time` | Recorded client-side delay before the request. It is analyzed directly, but it is not the default replay delay. |
| `ttft` | Recorded time to first token for streaming calls. |

For a dependent stream, AIPerf computes the faithful idle delay as
`max(0, t_current - (t_previous + api_time_previous))`. Using the raw
start-to-start timestamp difference would double-count the previous server's
latency. `think_time` is used instead only in AIPerf's explicit
`use_think_time_only` mode.

## `hash_ids` and theoretical prefix reuse

`hash_ids` describe reusable content structure, not an observed cache hit.
This analysis reports two block-granular quantities:

1. **Consecutive-stream LCP:** longest common prefix against the previous
   request on the same reconstructed stream.
2. **Weka theoretical available prefix:** AIPerf's `weka_metric_prepass`
   semantics—within one trace-local namespace, consume all reconstructed
   streams in deterministic global time order and count the leading blocks
   already present in a shared `seen` set.

The second measure captures reuse across branches. Neither measure accounts
for finite cache capacity, eviction, block residency, routing, or whether a
serving system retained the blocks. They are theoretical reuse opportunities.

## Explicit subagent entries and SPAWN/JOIN

A `type: "subagent"` outer entry is a marker with `agent_id`, `t`,
`duration_ms`, status/summary fields, and nested model `requests`.

AIPerf reconstructs it as follows:

- Hash-LCP chain detection is run inside the nested request list. One recorded
  subagent entry can therefore become multiple child streams.
- The branch SPAWNs from the last reconstructed parent turn preceding the
  subagent marker's outer-list position.
- Its end is `entry.t + duration_ms / 1000`; if duration is absent, AIPerf uses
  the latest inner request end.
- The JOIN gate is the first later parent turn whose start timestamp is at or
  after that end.
- No preceding parent turn means the subagent is dropped. No following parent
  turn means a background branch with no JOIN gate.
- Adjacent subagents with the same `(preceding, following)` anchors collapse
  into one multi-child branch, so the following parent turn waits for all of
  them.

## Reconstructing flattened independent chains

The outer list can contain interleaved requests from unlabeled workers. AIPerf
does not serialize it naively. Its default detector:

1. processes requests in `(t, outer_index)` order;
2. extends the deepest same-model chain whose entire tail `hash_ids` is a
   prefix of the new request and whose prior request has ended;
3. otherwise forks from the chain with the deepest hash LCP;
4. performs an offline seam pass: a context shrink is spliced back onto the
   original chain only when the longer pre-shrink state is not later resumed;
5. always treats cross-model attachment as a new stream; and
6. keeps hash-less requests on the main stream because they carry no LCP
   evidence.

Live non-main chains are classified with AIPerf's defaults as solo agents
(`::fa:`), parallel worker-group members (`::wg:`), or one-shot/reduction
sidecars (`::aux:`). Worker groups require both a shared fork point and
overlapping active intervals; three members are required by default.

Within each reconstructed stream requests remain sequential. Across streams,
AIPerf derives a completion frontier: a request depends on the latest request
from every other stream that is known to have completed by its recorded start.
Requests whose intervals overlap receive no dependency edge and remain
parallel.
