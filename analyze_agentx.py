#!/usr/bin/env python3
"""Analyze the SemiAnalysis AgentX Claude Code WekaTrace corpus.

The script operates directly on the pinned Hugging Face JSONL file, follows
AIPerf's hash-LCP chain and theoretical-prefix semantics, and writes reusable
CSV/JSON, plots, and a Markdown report.

Usage:
    python analyze_agentx.py
    python analyze_agentx.py --input agentx_data/traces.jsonl --output agentx_outputs
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import orjson

from agentx_analysis.weka_semantics import (
    EPSILON_SECONDS,
    active_interval_peak,
    detect_with_preamble,
    lcp_blocks,
    req_end,
)

DATASET_REPO = "semianalysisai/cc-traces-weka-062126"
DATASET_REVISION = "23f152f6f0f9399a85901b89a6458def0ef16729"
AIPERF_REPO = "https://github.com/ai-dynamo/aiperf"
AIPERF_REVISION = "c26fe88bd88d32740c83b4637c1e573e06df5ecc"
CONTEXT_THRESHOLDS = [32768, 65536, 131072, 262144, 524288, 1_000_000]
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def finite_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return default


def pct(values: Iterable[float], q: float) -> float:
    vals = np.asarray(list(values), dtype=float)
    return float(np.percentile(vals, q)) if vals.size else 0.0


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    vals = np.asarray(list(values), dtype=float)
    if not vals.size:
        return {"n": 0}
    out: dict[str, float | int] = {
        "n": int(vals.size),
        "min": float(vals.min()),
        "mean": float(vals.mean()),
    }
    for q in PERCENTILES:
        out[f"p{q}"] = float(np.percentile(vals, q))
    out["max"] = float(vals.max())
    return out


def fmt_num(value: float | int) -> str:
    value = float(value)
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:.0f}" if value.is_integer() else f"{value:.1f}"


def fmt_duration(seconds: float) -> str:
    if seconds >= 86400:
        return f"{seconds / 86400:.1f} d"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def normalized_inner_request(entry: dict[str, Any], req: dict[str, Any]) -> dict[str, Any]:
    if float(req["t"]) + EPSILON_SECONDS >= float(entry["t"]):
        return req
    out = dict(req)
    out["t"] = float(entry["t"]) + float(req["t"])
    return out


def classify_stream_suffix(
    worker_number: int,
    classification: tuple[str, bool, bool, tuple[int, int] | None],
) -> str:
    kind, _, _, coord = classification
    if kind == "aux":
        return f"aux:{worker_number:03d}"
    if kind == "worker_group" and coord is not None:
        return f"wg:{coord[0]:03d}_{coord[1]:03d}"
    return f"fa:{worker_number:03d}"


def make_stream_plans(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_id = trace["id"]
    top_normals = [
        (idx, entry)
        for idx, entry in enumerate(trace["requests"])
        if entry["type"] in {"n", "s"}
    ]
    subagents = [
        (idx, entry)
        for idx, entry in enumerate(trace["requests"])
        if entry["type"] == "subagent"
    ]
    top_detection, top_preamble, top_classes = detect_with_preamble(top_normals)
    streams: list[dict[str, Any]] = []

    main_chain = top_detection.chains[top_detection.main_index]
    main_requests = sorted(main_chain.requests, key=lambda x: (float(x[1]["t"]), x[0]))
    streams.append(
        {
            "stream_id": trace_id,
            "stream_kind": "parent",
            "requests": main_requests,
            "explicit_subagent_id": "",
            "source_outer_idx": None,
            "chain_index": 0,
            "fork_depth_blocks": 0,
            "replay_eligible": True,
        }
    )

    for number, ci in enumerate(top_detection.worker_indices):
        chain = top_detection.chains[ci]
        classification = top_classes[ci]
        suffix = classify_stream_suffix(number, classification)
        streams.append(
            {
                "stream_id": f"{trace_id}::{suffix}",
                "stream_kind": classification[0],
                "requests": sorted(chain.requests, key=lambda x: (float(x[1]["t"]), x[0])),
                "explicit_subagent_id": "",
                "source_outer_idx": None,
                "chain_index": number + 1,
                "fork_depth_blocks": chain.fork.depth if chain.fork else 0,
                "replay_eligible": True,
            }
        )

    main_outer_to_turn = {outer_idx: pos for pos, (outer_idx, _) in enumerate(main_requests)}
    main_outer_to_t = {outer_idx: float(req["t"]) for outer_idx, req in main_requests}
    subagent_meta: list[dict[str, Any]] = []
    nested_seams = nested_workers = nested_aux = nested_wg = 0

    for sa_index, (outer_idx, entry) in enumerate(subagents):
        normalized = [
            (inner_idx, normalized_inner_request(entry, req))
            for inner_idx, req in enumerate(entry.get("requests", []))
        ]
        detection, preamble, classes = detect_with_preamble(normalized)
        nested_seams += detection.seams_merged
        nested_workers += len(detection.worker_indices)
        nested_aux += sum(1 for ci in detection.worker_indices if classes[ci][0] == "aux")
        nested_wg += sum(
            1 for ci in detection.worker_indices if classes[ci][0] == "worker_group"
        )

        preceding = max(
            (pos for oi, pos in main_outer_to_turn.items() if oi < outer_idx),
            default=None,
        )
        replay_eligible = preceding is not None
        duration_s = max(0.0, finite_number(entry.get("duration_ms")) / 1000.0)
        if entry.get("duration_ms") is None:
            duration_s = max(
                (req_end(req) for _, req in normalized),
                default=float(entry["t"]),
            ) - float(entry["t"])
        sa_end = float(entry["t"]) + duration_s
        join_turn = None
        if replay_eligible:
            for oi, pos in sorted(main_outer_to_turn.items()):
                if oi <= outer_idx:
                    continue
                if main_outer_to_t[oi] + EPSILON_SECONDS >= sa_end:
                    join_turn = pos
                    break

        stream_ids: list[str] = []
        if detection.chains:
            chain_indices = [detection.main_index] + detection.worker_indices
            for local_pos, ci in enumerate(chain_indices):
                chain = detection.chains[ci]
                if local_pos == 0:
                    stream_id = f"{trace_id}::sa:{entry['agent_id']}"
                    stream_kind = "subagent"
                else:
                    worker_number = local_pos - 1
                    classification = classes[ci]
                    suffix = classify_stream_suffix(worker_number, classification)
                    stream_id = f"{trace_id}::sa:{entry['agent_id']}:{suffix}"
                    stream_kind = (
                        "subagent_aux"
                        if classification[0] == "aux"
                        else "subagent_worker_group"
                        if classification[0] == "worker_group"
                        else "nested_subagent"
                    )
                stream_ids.append(stream_id)
                streams.append(
                    {
                        "stream_id": stream_id,
                        "stream_kind": stream_kind,
                        "requests": sorted(
                            chain.requests, key=lambda x: (float(x[1]["t"]), x[0])
                        ),
                        "explicit_subagent_id": entry["agent_id"],
                        "source_outer_idx": outer_idx,
                        "chain_index": local_pos,
                        "fork_depth_blocks": chain.fork.depth if chain.fork else 0,
                        "replay_eligible": replay_eligible,
                    }
                )

        subagent_meta.append(
            {
                "trace_id": trace_id,
                "subagent_index": sa_index,
                "outer_index": outer_idx,
                "agent_id": entry["agent_id"],
                "subagent_type": entry.get("subagent_type", ""),
                "status": entry.get("status", ""),
                "spawn_t_s": float(entry["t"]),
                "end_t_s": sa_end,
                "duration_s": duration_s,
                "raw_inner_requests": len(entry.get("requests", [])),
                "reconstructed_streams": len(stream_ids),
                "preceding_parent_turn": preceding,
                "following_parent_turn": join_turn,
                "is_background": replay_eligible and join_turn is None,
                "replay_eligible": replay_eligible,
                "models": "|".join(entry.get("models", [])),
                "stream_ids": stream_ids,
            }
        )

    topology_stats = {
        "top_seams": top_detection.seams_merged,
        "top_workers": len(top_detection.worker_indices),
        "top_aux": sum(1 for ci in top_detection.worker_indices if top_classes[ci][0] == "aux"),
        "top_worker_group_members": sum(
            1 for ci in top_detection.worker_indices if top_classes[ci][0] == "worker_group"
        ),
        "top_preamble_requests": len(top_preamble),
        "nested_seams": nested_seams,
        "nested_workers": nested_workers,
        "nested_aux": nested_aux,
        "nested_worker_group_members": nested_wg,
        "subagents": subagent_meta,
    }
    return streams, topology_stats


def _leading_seen(hash_ids: list[int], seen: set[int]) -> int:
    for idx, hid in enumerate(hash_ids):
        if hid not in seen:
            return idx
    return len(hash_ids)


def enrich_prefix_metrics(rows: list[dict[str, Any]], block_size: int) -> None:
    seen: set[int] = set()
    pending: list[tuple[float, int, list[int]]] = []
    completed_seen: set[int] = set()
    sequence = 0
    for row in sorted(
        (r for r in rows if r["replay_eligible"]),
        key=lambda r: (
            r["t_s"],
            r["source_outer_index"],
            r["chain_index"],
            r["turn_index"],
        ),
    ):
        while pending and pending[0][0] <= row["t_s"] + EPSILON_SECONDS:
            _, _, hashes = heapq.heappop(pending)
            completed_seen.update(hashes)
        hashes = row["_hash_ids"]
        available_blocks = _leading_seen(hashes, seen)
        completed_blocks = _leading_seen(hashes, completed_seen)
        total_blocks = len(hashes)
        row["available_prefix_blocks"] = available_blocks
        row["available_prefix_tokens"] = min(available_blocks * block_size, row["isl_tokens"])
        row["available_prefix_ratio"] = (
            available_blocks / total_blocks if total_blocks else 0.0
        )
        row["new_kv_blocks"] = max(total_blocks - available_blocks, 0)
        row["new_kv_tokens"] = max(
            row["isl_tokens"] - row["available_prefix_tokens"], 0
        )
        row["completed_prefix_blocks"] = completed_blocks
        row["completed_prefix_ratio"] = (
            completed_blocks / total_blocks if total_blocks else 0.0
        )
        seen.update(hashes)
        heapq.heappush(pending, (row["end_s"], sequence, hashes))
        sequence += 1


def process_trace(trace: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trace_id = trace["id"]
    block_size = int(trace["block_size"])
    streams, topology = make_stream_plans(trace)
    request_rows: list[dict[str, Any]] = []

    for stream in streams:
        prev_req: dict[str, Any] | None = None
        for turn_index, (source_idx, req) in enumerate(stream["requests"]):
            hashes = req.get("hash_ids", [])
            prev_hashes = prev_req.get("hash_ids", []) if prev_req else []
            common_blocks = lcp_blocks(prev_hashes, hashes) if prev_req else 0
            t_s = float(req["t"])
            api_s = max(0.0, finite_number(req.get("api_time")))
            end_s = t_s + api_s
            source_outer = (
                stream["source_outer_idx"]
                if stream["source_outer_idx"] is not None
                else source_idx
            )
            isl = int(req["in"])
            osl = int(req["out"])
            row = {
                "trace_id": trace_id,
                "stream_id": stream["stream_id"],
                "stream_kind": stream["stream_kind"],
                "explicit_subagent_id": stream["explicit_subagent_id"],
                "turn_index": turn_index,
                "source_outer_index": source_outer,
                "source_inner_index": source_idx if stream["source_outer_idx"] is not None else "",
                "chain_index": stream["chain_index"],
                "replay_eligible": bool(stream["replay_eligible"]),
                "model": req["model"],
                "transport_type": req["type"],
                "t_s": t_s,
                "end_s": end_s,
                "api_time_s": api_s,
                "think_time_s": finite_number(req.get("think_time"), math.nan),
                "ttft_s": finite_number(req.get("ttft"), math.nan),
                "isl_tokens": isl,
                "osl_tokens": osl,
                "context_tokens": isl + osl,
                "hash_blocks": len(hashes),
                "previous_lcp_blocks": common_blocks,
                "previous_lcp_tokens": min(common_blocks * block_size, isl),
                "previous_lcp_ratio": common_blocks / len(hashes) if hashes else 0.0,
                "isl_delta_tokens": isl - int(prev_req["in"]) if prev_req else 0,
                "context_growth_after_previous_tokens": (
                    isl - int(prev_req["in"]) - int(prev_req["out"])
                    if prev_req
                    else isl
                ),
                "start_to_start_gap_s": t_s - float(prev_req["t"]) if prev_req else math.nan,
                "dependent_idle_s": (
                    max(0.0, t_s - req_end(prev_req)) if prev_req else math.nan
                ),
                "fork_depth_blocks": stream["fork_depth_blocks"] if turn_index == 0 else 0,
                "_hash_ids": hashes,
            }
            request_rows.append(row)
            prev_req = req

    enrich_prefix_metrics(request_rows, block_size)
    for row in request_rows:
        if not row["replay_eligible"]:
            for key in (
                "available_prefix_blocks",
                "available_prefix_tokens",
                "available_prefix_ratio",
                "new_kv_blocks",
                "new_kv_tokens",
                "completed_prefix_blocks",
                "completed_prefix_ratio",
            ):
                row[key] = 0

    raw_requests: list[dict[str, Any]] = []
    explicit_subagent_requests = 0
    parent_requests = 0
    subagent_intervals: list[tuple[float, float]] = []
    trace_end = 0.0
    for outer_idx, entry in enumerate(trace["requests"]):
        if entry["type"] == "subagent":
            start = float(entry["t"])
            duration = finite_number(entry.get("duration_ms")) / 1000.0
            if entry.get("duration_ms") is None:
                inner_norm = [normalized_inner_request(entry, q) for q in entry.get("requests", [])]
                end = max((req_end(q) for q in inner_norm), default=start)
            else:
                end = start + max(duration, 0.0)
            subagent_intervals.append((start, end))
            trace_end = max(trace_end, end)
            for req in entry.get("requests", []):
                q = normalized_inner_request(entry, req)
                raw_requests.append(q)
                explicit_subagent_requests += 1
                trace_end = max(trace_end, req_end(q))
        else:
            raw_requests.append(entry)
            parent_requests += 1
            trace_end = max(trace_end, req_end(entry))
    starts = sorted(float(req["t"]) for req in raw_requests)
    arrival_gaps = [b - a for a, b in zip(starts, starts[1:])]
    intervals = [(float(req["t"]), req_end(req)) for req in raw_requests]
    busy_request_seconds = sum(max(0.0, end - start) for start, end in intervals)
    replay_rows = [r for r in request_rows if r["replay_eligible"]]
    total_isl = sum(r["isl_tokens"] for r in replay_rows)
    total_prefix = sum(r["available_prefix_tokens"] for r in replay_rows)
    main_parent_requests = sum(1 for r in replay_rows if r["stream_kind"] == "parent")
    reconstructed_agent_requests = sum(
        1
        for r in replay_rows
        if r["stream_kind"]
        in {"subagent", "nested_subagent", "flat_agent", "worker_group", "subagent_worker_group"}
    )
    aux_requests = sum(1 for r in replay_rows if "aux" in r["stream_kind"])

    max_requests_10s = 0
    window: deque[float] = deque()
    for start in starts:
        while window and window[0] < start - 10.0:
            window.popleft()
        window.append(start)
        max_requests_10s = max(max_requests_10s, len(window))

    subagent_meta = topology["subagents"]
    trace_row = {
        "trace_id": trace_id,
        "models": "|".join(trace.get("models", [])),
        "block_size": block_size,
        "hash_id_scope": trace["hash_id_scope"],
        "raw_requests": len(raw_requests),
        "parent_outer_requests": parent_requests,
        "explicit_subagent_requests": explicit_subagent_requests,
        "main_parent_requests": main_parent_requests,
        "reconstructed_agent_requests": reconstructed_agent_requests,
        "auxiliary_requests": aux_requests,
        "reconstructed_streams": sum(1 for s in streams if s["replay_eligible"]),
        "top_level_detected_workers": topology["top_workers"],
        "nested_detected_workers": topology["nested_workers"],
        "explicit_subagent_groups": len(subagent_meta),
        "active_subagent_groups": sum(1 for s in subagent_meta if s["replay_eligible"]),
        "background_subagent_groups": sum(1 for s in subagent_meta if s["is_background"]),
        "session_duration_s": trace_end,
        "max_context_tokens": max((r["context_tokens"] for r in request_rows), default=0),
        "max_isl_tokens": max((r["isl_tokens"] for r in request_rows), default=0),
        "total_input_tokens": sum(int(req["in"]) for req in raw_requests),
        "total_output_tokens": sum(int(req["out"]) for req in raw_requests),
        "theoretical_prefix_ratio_weighted": total_prefix / total_isl if total_isl else 0.0,
        "max_simultaneous_requests": active_interval_peak(intervals),
        "mean_inflight_requests_time_weighted": busy_request_seconds / trace_end if trace_end else 0.0,
        "max_simultaneous_subagent_groups": active_interval_peak(subagent_intervals),
        "median_arrival_gap_s": pct(arrival_gaps, 50),
        "p95_arrival_gap_s": pct(arrival_gaps, 95),
        "fraction_arrival_gaps_le_1s": (
            sum(g <= 1.0 for g in arrival_gaps) / len(arrival_gaps) if arrival_gaps else 0.0
        ),
        "fraction_arrival_gaps_gt_60s": (
            sum(g > 60.0 for g in arrival_gaps) / len(arrival_gaps) if arrival_gaps else 0.0
        ),
        "bursts_10s_gap": 1 + sum(g > 10.0 for g in arrival_gaps) if starts else 0,
        "max_requests_in_10s": max_requests_10s,
        "top_seams_merged": topology["top_seams"],
        "nested_seams_merged": topology["nested_seams"],
    }
    for threshold in CONTEXT_THRESHOLDS:
        trace_row[f"requests_ge_{threshold}"] = sum(
            r["context_tokens"] >= threshold for r in request_rows
        )

    topology_export = {
        "trace_id": trace_id,
        "streams": [
            {
                "stream_id": s["stream_id"],
                "stream_kind": s["stream_kind"],
                "turns": len(s["requests"]),
                "start_s": min((float(r[1]["t"]) for r in s["requests"]), default=0.0),
                "end_s": max((req_end(r[1]) for r in s["requests"]), default=0.0),
                "fork_depth_blocks": s["fork_depth_blocks"],
                "replay_eligible": s["replay_eligible"],
            }
            for s in streams
        ],
        "subagents": subagent_meta,
        "reconstruction": {k: v for k, v in topology.items() if k != "subagents"},
    }
    return trace_row, request_rows, subagent_meta, topology_export


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [key for key in rows[0] if not key.startswith("_")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b1116",
            "axes.facecolor": "#111a22",
            "axes.edgecolor": "#334454",
            "axes.labelcolor": "#d9e3ea",
            "xtick.color": "#9eb0bd",
            "ytick.color": "#9eb0bd",
            "text.color": "#e8f0f5",
            "grid.color": "#263642",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
        }
    )


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close()


def plot_distributions(requests: list[dict[str, Any]], traces: list[dict[str, Any]], plot_dir: Path) -> None:
    setup_plot_style()
    isl = np.array([r["isl_tokens"] for r in requests], dtype=float)
    osl = np.array([max(r["osl_tokens"], 1) for r in requests], dtype=float)
    contexts = np.array([r["context_tokens"] for r in requests], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins_isl = np.logspace(np.log10(max(isl.min(), 1)), np.log10(isl.max()), 70)
    bins_osl = np.logspace(0, np.log10(osl.max()), 70)
    axes[0].hist(isl, bins=bins_isl, color="#53d6c7", alpha=0.88)
    axes[0].set_xscale("log")
    axes[0].set_title("Input sequence length")
    axes[0].set_xlabel("ISL tokens (64-token block estimate)")
    axes[0].set_ylabel("Requests")
    axes[1].hist(osl, bins=bins_osl, color="#ffb45b", alpha=0.9)
    axes[1].set_xscale("log")
    axes[1].set_title("Output sequence length")
    axes[1].set_xlabel("OSL tokens")
    for ax in axes:
        ax.grid(True, alpha=0.45)
    savefig(plot_dir / "isl_osl_distributions.png")

    plt.figure(figsize=(8.6, 6.4))
    plt.hexbin(
        np.log10(np.maximum(isl, 1)),
        np.log10(np.maximum(osl, 1)),
        gridsize=58,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    cbar = plt.colorbar()
    cbar.set_label("log request count")
    plt.xlabel("log10 ISL tokens")
    plt.ylabel("log10 OSL tokens")
    plt.title("Joint ISL / OSL workload shape")
    plt.grid(True, alpha=0.25)
    savefig(plot_dir / "joint_isl_osl.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    trace_reqs = np.array([t["raw_requests"] for t in traces])
    durations = np.array([max(t["session_duration_s"], 1) for t in traces])
    max_ctx = np.array([t["max_context_tokens"] for t in traces])
    total_osl = np.array([t["total_output_tokens"] for t in traces])
    axes[0, 0].hist(trace_reqs, bins=50, color="#53d6c7")
    axes[0, 0].set_title("Requests per trace")
    axes[0, 1].hist(durations, bins=np.logspace(0, np.log10(durations.max()), 55), color="#8e8cf4")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Session duration")
    axes[0, 1].set_xlabel("seconds")
    axes[1, 0].hist(max_ctx, bins=np.logspace(np.log10(max_ctx.min()), np.log10(max_ctx.max()), 55), color="#ffb45b")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Maximum context per trace")
    axes[1, 0].set_xlabel("tokens")
    axes[1, 1].hist(total_osl, bins=np.logspace(np.log10(max(total_osl.min(), 1)), np.log10(total_osl.max()), 55), color="#f27187")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("Total generated tokens per trace")
    axes[1, 1].set_xlabel("tokens")
    for ax in axes.flat:
        ax.grid(True, alpha=0.4)
    savefig(plot_dir / "session_distributions.png")

    reuse = np.array([r["available_prefix_ratio"] for r in requests])
    new_tokens = np.array([max(r["new_kv_tokens"], 1) for r in requests])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].hist(reuse * 100, bins=50, color="#53d6c7")
    axes[0].set_title("Theoretical reusable prefix")
    axes[0].set_xlabel("% of request input")
    axes[1].hist(new_tokens, bins=np.logspace(0, np.log10(new_tokens.max()), 55), color="#f27187")
    axes[1].set_xscale("log")
    axes[1].set_title("New KV per request")
    axes[1].set_xlabel("tokens")
    turn_bins: dict[int, list[float]] = defaultdict(list)
    for row in requests:
        turn_bins[min(row["turn_index"], 50)].append(row["available_prefix_ratio"])
    xs = sorted(turn_bins)
    axes[2].plot(xs, [statistics.median(turn_bins[x]) * 100 for x in xs], color="#ffb45b", lw=2)
    axes[2].set_title("Reuse rises with stream turn")
    axes[2].set_xlabel("turn index (50 = 50+)")
    axes[2].set_ylabel("median reusable prefix %")
    for ax in axes:
        ax.grid(True, alpha=0.4)
    savefig(plot_dir / "prefix_reuse.png")

    think = np.array([r["think_time_s"] for r in requests if math.isfinite(r["think_time_s"])])
    dep = np.array([r["dependent_idle_s"] for r in requests if math.isfinite(r["dependent_idle_s"])])
    peak = np.array([t["max_simultaneous_requests"] for t in traces])
    burst = np.array([t["max_requests_in_10s"] for t in traces])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, vals, title, color in [
        (axes[0, 0], np.maximum(think, 1e-3), "Recorded think time", "#8e8cf4"),
        (axes[0, 1], np.maximum(dep, 1e-3), "Dependent end-to-start idle", "#53d6c7"),
    ]:
        ax.hist(vals, bins=np.logspace(-3, np.log10(max(vals.max(), 1)), 60), color=color)
        ax.set_xscale("log")
        ax.set_xlabel("seconds")
        ax.set_title(title)
    axes[1, 0].hist(peak, bins=np.arange(0.5, peak.max() + 1.5), color="#ffb45b")
    axes[1, 0].set_title("Peak simultaneous model requests / trace")
    axes[1, 1].hist(burst, bins=45, color="#f27187")
    axes[1, 1].set_title("Maximum arrivals in any 10-second window")
    for ax in axes.flat:
        ax.grid(True, alpha=0.4)
    savefig(plot_dir / "timing_behavior.png")


def plot_subagents(subagents: list[dict[str, Any]], traces: list[dict[str, Any]], plot_dir: Path) -> None:
    durations = np.array([max(s["duration_s"], 1e-3) for s in subagents])
    groups = np.array([t["explicit_subagent_groups"] for t in traces])
    peaks = np.array([t["max_simultaneous_subagent_groups"] for t in traces])
    fractions = np.array(
        [t["explicit_subagent_requests"] / t["raw_requests"] for t in traces]
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].hist(groups, bins=range(int(groups.max()) + 2), color="#53d6c7")
    axes[0, 0].set_title("Explicit subagent groups per trace")
    axes[0, 1].hist(durations, bins=np.logspace(-3, np.log10(durations.max()), 55), color="#8e8cf4")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Explicit subagent duration")
    axes[0, 1].set_xlabel("seconds")
    axes[1, 0].hist(peaks, bins=np.arange(-0.5, peaks.max() + 1.5), color="#ffb45b")
    axes[1, 0].set_title("Peak simultaneous subagent groups")
    axes[1, 1].hist(fractions * 100, bins=40, color="#f27187")
    axes[1, 1].set_title("Request share from explicit subagents")
    axes[1, 1].set_xlabel("% of trace requests")
    for ax in axes.flat:
        ax.grid(True, alpha=0.4)
    savefig(plot_dir / "subagent_concurrency.png")


def plot_timeline(
    trace_id: str,
    requests: list[dict[str, Any]],
    subagents: list[dict[str, Any]],
    plot_dir: Path,
) -> None:
    rows = [r for r in requests if r["trace_id"] == trace_id and r["replay_eligible"]]
    by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stream[row["stream_id"]].append(row)
    ranked = sorted(
        by_stream,
        key=lambda sid: (
            0 if sid == trace_id else 1,
            min(r["t_s"] for r in by_stream[sid]),
            sid,
        ),
    )[:24]
    fig_h = max(6.5, 0.42 * len(ranked) + 2.4)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    colors = {
        "parent": "#53d6c7",
        "subagent": "#8e8cf4",
        "nested_subagent": "#b18cf4",
        "flat_agent": "#ffb45b",
        "worker_group": "#f27187",
        "subagent_worker_group": "#f27187",
        "aux": "#7f91a2",
        "subagent_aux": "#7f91a2",
    }
    labels = []
    for y, sid in enumerate(ranked):
        stream_rows = by_stream[sid]
        kind = stream_rows[0]["stream_kind"]
        color = colors.get(kind, "#d9e3ea")
        for row in stream_rows:
            width = max(row["api_time_s"], 0.08)
            ax.broken_barh([(row["t_s"], width)], (y - 0.32, 0.64), facecolors=color)
        short = sid.replace(trace_id, "root")
        labels.append(f"{short} · {kind} ({len(stream_rows)})")
    for sa in subagents:
        if sa["trace_id"] == trace_id:
            ax.axvspan(sa["spawn_t_s"], sa["end_t_s"], color="#8e8cf4", alpha=0.035)
    ax.set_yticks(range(len(ranked)), labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("seconds from trace start")
    ax.set_title(f"Parent and subagent request timeline — {trace_id}")
    ax.grid(True, axis="x", alpha=0.4)
    savefig(plot_dir / "representative_complex_timeline.png")


def nearest_unique(traces: list[dict[str, Any]], field: str, target: float, used: set[str]) -> dict[str, Any]:
    choices = [t for t in traces if t["trace_id"] not in used]
    return min(choices, key=lambda t: abs(float(t[field]) - target))


def choose_representatives(traces: list[dict[str, Any]]) -> list[dict[str, str]]:
    used: set[str] = set()
    selected: list[dict[str, str]] = []

    def add(row: dict[str, Any], reason: str) -> None:
        if row["trace_id"] in used:
            return
        used.add(row["trace_id"])
        selected.append({"trace_id": row["trace_id"], "reason": reason})

    ctx = [t["max_context_tokens"] for t in traces]
    add(nearest_unique(traces, "max_context_tokens", pct(ctx, 10), used), "short-context representative near the corpus p10 maximum context")
    add(nearest_unique(traces, "max_context_tokens", pct(ctx, 50), used), "medium-context representative near the corpus median maximum context")
    add(nearest_unique(traces, "max_context_tokens", pct(ctx, 90), used), "long-context representative near the corpus p90 maximum context")
    add(max(traces, key=lambda t: t["max_context_tokens"]), "largest observed context")
    add(max(traces, key=lambda t: t["raw_requests"]), "most model requests")
    add(max(traces, key=lambda t: (t["max_simultaneous_subagent_groups"], t["explicit_subagent_groups"])), "strongest explicit subagent fan-out")
    add(max(traces, key=lambda t: t["max_simultaneous_requests"]), "highest recorded simultaneous model-request concurrency")
    add(max(traces, key=lambda t: t["session_duration_s"]), "longest wall-clock session")
    add(max(traces, key=lambda t: t["theoretical_prefix_ratio_weighted"]), "highest weighted theoretical prefix reuse")
    add(min(traces, key=lambda t: t["theoretical_prefix_ratio_weighted"]), "lowest weighted theoretical prefix reuse / strongest context discontinuity")
    return selected[:10]


def binned_reuse(requests: list[dict[str, Any]], field: str, edges: list[float]) -> list[dict[str, Any]]:
    out = []
    for low, high in zip(edges, edges[1:]):
        vals = [r["available_prefix_ratio"] for r in requests if low <= r[field] < high]
        if vals:
            out.append(
                {
                    "low": low,
                    "high": high,
                    "n": len(vals),
                    "mean": statistics.fmean(vals),
                    "p50": statistics.median(vals),
                }
            )
    return out


def build_summary(
    traces: list[dict[str, Any]], requests: list[dict[str, Any]], subagents: list[dict[str, Any]], representatives: list[dict[str, str]]
) -> dict[str, Any]:
    replay = [r for r in requests if r["replay_eligible"]]
    total_isl = sum(r["isl_tokens"] for r in replay)
    total_prefix = sum(r["available_prefix_tokens"] for r in replay)
    thresholds = {
        str(threshold): {
            "requests": sum(r["context_tokens"] >= threshold for r in requests),
            "fraction": sum(r["context_tokens"] >= threshold for r in requests) / len(requests),
            "sessions": sum(t[f"requests_ge_{threshold}"] > 0 for t in traces),
            "session_fraction": sum(t[f"requests_ge_{threshold}"] > 0 for t in traces) / len(traces),
        }
        for threshold in CONTEXT_THRESHOLDS
    }
    arrival_gaps = []
    for trace_id in {r["trace_id"] for r in requests}:
        starts = sorted(r["t_s"] for r in requests if r["trace_id"] == trace_id)
        arrival_gaps.extend(b - a for a, b in zip(starts, starts[1:]))
    dependent = [r["dependent_idle_s"] for r in replay if math.isfinite(r["dependent_idle_s"])]
    think = [r["think_time_s"] for r in requests if math.isfinite(r["think_time_s"])]
    duration_mean = statistics.fmean(t["session_duration_s"] for t in traces)
    overlap_scenarios = [
        {
            "session_starts_per_minute": rate,
            "expected_active_sessions": rate / 60.0 * duration_mean,
        }
        for rate in [0.1, 0.5, 1, 2, 5, 10]
    ]
    model_counts = Counter(r["model"] for r in requests)
    stream_kind_counts = Counter(r["stream_kind"] for r in replay)
    summary = {
        "provenance": {
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "aiperf_repo": AIPERF_REPO,
            "aiperf_revision": AIPERF_REVISION,
            "analysis_semantics": "AIPerf Weka hash-LCP and global-time shared seen-set",
        },
        "corpus": {
            "traces": len(traces),
            "requests": len(requests),
            "parent_outer_requests": sum(t["parent_outer_requests"] for t in traces),
            "explicit_subagent_requests": sum(t["explicit_subagent_requests"] for t in traces),
            "explicit_subagent_groups": len(subagents),
            "sessions_with_subagents": sum(t["explicit_subagent_groups"] > 0 for t in traces),
            "fraction_sessions_with_subagents": sum(t["explicit_subagent_groups"] > 0 for t in traces) / len(traces),
            "reconstructed_streams": sum(t["reconstructed_streams"] for t in traces),
            "top_level_detected_workers": sum(t["top_level_detected_workers"] for t in traces),
            "nested_detected_workers": sum(t["nested_detected_workers"] for t in traces),
            "models": dict(model_counts.most_common()),
            "stream_kinds": dict(stream_kind_counts),
        },
        "distributions": {
            "requests_per_trace": distribution(t["raw_requests"] for t in traces),
            "session_duration_s": distribution(t["session_duration_s"] for t in traces),
            "subagent_groups_per_trace": distribution(t["explicit_subagent_groups"] for t in traces),
            "isl_tokens": distribution(r["isl_tokens"] for r in requests),
            "osl_tokens": distribution(r["osl_tokens"] for r in requests),
            "context_tokens": distribution(r["context_tokens"] for r in requests),
            "max_context_per_trace": distribution(t["max_context_tokens"] for t in traces),
            "parent_turns_per_trace": distribution(t["main_parent_requests"] for t in traces),
            "turns_per_stream": distribution(
                Counter(r["stream_id"] for r in replay).values()
            ),
            "requests_per_explicit_subagent": distribution(
                s["raw_inner_requests"] for s in subagents
            ),
            "isl_delta_tokens": distribution(
                r["isl_delta_tokens"] for r in replay if r["turn_index"] > 0
            ),
            "context_growth_after_previous_tokens": distribution(
                r["context_growth_after_previous_tokens"]
                for r in replay
                if r["turn_index"] > 0
            ),
            "new_kv_tokens": distribution(r["new_kv_tokens"] for r in replay),
            "total_input_per_trace": distribution(t["total_input_tokens"] for t in traces),
            "total_output_per_trace": distribution(t["total_output_tokens"] for t in traces),
            "prefix_reuse_ratio": distribution(r["available_prefix_ratio"] for r in replay),
            "consecutive_lcp_ratio": distribution(r["previous_lcp_ratio"] for r in replay),
            "dependent_idle_s": distribution(dependent),
            "think_time_s": distribution(think),
            "arrival_gap_s": distribution(arrival_gaps),
            "max_simultaneous_requests": distribution(t["max_simultaneous_requests"] for t in traces),
            "mean_inflight_requests_time_weighted": distribution(
                t["mean_inflight_requests_time_weighted"] for t in traces
            ),
            "max_requests_in_10s": distribution(t["max_requests_in_10s"] for t in traces),
            "max_simultaneous_subagent_groups": distribution(t["max_simultaneous_subagent_groups"] for t in traces),
            "subagent_duration_s": distribution(s["duration_s"] for s in subagents),
        },
        "context_thresholds": thresholds,
        "prefix": {
            "aggregate_theoretical_reuse_ratio": total_prefix / total_isl if total_isl else 0.0,
            "aggregate_new_prefill_tokens": total_isl - total_prefix,
            "aggregate_input_tokens": total_isl,
            "completed_before_arrival_reuse_ratio": (
                sum(r["completed_prefix_blocks"] for r in replay)
                / sum(r["hash_blocks"] for r in replay)
                if replay
                else 0.0
            ),
            "reuse_by_turn": binned_reuse(replay, "turn_index", [0, 1, 2, 4, 8, 16, 32, 64, 128, 10_000]),
            "reuse_by_context": binned_reuse(
                replay,
                "isl_tokens",
                [0, 32768, 65536, 131072, 262144, 524288, 1_000_000, 2_000_000],
            ),
        },
        "timing": {
            "fraction_arrival_gaps_le_1s": sum(g <= 1 for g in arrival_gaps) / len(arrival_gaps),
            "fraction_arrival_gaps_le_10s": sum(g <= 10 for g in arrival_gaps) / len(arrival_gaps),
            "fraction_arrival_gaps_gt_60s": sum(g > 60 for g in arrival_gaps) / len(arrival_gaps),
            "fraction_arrival_gaps_gt_600s": sum(g > 600 for g in arrival_gaps) / len(arrival_gaps),
            "overlap_scenarios": overlap_scenarios,
            "cross_session_overlap_note": "Trace timestamps are relative to each session; corpus-global wall-clock start times are absent. Actual historical cross-session overlap is not identifiable.",
        },
        "subagents": {
            "fraction_requests_from_explicit_subagents": sum(t["explicit_subagent_requests"] for t in traces) / len(requests),
            "background_groups": sum(t["background_subagent_groups"] for t in traces),
            "orphaned_groups": sum(not s["replay_eligible"] for s in subagents),
            "multi_stream_entries": sum(s["reconstructed_streams"] > 1 for s in subagents),
        },
        "representative_traces": representatives,
    }
    return summary


def turn_example(trace_id: str, requests: list[dict[str, Any]], max_rows: int = 12) -> str:
    parent = sorted(
        [r for r in requests if r["trace_id"] == trace_id and r["stream_kind"] == "parent"],
        key=lambda r: r["turn_index"],
    )
    shown = parent[:max_rows]
    lines = [f"`{trace_id}` — {len(parent)} reconstructed parent turns"]
    for row in shown:
        lines.append(
            f"- turn {row['turn_index'] + 1}: {fmt_num(row['isl_tokens'])} ISL → "
            f"{fmt_num(row['osl_tokens'])} OSL; {row['available_prefix_ratio'] * 100:.1f}% theoretically reusable"
        )
    if len(parent) > max_rows:
        lines.append(f"- … {len(parent) - max_rows} additional parent turns")
    return "\n".join(lines)


def generate_report(
    output_path: Path,
    summary: dict[str, Any],
    traces: list[dict[str, Any]],
    requests: list[dict[str, Any]],
) -> None:
    d = summary["distributions"]
    corpus = summary["corpus"]
    prefix = summary["prefix"]
    timing = summary["timing"]
    sub = summary["subagents"]
    trace_by_id = {t["trace_id"]: t for t in traces}
    representatives = summary["representative_traces"]
    short_med_long = representatives[:3]

    lines = [
        "# SemiAnalysis AgentX Claude Code trace analysis",
        "",
        f"**Corpus:** `{DATASET_REPO}` at revision `{DATASET_REVISION}`  ",
        f"**Weka semantics:** AIPerf `{AIPERF_REVISION}`  ",
        "**Generated by:** `analyze_agentx.py`",
        "",
        "> `hash_ids` quantify block-granular prefix reuse that is theoretically available. "
        "They do **not** report an actual runtime KV-cache hit rate; retention, eviction, routing, "
        "and finite cache capacity are not observed.",
        "",
        "## Executive summary",
        "",
        f"The corpus contains **{corpus['traces']:,} Claude Code sessions** and "
        f"**{corpus['requests']:,} model requests**. A session has a median of "
        f"**{d['requests_per_trace']['p50']:.0f} requests** and lasts **{fmt_duration(d['session_duration_s']['p50'])}** "
        f"at the median. Explicit subagents produce **{sub['fraction_requests_from_explicit_subagents'] * 100:.1f}%** "
        f"of all model requests; **{corpus['fraction_sessions_with_subagents'] * 100:.1f}%** of sessions spawn at least one.",
        "",
        f"The median request is **{fmt_num(d['isl_tokens']['p50'])} input / {fmt_num(d['osl_tokens']['p50'])} output tokens**. "
        f"The request-level p99 input is **{fmt_num(d['isl_tokens']['p99'])}**, while the largest observed "
        f"input+requested-output context is **{fmt_num(d['context_tokens']['max'])}**. The aggregate Weka-theoretical "
        f"reusable prefix is **{prefix['aggregate_theoretical_reuse_ratio'] * 100:.1f}%** of input tokens, leaving "
        f"**{fmt_num(prefix['aggregate_new_prefill_tokens'])}** block-estimated tokens as new prefill across the corpus.",
        "",
        "## 1. Corpus overview",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| traces / sessions | {corpus['traces']:,} |",
        f"| model requests | {corpus['requests']:,} |",
        f"| outer parent-list requests | {corpus['parent_outer_requests']:,} |",
        f"| requests nested in explicit subagents | {corpus['explicit_subagent_requests']:,} |",
        f"| explicit subagent groups | {corpus['explicit_subagent_groups']:,} |",
        f"| sessions with explicit subagents | {corpus['sessions_with_subagents']:,} ({corpus['fraction_sessions_with_subagents'] * 100:.1f}%) |",
        f"| reconstructed logical streams | {corpus['reconstructed_streams']:,} |",
        f"| detected top-level worker streams | {corpus['top_level_detected_workers']:,} |",
        f"| detected nested worker streams | {corpus['nested_detected_workers']:,} |",
        "",
        "The raw outer list should not be read as one serial parent loop. AIPerf's hash-LCP detector "
        "finds independent flattened workers at the top level and additional child streams inside explicit "
        "subagent blocks. Counts above therefore separate raw storage shape from replay topology.",
        "",
        "### Session distributions",
        "",
        "| metric | p10 | p50 | p90 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| requests/session | {d['requests_per_trace']['p10']:.0f} | {d['requests_per_trace']['p50']:.0f} | {d['requests_per_trace']['p90']:.0f} | {d['requests_per_trace']['p95']:.0f} | {d['requests_per_trace']['p99']:.0f} | {d['requests_per_trace']['max']:.0f} |",
        f"| duration | {fmt_duration(d['session_duration_s']['p10'])} | {fmt_duration(d['session_duration_s']['p50'])} | {fmt_duration(d['session_duration_s']['p90'])} | {fmt_duration(d['session_duration_s']['p95'])} | {fmt_duration(d['session_duration_s']['p99'])} | {fmt_duration(d['session_duration_s']['max'])} |",
        f"| explicit subagents/session | {d['subagent_groups_per_trace']['p10']:.0f} | {d['subagent_groups_per_trace']['p50']:.0f} | {d['subagent_groups_per_trace']['p90']:.0f} | {d['subagent_groups_per_trace']['p95']:.0f} | {d['subagent_groups_per_trace']['p99']:.0f} | {d['subagent_groups_per_trace']['max']:.0f} |",
        "",
        "![Session distributions](agentx_outputs/plots/session_distributions.png)",
        "",
        "## 2. ISL / OSL characteristics",
        "",
        "| tokens/request | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| ISL | {fmt_num(d['isl_tokens']['p10'])} | {fmt_num(d['isl_tokens']['p25'])} | {fmt_num(d['isl_tokens']['p50'])} | {fmt_num(d['isl_tokens']['p75'])} | {fmt_num(d['isl_tokens']['p90'])} | {fmt_num(d['isl_tokens']['p95'])} | {fmt_num(d['isl_tokens']['p99'])} | {fmt_num(d['isl_tokens']['max'])} |",
        f"| OSL | {fmt_num(d['osl_tokens']['p10'])} | {fmt_num(d['osl_tokens']['p25'])} | {fmt_num(d['osl_tokens']['p50'])} | {fmt_num(d['osl_tokens']['p75'])} | {fmt_num(d['osl_tokens']['p90'])} | {fmt_num(d['osl_tokens']['p95'])} | {fmt_num(d['osl_tokens']['p99'])} | {fmt_num(d['osl_tokens']['max'])} |",
        f"| context (ISL+OSL) | {fmt_num(d['context_tokens']['p10'])} | {fmt_num(d['context_tokens']['p25'])} | {fmt_num(d['context_tokens']['p50'])} | {fmt_num(d['context_tokens']['p75'])} | {fmt_num(d['context_tokens']['p90'])} | {fmt_num(d['context_tokens']['p95'])} | {fmt_num(d['context_tokens']['p99'])} | {fmt_num(d['context_tokens']['max'])} |",
        "",
        "### Context-window exceedance",
        "",
        "| threshold | requests | request fraction | sessions containing ≥1 | session fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for threshold in CONTEXT_THRESHOLDS:
        item = summary["context_thresholds"][str(threshold)]
        lines.append(
            f"| {fmt_num(threshold)} | {item['requests']:,} | {item['fraction'] * 100:.2f}% | "
            f"{item['sessions']:,} | {item['session_fraction'] * 100:.1f}% |"
        )
    lines += [
        "",
        "`in` is a 64-token-block estimate, not an exact tokenizer count. The source corpus already removes "
        "rows above 990,016 input tokens because heavy cache-write telemetry can overcount the real prompt.",
        "",
        "![ISL and OSL distributions](agentx_outputs/plots/isl_osl_distributions.png)",
        "",
        "![Joint ISL/OSL distribution](agentx_outputs/plots/joint_isl_osl.png)",
        "",
        "## 3. Multi-turn context growth",
        "",
        f"A reconstructed stream has **{d['turns_per_stream']['p50']:.0f} turns at p50**, "
        f"**{d['turns_per_stream']['p90']:.0f} at p90**, and **{d['turns_per_stream']['max']:.0f} at maximum**. "
        f"The median ISL change after turn 1 is **{fmt_num(d['isl_delta_tokens']['p50'])} tokens**; "
        f"the median newly introduced KV is **{fmt_num(d['new_kv_tokens']['p50'])} tokens**. Negative ISL "
        "deltas are real context shrinks/compactions and are kept rather than forced into append-only arithmetic.",
        "",
        "| session/stream metric | p50 | p90 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|",
        f"| reconstructed parent turns/session | {d['parent_turns_per_trace']['p50']:.0f} | {d['parent_turns_per_trace']['p90']:.0f} | {d['parent_turns_per_trace']['p95']:.0f} | {d['parent_turns_per_trace']['p99']:.0f} | {d['parent_turns_per_trace']['max']:.0f} |",
        f"| turns/logical stream | {d['turns_per_stream']['p50']:.0f} | {d['turns_per_stream']['p90']:.0f} | {d['turns_per_stream']['p95']:.0f} | {d['turns_per_stream']['p99']:.0f} | {d['turns_per_stream']['max']:.0f} |",
        f"| max context/session | {fmt_num(d['max_context_per_trace']['p50'])} | {fmt_num(d['max_context_per_trace']['p90'])} | {fmt_num(d['max_context_per_trace']['p95'])} | {fmt_num(d['max_context_per_trace']['p99'])} | {fmt_num(d['max_context_per_trace']['max'])} |",
        f"| generated tokens/session | {fmt_num(d['total_output_per_trace']['p50'])} | {fmt_num(d['total_output_per_trace']['p90'])} | {fmt_num(d['total_output_per_trace']['p95'])} | {fmt_num(d['total_output_per_trace']['p99'])} | {fmt_num(d['total_output_per_trace']['max'])} |",
        f"| requests/explicit subagent | {d['requests_per_explicit_subagent']['p50']:.0f} | {d['requests_per_explicit_subagent']['p90']:.0f} | {d['requests_per_explicit_subagent']['p95']:.0f} | {d['requests_per_explicit_subagent']['p99']:.0f} | {d['requests_per_explicit_subagent']['max']:.0f} |",
        "",
        "### Short, medium, and long-context examples",
        "",
    ]
    for rep in short_med_long:
        lines += [turn_example(rep["trace_id"], requests), ""]
    lines += [
        "## 4. Prefix / KV reuse",
        "",
        f"Across all replay-eligible streams, **{prefix['aggregate_theoretical_reuse_ratio'] * 100:.1f}%** of "
        "block-estimated input tokens are a leading prefix whose block IDs appeared earlier in the same trace. "
        f"Request-level reuse is **{d['prefix_reuse_ratio']['p50'] * 100:.1f}% at p50** and "
        f"**{d['prefix_reuse_ratio']['p90'] * 100:.1f}% at p90**. The median request introduces "
        f"**{fmt_num(d['new_kv_tokens']['p50'])} new KV tokens**, p90 **{fmt_num(d['new_kv_tokens']['p90'])}**.",
        "",
        f"Against only the previous request on the same reconstructed stream, the common-prefix ratio is "
        f"**{d['consecutive_lcp_ratio']['p50'] * 100:.1f}% at p50** and "
        f"**{d['consecutive_lcp_ratio']['p90'] * 100:.1f}% at p90**. The global-time metric can be higher "
        "because it also credits blocks introduced by parent or sibling branches.",
        "",
        "The cross-branch metric follows AIPerf exactly: within a trace-local hash namespace, requests are "
        "consumed in deterministic global timestamp order and the leading run already present in a shared seen-set "
        "is reusable. Consecutive-stream LCP is also exported separately. This is a theoretical opportunity, "
        "not proof that a runtime retained or routed those blocks to the same cache.",
        "",
        "### Reuse by stream age",
        "",
        "| stream turn range | requests | median reuse | mean reuse |",
        "|---|---:|---:|---:|",
    ]
    for item in prefix["reuse_by_turn"]:
        high = "+" if item["high"] >= 10_000 else f"–{int(item['high']) - 1}"
        lines.append(
            f"| {int(item['low'])}{high} | {item['n']:,} | {item['p50'] * 100:.1f}% | {item['mean'] * 100:.1f}% |"
        )
    lines += [
        "",
        "### Reuse by input-context range",
        "",
        "| ISL range | requests | median reuse | mean reuse |",
        "|---|---:|---:|---:|",
    ]
    for item in prefix["reuse_by_context"]:
        lines.append(
            f"| {fmt_num(item['low'])}–{fmt_num(item['high'])} | {item['n']:,} | "
            f"{item['p50'] * 100:.1f}% | {item['mean'] * 100:.1f}% |"
        )
    lines += [
        "",
        "![Prefix reuse](agentx_outputs/plots/prefix_reuse.png)",
        "",
        "## 5. Timing / agent behavior",
        "",
        f"The dependent end-to-start idle gap is **{d['dependent_idle_s']['p50']:.2f}s at p50**, "
        f"**{d['dependent_idle_s']['p90']:.1f}s at p90**, and **{fmt_duration(d['dependent_idle_s']['p99'])} at p99**. "
        f"Recorded think time is **{d['think_time_s']['p50']:.2f}s at p50** and **{d['think_time_s']['p90']:.1f}s at p90**. "
        f"Across each trace's global arrival stream, {timing['fraction_arrival_gaps_le_1s'] * 100:.1f}% of gaps are ≤1s, "
        f"{timing['fraction_arrival_gaps_le_10s'] * 100:.1f}% are ≤10s, and {timing['fraction_arrival_gaps_gt_60s'] * 100:.1f}% exceed 60s.",
        "",
        f"Within a 10-second sliding window, a session produces **{d['max_requests_in_10s']['p50']:.0f} arrivals at p50**, "
        f"**{d['max_requests_in_10s']['p90']:.0f} at p90**, and **{d['max_requests_in_10s']['max']:.0f} maximum**. "
        "This combines tight subagent bursts with long waiting tails rather than resembling a stationary Poisson stream.",
        "",
        "The corpus has only session-relative timestamps. It cannot reveal how the 393 historical sessions "
        "overlapped in absolute wall clock. Using the measured duration distribution, Little's Law gives the "
        "following workload-planning scenarios:",
        "",
        "| new sessions/min | expected concurrently active sessions |",
        "|---:|---:|",
    ]
    for item in timing["overlap_scenarios"]:
        lines.append(
            f"| {item['session_starts_per_minute']:.1f} | {item['expected_active_sessions']:.1f} |"
        )
    lines += [
        "",
        "![Timing behavior](agentx_outputs/plots/timing_behavior.png)",
        "",
        "## 6. Subagent concurrency",
        "",
        f"Explicit subagents account for **{sub['fraction_requests_from_explicit_subagents'] * 100:.1f}%** of requests. "
        f"Their median duration is **{fmt_duration(d['subagent_duration_s']['p50'])}**, p90 "
        f"**{fmt_duration(d['subagent_duration_s']['p90'])}**, and p99 **{fmt_duration(d['subagent_duration_s']['p99'])}**. "
        f"Peak simultaneous explicit subagent groups are **{d['max_simultaneous_subagent_groups']['p50']:.0f} at p50**, "
        f"**{d['max_simultaneous_subagent_groups']['p90']:.0f} at p90**, and "
        f"**{d['max_simultaneous_subagent_groups']['max']:.0f} maximum**. Recorded model-request concurrency "
        f"reaches **{d['max_simultaneous_requests']['p50']:.0f} at p50** and "
        f"**{d['max_simultaneous_requests']['max']:.0f} maximum** per session.",
        "",
        f"AIPerf reconstructs {sub['multi_stream_entries']:,} explicit subagent entries as more than one inner "
        f"context stream. It identifies {sub['background_groups']:,} background branches and "
        f"{sub['orphaned_groups']:,} orphaned groups with no preceding parent anchor.",
        "",
        "![Subagent concurrency](agentx_outputs/plots/subagent_concurrency.png)",
        "",
        "### Representative complex timeline",
        "",
        "Each bar is one recorded model request interval. Shaded spans mark explicit subagent lifetimes; "
        "rows are AIPerf-reconstructed context streams.",
        "",
        "![Representative parent/subagent timeline](agentx_outputs/plots/representative_complex_timeline.png)",
        "",
        "## 7. Implications for serving",
        "",
        "- **Prefix caching:** The workload offers substantial theoretical leading-prefix reuse, including across "
        "  parent/subagent branches. Realized hit rate will depend on cache residency, routing affinity, eviction, "
        "  block size, and whether concurrent branches reach the same cache shard.",
        "- **KV memory:** Long-lived sessions combine high ISL with many simultaneous branches. Capacity planning "
        "  should use the context-tail percentiles and concurrent active streams, not only median ISL.",
        "- **Prefill vs decode:** The measured ISL/OSL asymmetry is strongly prefill-heavy per request, but high prefix "
        "  reuse can convert much of that apparent prefill into cache lookup/reuse. New-KV tokens are the more useful "
        "  prefill demand signal.",
        "- **Long-context attention:** A material tail crosses 128K and 256K. Experiments should preserve that tail "
        "  rather than truncate all traffic to a single context bucket.",
        "- **Batching/concurrency:** One user session can generate overlapping parent, worker, and subagent requests; "
        "  a concurrency limit on session trees is not a request-concurrency limit.",
        "- **P/D disaggregation:** Bursty cached-prefill arrivals, long decode-light loops, and idle think periods make "
        "  phase-specific queues plausible, but any experiment should replay the recorded timing and branch topology.",
        "- **Speculative decoding:** These traces provide OSL, arrival, and concurrency shape only. They contain no "
        "  target/draft token agreement and therefore cannot supply acceptance rates. Inject separately calibrated, "
        "  model-specific acceptance behavior (for example SPEED-Bench) into this traffic shape.",
        "",
        "## Representative trace IDs",
        "",
        "| trace ID | why it is useful | requests | duration | max context | subagents | peak request concurrency |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for rep in representatives:
        row = trace_by_id[rep["trace_id"]]
        lines.append(
            f"| `{rep['trace_id']}` | {rep['reason']} | {row['raw_requests']:,} | "
            f"{fmt_duration(row['session_duration_s'])} | {fmt_num(row['max_context_tokens'])} | "
            f"{row['explicit_subagent_groups']} | {row['max_simultaneous_requests']} |"
        )
    lines += [
        "",
        "## Reproducibility and limitations",
        "",
        "Run `python analyze_agentx.py` after downloading the pinned dataset revision. Outputs are "
        "`agentx_outputs/per_trace_metrics.csv`, `per_request_metrics.csv`, `subagent_metrics.csv`, "
        "`topology.json`, `summary.json`, and `plots/*.png`.",
        "",
        "Important limitations:",
        "",
        "- `in` is `len(hash_ids) × 64`, not an exact model-tokenizer count. The dataset publisher documents "
        "  overcount in the heavy cache-write tail.",
        "- Hash scope is local to one trace. Cross-session prompt sharing is neither represented nor credited.",
        "- Prefix reuse is theoretical availability, not an observed runtime hit rate.",
        "- Session timestamps are relative, so historical cross-session overlap cannot be reconstructed.",
        "- Explicit SPAWN/JOIN topology and hidden stream reconstruction follow the inspected AIPerf commit; later "
        "  AIPerf versions may change classification defaults.",
        "- No speculative-decoding acceptance-rate inference is made.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the SemiAnalysis AgentX Claude Code WekaTrace corpus and "
            "write reusable metrics, plots, topology data, and a Markdown report."
        )
    )
    parser.add_argument("--input", type=Path, default=Path("agentx_data/traces.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("agentx_outputs"))
    parser.add_argument("--report", type=Path, default=Path("AGENTX_TRACE_ANALYSIS.md"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    traces: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    subagents: list[dict[str, Any]] = []
    topologies: list[dict[str, Any]] = []
    with args.input.open("rb") as handle:
        for index, line in enumerate(handle, 1):
            trace = orjson.loads(line)
            trace_row, request_rows, subagent_rows, topology = process_trace(trace)
            traces.append(trace_row)
            requests.extend(request_rows)
            subagents.extend(subagent_rows)
            topologies.append(topology)
            if index % 25 == 0:
                print(f"processed {index} traces", flush=True)

    for row in requests:
        row.pop("_hash_ids", None)
    representatives = choose_representatives(traces)
    summary = build_summary(traces, requests, subagents, representatives)

    write_csv(args.output / "per_trace_metrics.csv", traces)
    write_csv(args.output / "per_request_metrics.csv", requests)
    write_csv(args.output / "subagent_metrics.csv", subagents)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "topology.json").write_text(json.dumps(topologies, indent=1), encoding="utf-8")
    (args.output / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "dataset_repo": DATASET_REPO,
                "dataset_revision": DATASET_REVISION,
                "aiperf_repo": AIPERF_REPO,
                "aiperf_revision": AIPERF_REVISION,
                "input_file": str(args.input),
                "input_size_bytes": args.input.stat().st_size,
                "trace_count": len(traces),
                "request_count": len(requests),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_distributions(requests, traces, plot_dir)
    plot_subagents(subagents, traces, plot_dir)
    complex_trace = max(
        traces,
        key=lambda t: (
            t["max_simultaneous_subagent_groups"],
            t["max_simultaneous_requests"],
            t["explicit_subagent_groups"],
        ),
    )["trace_id"]
    plot_timeline(complex_trace, requests, subagents, plot_dir)
    summary["timeline_trace_id"] = complex_trace
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generate_report(args.report, summary, traces, requests)
    print(
        f"wrote {len(traces)} traces / {len(requests)} requests to {args.output}; "
        f"report={args.report}; timeline={complex_trace}"
    )


if __name__ == "__main__":
    main()
