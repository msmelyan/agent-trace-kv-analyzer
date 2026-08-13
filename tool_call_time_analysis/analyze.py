#!/usr/bin/env python3
"""Reproduce aggregate TraceLab and SWE-chat tool-call timing tables.

Only aggregate counts and percentiles are written. Transcript content, commands,
file paths, identifiers, and per-call examples are never exported.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import duckdb


BUCKET_SQL = """
CASE
  WHEN {value} < 10 THEN '<10 ms'
  WHEN {value} < 25 THEN '10-25 ms'
  WHEN {value} < 50 THEN '25-50 ms'
  WHEN {value} < 100 THEN '50-100 ms'
  WHEN {value} < 250 THEN '100-250 ms'
  WHEN {value} < 500 THEN '250-500 ms'
  WHEN {value} < 1000 THEN '500 ms-1 s'
  WHEN {value} < 10000 THEN '1-10 s'
  WHEN {value} < 60000 THEN '10 s-1 min'
  WHEN {value} < 600000 THEN '1-10 min'
  WHEN {value} < 3600000 THEN '10 min-1 h'
  ELSE '>=1 h'
END
"""

BUCKET_ORDER_SQL = """
CASE bucket
  WHEN '<10 ms' THEN 1 WHEN '10-25 ms' THEN 2
  WHEN '25-50 ms' THEN 3 WHEN '50-100 ms' THEN 4
  WHEN '100-250 ms' THEN 5 WHEN '250-500 ms' THEN 6
  WHEN '500 ms-1 s' THEN 7 WHEN '1-10 s' THEN 8
  WHEN '10 s-1 min' THEN 9 WHEN '1-10 min' THEN 10
  WHEN '10 min-1 h' THEN 11 ELSE 12
END
"""


def write_query(con: duckdb.DuckDBPyConnection, path: Path, query: str) -> None:
    cursor = con.execute(query)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([column[0] for column in cursor.description])
        writer.writerows(cursor.fetchall())


def analyze_tracelab(db: Path, output: Path) -> None:
    con = duckdb.connect(str(db), read_only=True)
    effective = "coalesce(tc.tool_internal_latency_ms, tc.tool_wall_latency_ms)"
    bucket = BUCKET_SQL.format(value="latency_ms")
    write_query(
        con,
        output / "tracelab_effective_latency_by_provider.csv",
        f"""
        WITH base AS (
          SELECT r.provider, {effective} AS latency_ms
          FROM tool_calls tc JOIN rounds r USING (round_pk)
          WHERE {effective} > 0 AND r.provider IN ('claude', 'codex')
        ), bucketed AS (
          SELECT provider, {bucket} AS bucket FROM base
        ), agg AS (
          SELECT provider, bucket, count(*) AS calls
          FROM bucketed GROUP BY provider, bucket
        )
        SELECT provider, bucket, calls,
               sum(calls) OVER (PARTITION BY provider) AS total_calls,
               round(100.0 * calls / sum(calls) OVER (PARTITION BY provider), 3) AS call_pct
        FROM agg ORDER BY {BUCKET_ORDER_SQL}, provider
        """,
    )

    internal_bucket = BUCKET_SQL.format(value="internal_ms")
    write_query(
        con,
        output / "tracelab_codex_overhead_by_internal_bucket.csv",
        f"""
        WITH paired AS (
          SELECT tc.tool_internal_latency_ms AS internal_ms,
                 tc.tool_wall_latency_ms AS wall_ms,
                 greatest(tc.tool_wall_latency_ms - tc.tool_internal_latency_ms, 0) AS overhead_ms
          FROM tool_calls tc JOIN rounds r USING (round_pk)
          WHERE r.provider = 'codex'
            AND tc.tool_wall_latency_ms IS NOT NULL
            AND tc.tool_internal_latency_ms IS NOT NULL
            AND tc.tool_wall_latency_ms > 0
            AND tc.tool_internal_latency_ms >= 0
        ), bucketed AS (
          SELECT *, {internal_bucket} AS bucket FROM paired
        )
        SELECT bucket, count(*) AS calls,
               round(100.0 * count(*) / sum(count(*)) OVER (), 3) AS call_pct,
               round(quantile_cont(overhead_ms, 0.50), 3) AS overhead_p50_ms,
               round(quantile_cont(overhead_ms, 0.90), 3) AS overhead_p90_ms,
               round(quantile_cont(wall_ms, 0.50), 3) AS e2e_p50_ms,
               round(quantile_cont(internal_ms, 0.50), 3) AS internal_p50_ms
        FROM bucketed GROUP BY bucket ORDER BY {BUCKET_ORDER_SQL}
        """,
    )

    write_query(
        con,
        output / "tracelab_codex_overhead_by_tool.csv",
        """
        WITH paired AS (
          SELECT tc.tool_name,
                 tc.tool_wall_latency_ms AS wall_ms,
                 tc.tool_internal_latency_ms AS internal_ms,
                 greatest(tc.tool_wall_latency_ms - tc.tool_internal_latency_ms, 0) AS overhead_ms
          FROM tool_calls tc JOIN rounds r USING (round_pk)
          WHERE r.provider = 'codex'
            AND tc.tool_wall_latency_ms IS NOT NULL
            AND tc.tool_internal_latency_ms IS NOT NULL
            AND tc.tool_wall_latency_ms > 0
            AND tc.tool_internal_latency_ms >= 0
        )
        SELECT tool_name, count(*) AS calls,
               round(sum(wall_ms) / 3600000.0, 3) AS e2e_hours,
               round(sum(internal_ms) / 3600000.0, 3) AS internal_hours,
               round(sum(overhead_ms) / 3600000.0, 3) AS overhead_hours,
               round(avg(overhead_ms), 3) AS overhead_mean_ms,
               round(quantile_cont(overhead_ms, 0.50), 3) AS overhead_p50_ms,
               round(quantile_cont(overhead_ms, 0.90), 3) AS overhead_p90_ms,
               round(quantile_cont(overhead_ms, 0.99), 3) AS overhead_p99_ms
        FROM paired
        WHERE tool_name IN ('exec_command', 'write_stdin', 'shell_command', 'apply_patch')
        GROUP BY tool_name ORDER BY calls DESC
        """,
    )
    con.close()


def analyze_swe_chat(conversations: Path, sessions: Path, output: Path) -> None:
    con = duckdb.connect()
    con.read_parquet(str(conversations)).create_view("conversations")
    con.read_parquet(str(sessions)).create_view("sessions")
    con.execute(
        """
        CREATE TEMP TABLE swe_pairs AS
        WITH uses AS (
          SELECT session_id, tool_call_id, turn_id, timestamp, agent, tool_name,
                 row_number() OVER (
                   PARTITION BY session_id, tool_call_id ORDER BY turn_id
                 ) AS occurrence
          FROM conversations
          WHERE turn_type='tool_use' AND timestamp IS NOT NULL
            AND tool_call_id IS NOT NULL
        ), results AS (
          SELECT session_id, tool_call_id, turn_id, timestamp,
                 row_number() OVER (
                   PARTITION BY session_id, tool_call_id ORDER BY turn_id
                 ) AS occurrence
          FROM conversations
          WHERE turn_type='tool_result' AND timestamp IS NOT NULL
            AND tool_call_id IS NOT NULL
        )
        SELECT u.agent, u.tool_name,
               epoch_ms(r.timestamp) - epoch_ms(u.timestamp) AS wall_ms
        FROM uses u JOIN results r USING (session_id, tool_call_id, occurrence)
        """
    )
    bucket = BUCKET_SQL.format(value="wall_ms")
    write_query(
        con,
        output / "swe_chat_wall_latency.csv",
        f"""
        WITH bucketed AS (
          SELECT {bucket} AS bucket FROM swe_pairs WHERE wall_ms > 0
        )
        SELECT bucket, count(*) AS calls,
               sum(count(*)) OVER () AS total_calls,
               round(100.0 * count(*) / sum(count(*)) OVER (), 3) AS call_pct
        FROM bucketed GROUP BY bucket ORDER BY {BUCKET_ORDER_SQL}
        """,
    )
    write_query(
        con,
        output / "swe_chat_top_tools.csv",
        """
        SELECT coalesce(tool_name, '<unknown-tool>') AS tool_name,
               count(*) AS calls,
               round(100.0 * count(*) / sum(count(*)) OVER (), 3) AS call_pct,
               round(quantile_cont(wall_ms, 0.50), 3) AS p50_ms,
               round(quantile_cont(wall_ms, 0.90), 3) AS p90_ms,
               round(quantile_cont(wall_ms, 0.99), 3) AS p99_ms
        FROM swe_pairs WHERE wall_ms > 0
        GROUP BY tool_name ORDER BY calls DESC LIMIT 15
        """,
    )
    write_query(
        con,
        output / "swe_chat_timing_coverage_by_agent.csv",
        """
        WITH uses AS (
          SELECT coalesce(agent, '<unknown-agent>') AS agent,
                 count(*) AS tool_uses,
                 count(*) FILTER (
                   WHERE timestamp IS NOT NULL AND tool_call_id IS NOT NULL
                 ) AS timed_uses
          FROM conversations WHERE turn_type='tool_use' GROUP BY agent
        ), pairs AS (
          SELECT coalesce(agent, '<unknown-agent>') AS agent,
                 count(*) AS paired,
                 count(*) FILTER (WHERE wall_ms > 0) AS positive_pairs
          FROM swe_pairs GROUP BY agent
        )
        SELECT u.agent, u.tool_uses, u.timed_uses,
               coalesce(p.paired, 0) AS paired,
               coalesce(p.positive_pairs, 0) AS positive_pairs
        FROM uses u LEFT JOIN pairs p USING (agent)
        ORDER BY tool_uses DESC
        """,
    )
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracelab-db", type=Path)
    parser.add_argument("--swe-conversations", type=Path)
    parser.add_argument("--swe-sessions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    if args.tracelab_db:
        analyze_tracelab(args.tracelab_db, args.output_dir)
    if bool(args.swe_conversations) != bool(args.swe_sessions):
        parser.error("--swe-conversations and --swe-sessions must be supplied together")
    if args.swe_conversations:
        analyze_swe_chat(args.swe_conversations, args.swe_sessions, args.output_dir)
    if not args.tracelab_db and not args.swe_conversations:
        parser.error("provide TraceLab, SWE-chat, or both input datasets")


if __name__ == "__main__":
    main()
