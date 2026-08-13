"""Weka hash-LCP chain reconstruction used by the AgentX analysis.

This is a compact, dictionary-based adaptation of the Apache-2.0 AIPerf
implementation in ``aiperf.dataset.loader.weka_agent_chains`` at commit
``c26fe88bd88d32740c83b4637c1e573e06df5ecc``.  The defaults intentionally
match AIPerf so this analysis and a replay reconstruct the same logical streams.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

Request = dict[str, Any]
IndexedRequest = tuple[int, Request]
EPSILON_SECONDS = 1e-6


def req_end(req: Request) -> float:
    duration = req.get("api_time")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration):
        duration = 0.0
    return float(req["t"]) + max(float(duration), 0.0)


def lcp_blocks(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass(slots=True)
class ChainFork:
    parent_chain: int | None
    fork_outer_idx: int | None
    depth: int
    fork_time: float


@dataclass(slots=True)
class AgentChain:
    requests: list[IndexedRequest] = field(default_factory=list)
    fork: ChainFork | None = None
    spliced_into: int | None = None
    tail_outer_idx: int = -1
    tail_hash: list[int] = field(default_factory=list)
    tail_end: float = 0.0
    tail_model: str = ""


@dataclass(slots=True)
class ChainDetectionResult:
    chains: list[AgentChain]
    main_index: int
    worker_indices: list[int]
    seams_merged: int
    unclassified_empty_hash: int


def _find_extension_target(
    chains: list[AgentChain], h: list[int], t: float, model: str
) -> int | None:
    best: int | None = None
    best_len = -1
    for idx, chain in enumerate(chains):
        tail_len = len(chain.tail_hash)
        if tail_len == 0 or tail_len > len(h) or tail_len <= best_len:
            continue
        if chain.tail_model != model or chain.tail_end > t + EPSILON_SECONDS:
            continue
        if h[:tail_len] == chain.tail_hash:
            best, best_len = idx, tail_len
    return best


def _max_lcp_chain(chains: list[AgentChain], h: list[int]) -> tuple[int | None, int]:
    best_idx: int | None = None
    best_key = (0, 0)
    for idx, chain in enumerate(chains):
        if not chain.tail_hash:
            continue
        depth = lcp_blocks(chain.tail_hash, h)
        key = (depth, len(chain.tail_hash))
        if depth and key > best_key:
            best_idx, best_key = idx, key
    return best_idx, best_key[0]


def _append(
    chains: list[AgentChain], chain_idx: int, outer_idx: int, req: Request
) -> None:
    chain = chains[chain_idx]
    chain.requests.append((outer_idx, req))
    if req.get("hash_ids"):
        chain.tail_outer_idx = outer_idx
        chain.tail_hash = req["hash_ids"]
        chain.tail_end = req_end(req)
        chain.tail_model = req["model"]


def _last_hash_outer_idx(chain: AgentChain) -> int | None:
    for outer_idx, req in reversed(chain.requests):
        if req.get("hash_ids"):
            return outer_idx
    return None


def _elect_continuation(
    chains: list[AgentChain],
    registered: list[int],
    tail_req: Request,
    *,
    max_gap_seconds: float,
    min_overlap_ratio: float,
) -> int | None:
    tail_end = req_end(tail_req)
    tail_blocks = len(tail_req.get("hash_ids", []))

    def blocked(ci: int) -> bool:
        if not tail_blocks:
            return False
        first_req = chains[ci].requests[0][1]
        gap = float(first_req["t"]) - tail_end
        overlap = chains[ci].fork.depth / tail_blocks  # type: ignore[union-attr]
        return gap > max_gap_seconds and overlap < min_overlap_ratio

    candidates = [
        ci
        for ci in registered
        if chains[ci].fork is not None
        and chains[ci].fork.depth > 0
        and tail_end <= float(chains[ci].requests[0][1]["t"]) + EPSILON_SECONDS
        and chains[ci].requests[0][1]["model"] == tail_req["model"]
        and not blocked(ci)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda ci: (
            chains[ci].fork.depth,  # type: ignore[union-attr]
            -chains[ci].fork.fork_time,  # type: ignore[union-attr]
            -ci,
        ),
    )


def _resolve_seams(
    chains: list[AgentChain],
    forks_by_tail: dict[int, list[int]],
    chain_of_request: dict[int, int],
    req_by_outer: dict[int, Request],
    *,
    max_gap_seconds: float,
    min_overlap_ratio: float,
) -> int:
    aliases: dict[int, int] = {}

    def resolve(i: int) -> int:
        while i in aliases:
            i = aliases[i]
        return i

    seams = 0
    keys = list(forks_by_tail)
    heapq.heapify(keys)
    processed: set[int] = set()
    while keys:
        fork_outer_idx = heapq.heappop(keys)
        if fork_outer_idx in processed:
            continue
        processed.add(fork_outer_idx)
        owner = resolve(chain_of_request[fork_outer_idx])
        owner_chain = chains[owner]
        if _last_hash_outer_idx(owner_chain) != fork_outer_idx:
            continue
        registered = [
            ci for ci in forks_by_tail[fork_outer_idx] if chains[ci].spliced_into is None
        ]
        elected = _elect_continuation(
            chains,
            registered,
            req_by_outer[fork_outer_idx],
            max_gap_seconds=max_gap_seconds,
            min_overlap_ratio=min_overlap_ratio,
        )
        if elected is None:
            continue
        target = chains[elected]
        owner_chain.requests.extend(target.requests)
        owner_chain.requests.sort(key=lambda item: (float(item[1]["t"]), item[0]))
        for outer_idx, _ in target.requests:
            chain_of_request[outer_idx] = owner
        target.spliced_into = owner
        aliases[elected] = owner
        seams += 1

        new_tail_outer = _last_hash_outer_idx(owner_chain)
        if new_tail_outer is None:
            continue
        new_tail_hash = req_by_outer[new_tail_outer].get("hash_ids", [])
        rekeyed = False
        for ci in registered:
            chain = chains[ci]
            if ci == elected or chain.spliced_into is not None or chain.fork is None:
                continue
            depth = lcp_blocks(new_tail_hash, chain.requests[0][1].get("hash_ids", []))
            if depth <= 0:
                continue
            chain.fork.fork_outer_idx = new_tail_outer
            chain.fork.depth = depth
            chain.fork.parent_chain = owner
            forks_by_tail.setdefault(new_tail_outer, []).append(ci)
            rekeyed = True
        if rekeyed:
            processed.discard(new_tail_outer)
            heapq.heappush(keys, new_tail_outer)
    return seams


def detect_agent_chains(
    normals: list[IndexedRequest],
    *,
    seam_max_gap_seconds: float = 3600.0,
    seam_min_overlap_ratio: float = 0.5,
) -> ChainDetectionResult:
    """Partition requests into AIPerf-compatible logical context chains."""
    if not normals:
        return ChainDetectionResult([], 0, [], 0, 0)

    chains: list[AgentChain] = []
    chain_of_request: dict[int, int] = {}
    forks_by_tail: dict[int, list[int]] = {}
    req_by_outer: dict[int, Request] = {}
    unclassified = 0

    ordered = sorted(normals, key=lambda item: (float(item[1]["t"]), item[0]))
    for outer_idx, req in ordered:
        req_by_outer[outer_idx] = req
        h = req.get("hash_ids", [])
        if not h:
            unclassified += 1
            if not chains:
                chains.append(AgentChain())
            chains[0].requests.append((outer_idx, req))
            chain_of_request[outer_idx] = 0
            continue
        if not chains:
            chains.append(AgentChain())
            _append(chains, 0, outer_idx, req)
            chain_of_request[outer_idx] = 0
            continue
        target = _find_extension_target(chains, h, float(req["t"]), req["model"])
        if target is not None:
            _append(chains, target, outer_idx, req)
            chain_of_request[outer_idx] = target
            continue
        parent, depth = _max_lcp_chain(chains, h)
        if parent is None and all(not c.tail_hash for c in chains):
            _append(chains, 0, outer_idx, req)
            chain_of_request[outer_idx] = 0
            continue
        fork = ChainFork(
            parent_chain=parent,
            fork_outer_idx=chains[parent].tail_outer_idx if parent is not None else None,
            depth=depth,
            fork_time=float(req["t"]),
        )
        new_idx = len(chains)
        chains.append(AgentChain(fork=fork))
        _append(chains, new_idx, outer_idx, req)
        chain_of_request[outer_idx] = new_idx
        if fork.fork_outer_idx is not None and depth > 0:
            forks_by_tail.setdefault(fork.fork_outer_idx, []).append(new_idx)

    seams = _resolve_seams(
        chains,
        forks_by_tail,
        chain_of_request,
        req_by_outer,
        max_gap_seconds=seam_max_gap_seconds,
        min_overlap_ratio=seam_min_overlap_ratio,
    )
    aliases = {
        i: c.spliced_into for i, c in enumerate(chains) if c.spliced_into is not None
    }

    def resolve(i: int) -> int:
        while i in aliases:
            i = aliases[i]  # type: ignore[assignment]
        return i

    for chain in chains:
        if (
            chain.spliced_into is None
            and chain.fork is not None
            and chain.fork.parent_chain is not None
        ):
            chain.fork.parent_chain = resolve(chain.fork.parent_chain)
    main_index = resolve(chain_of_request[ordered[0][0]])
    workers = [
        i for i, c in enumerate(chains) if c.spliced_into is None and i != main_index
    ]
    workers.sort(
        key=lambda i: (float(chains[i].requests[0][1]["t"]), chains[i].requests[0][0])
    )
    return ChainDetectionResult(chains, main_index, workers, seams, unclassified)


def split_off_preamble(
    normals: list[IndexedRequest], max_output_tokens: int = 64
) -> tuple[list[IndexedRequest], list[IndexedRequest]]:
    """Apply AIPerf's one-request title/one-shot preamble rule."""
    if len(normals) < 2:
        return [], normals
    ordered = sorted(normals, key=lambda item: (float(item[1]["t"]), item[0]))
    outer_idx, req = ordered[0]
    h = req.get("hash_ids", [])
    if not h:
        return [], normals
    rest = ordered[1:]
    if any(lcp_blocks(h, other.get("hash_ids", [])) > 0 for _, other in rest):
        return [], normals
    if int(req.get("out", 0)) > max_output_tokens:
        other_blocks = {hid for _, other in rest for hid in other.get("hash_ids", [])}
        if not other_blocks.isdisjoint(h):
            return [], normals
    return [(outer_idx, req)], sorted(rest, key=lambda item: item[0])


def is_aux_chain(
    requests: list[Request],
    main_peak_isl: int,
    *,
    main_model: str | None,
    max_requests: int = 1,
    isl_ratio: float = 0.10,
    isl_floor: int = 16384,
    cross_model: bool = True,
) -> bool:
    if not requests or len(requests) > max_requests:
        return False
    first = requests[0]
    if cross_model and main_model is not None and first["model"] != main_model:
        return True
    return int(first["in"]) < max(isl_floor, int(isl_ratio * main_peak_isl))


def is_reduction_chain(
    requests: list[Request],
    *,
    osl_max: int = 4000,
    ratio: float = 20.0,
    isl_floor: int = 16384,
) -> bool:
    if osl_max <= 0 or len(requests) != 1:
        return False
    first = requests[0]
    osl = int(first.get("out", 0))
    return (
        0 < osl < osl_max
        and int(first["in"]) >= isl_floor
        and int(first["in"]) > ratio * osl
    )


def _overlap_components(
    candidates: list[tuple[float, float, int, int]],
) -> list[list[tuple[float, float, int, int]]]:
    if not candidates:
        return []
    order = sorted(range(len(candidates)), key=lambda i: (candidates[i][0], candidates[i][2]))
    parent = list(range(len(candidates)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    active: list[tuple[float, int]] = []
    for i in order:
        t0 = candidates[i][0]
        while active and active[0][0] <= t0:
            heapq.heappop(active)
        if active:
            parent[find(i)] = find(active[0][1])
        heapq.heappush(active, (candidates[i][1], i))
    components: dict[int, list[tuple[float, float, int, int]]] = {}
    for i, candidate in enumerate(candidates):
        components.setdefault(find(i), []).append(candidate)
    return list(components.values())


def worker_group_assignment(
    result: ChainDetectionResult, *, group_min: int = 3
) -> dict[int, tuple[int, int]]:
    if group_min <= 0:
        return {}
    buckets: dict[tuple[int, int], list[tuple[float, float, int, int]]] = {}
    for ci in result.worker_indices:
        chain = result.chains[ci]
        fork = chain.fork
        if (
            fork is None
            or fork.depth <= 0
            or fork.parent_chain is None
            or fork.fork_outer_idx is None
            or not chain.requests
        ):
            continue
        t0 = float(chain.requests[0][1]["t"])
        t1 = max(req_end(req) for _, req in chain.requests)
        buckets.setdefault((fork.parent_chain, fork.fork_outer_idx), []).append(
            (t0, t1, chain.requests[0][0], ci)
        )
    components = [comp for vals in buckets.values() for comp in _overlap_components(vals)]
    groups = [comp for comp in components if len(comp) >= group_min]
    groups.sort(key=lambda comp: min((x[0], x[2]) for x in comp))
    out: dict[int, tuple[int, int]] = {}
    for group, comp in enumerate(groups):
        comp.sort(key=lambda x: (x[0], x[2]))
        for member, (_, _, _, ci) in enumerate(comp):
            out[ci] = (group, member)
    return out


def classify_worker_chains(
    result: ChainDetectionResult,
) -> dict[int, tuple[str, bool, bool, tuple[int, int] | None]]:
    """Return ``chain_idx -> (kind, is_aux, is_reduction, wg_coord)``."""
    if not result.chains:
        return {}
    main = result.chains[result.main_index]
    main_reqs = [req for _, req in main.requests]
    main_peak = max((int(req["in"]) for req in main_reqs), default=0)
    main_model = main_reqs[0]["model"] if main_reqs else None
    wg = worker_group_assignment(result)
    out: dict[int, tuple[str, bool, bool, tuple[int, int] | None]] = {}
    for ci in result.worker_indices:
        reqs = [req for _, req in result.chains[ci].requests]
        aux = is_aux_chain(reqs, main_peak, main_model=main_model)
        reduction = not aux and is_reduction_chain(reqs)
        coord = wg.get(ci) if not aux and not reduction else None
        kind = "aux" if aux or reduction else "worker_group" if coord else "flat_agent"
        out[ci] = (kind, aux, reduction, coord)
    return out


def detect_with_preamble(
    normals: list[IndexedRequest],
) -> tuple[
    ChainDetectionResult,
    list[IndexedRequest],
    dict[int, tuple[str, bool, bool, tuple[int, int] | None]],
]:
    """Detect/classify chains, then reattach AIPerf's preamble to main.

    Classification must happen before reattachment: AIPerf deliberately keeps
    a disjoint title-generation request from redefining the main model/peak ISL
    used to classify worker chains.
    """
    preamble, detect_normals = split_off_preamble(normals)
    result = detect_agent_chains(detect_normals)
    classifications = classify_worker_chains(result)
    if result.chains and preamble:
        result.chains[result.main_index].requests = sorted(
            preamble + result.chains[result.main_index].requests,
            key=lambda item: (float(item[1]["t"]), item[0]),
        )
    return result, preamble, classifications


def active_interval_peak(intervals: list[tuple[float, float]]) -> int:
    """Maximum simultaneous half-open intervals; end events win ties."""
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        events.append((start, 1))
        events.append((end, -1))
    active = peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        peak = max(peak, active)
    return peak
