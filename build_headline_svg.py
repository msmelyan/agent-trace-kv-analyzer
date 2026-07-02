#!/usr/bin/env python3
"""Render the 'prefix-cache hit ratio' headline chart as an SVG, from real *.stats.json.
Usage: python build_headline_svg.py  ->  assets/prefix_cache_hit.svg
"""
import os, json

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "agentic_traces")
OUT_DIR = os.path.join(BASE, "assets"); os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "prefix_cache_hit.svg")

# display order + short labels (match the dashboard)
ROWS = [
    ("axt-swe-agent-lm-32b", "axt-lm-32b"),
    ("nebius-swe-agent",     "nebius"),
    ("swegym-openhands",     "openhands"),
    ("r2egym-sft",           "r2egym-sft"),
    ("swegym-moatless",      "moatless"),
    ("r2egym-verifier",      "r2egym-verifier"),
    ("sharegpt-v3",          "sharegpt-v3  ◇ chat"),
]
def hit_of(key):
    fp = os.path.join(D, key, f"{key}.stats.json")
    return json.load(open(fp))["hit_agg"] * 100 if os.path.exists(fp) else None
def color(h):
    return "#4bb377" if h >= 90 else "#d9a441" if h >= 50 else "#d8664a"

GROUND, PANEL, INK, MUTED, FAINT, LINE, ACCENT = (
    "#0f1417", "#1b252a", "#e8eef0", "#95a7ac", "#6b7d82", "#26333a", "#5cc4bb")
W = 980
LABEL_X, BAR_X, BAR_W, VAL_X = 250, 268, 590, 872
Y0, ROWH, BARH = 150, 40, 24

rows = [(lbl, hit_of(k)) for k, lbl in ROWS]
rows = [(l, h) for l, h in rows if h is not None]
H = Y0 + len(rows) * ROWH + 20

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;")

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="system-ui,Segoe UI,Roboto,sans-serif">')
svg.append(f'<rect width="{W}" height="{H}" fill="{GROUND}"/>')
# title
svg.append(f'<text x="40" y="42" fill="{ACCENT}" font-family="ui-monospace,Menlo,Consolas,monospace" '
           f'font-size="15" letter-spacing="2" font-weight="700">PREFIX-CACHE HIT RATIO — THE HEADLINE</text>')
svg.append(f'<line x1="40" y1="58" x2="{W-40}" y2="58" stroke="{LINE}"/>')
# subtitle (manual wrap)
sub = ["1 − Σ nISL / Σ ISL: the share of input tokens already in cache (cISL), free to skip.",
       "Append-only ReAct/CodeAct agents cluster at 93–97% — nearly all input is reused prefix.",
       "Moatless (a state machine that rebuilds its prompt per phase) collapses to ~36%; the verifier is one call."]
for i, line in enumerate(sub):
    svg.append(f'<text x="40" y="{86+i*20}" fill="{MUTED}" font-size="14.5">{esc(line)}</text>')
# bars
for i, (lbl, h) in enumerate(rows):
    y = Y0 + i * ROWH
    cy = y + BARH/2
    svg.append(f'<text x="{LABEL_X}" y="{cy+5}" text-anchor="end" fill="{INK}" font-size="15">{esc(lbl)}</text>')
    svg.append(f'<rect x="{BAR_X}" y="{y}" width="{BAR_W}" height="{BARH}" rx="5" fill="{PANEL}"/>')
    w = max(h/100*BAR_W, 3)
    svg.append(f'<rect x="{BAR_X}" y="{y}" width="{w:.1f}" height="{BARH}" rx="5" fill="{color(h)}"/>')
    svg.append(f'<text x="{VAL_X}" y="{cy+5}" fill="{INK}" font-family="ui-monospace,Menlo,Consolas,monospace" '
               f'font-size="14">{h:.1f}%</text>')
svg.append('</svg>')
open(OUT, "w").write("\n".join(svg))
print(f"wrote {OUT}  ({len(rows)} datasets)")
