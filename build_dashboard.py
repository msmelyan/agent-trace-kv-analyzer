#!/usr/bin/env python3
"""Build dashboard.html from the per-dataset *.stats.json files.
Run AFTER analysis_engine.py has produced stats for each dataset.
Usage: python build_dashboard.py   ->  writes ./dashboard.html
"""
import os, glob, json

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "agentic_traces")
TEMPLATE = os.path.join(BASE, "dashboard_template.html")
OUT = os.path.join(BASE, "dashboard.html")

# editorial metadata per dataset (not derivable from numbers): display order, arch label,
# native-model tag, baseline flag. Keys must match dataset dir / stats key.
META = {
    "axt-swe-agent-lm-32b": dict(arch="SWE-agent-LM-32B · append-only ReAct", model="32B"),
    "nebius-swe-agent":     dict(arch="SWE-agent · ACI commands",             model="7B"),
    "swegym-openhands":     dict(arch="OpenHands · CodeAct tool-calling",     model="32B"),
    "r2egym-sft":           dict(arch="ReAct · &lt;function=&gt; (SFT set)",  model="32B"),
    "swegym-moatless":      dict(arch="Moatless · state machine",             model="32B"),
    "r2egym-verifier":      dict(arch="one-shot verifier / judge",            model="32B"),
    "sharegpt-v3":          dict(arch="human↔ChatGPT chat · baseline (no tools)", model="8B", baseline=True),
}
ORDER = list(META)  # display order

def load(key):
    fp = os.path.join(D, key, f"{key}.stats.json")
    return json.load(open(fp)) if os.path.exists(fp) else None

def main():
    ds = []
    scale = []
    for key in ORDER:
        s = load(key)
        if not s:
            print(f"  (skip {key}: no stats.json — run analysis_engine.py {key.split('-')[0]} first)")
            continue
        m = META[key]
        row = dict(key=key, arch=m["arch"], model=m["model"],
                   sess=s["nsess"],
                   steps=round(s["steps"]["p50"]),
                   isl=round(s["isl"]["p50"]),
                   osl=round(s["osl"]["p50"]),
                   hit=round(s["hit_agg"]*100, 1),
                   ctx50=round(s["peak"]["p50"]),
                   ctx99=round(s["peak"]["p99"]))
        if m.get("baseline"):
            row["baseline"] = True
        ds.append(row)
        # AxT context-budget scaling table from its groups
        if key == "axt-swe-agent-lm-32b":
            def budget_label(g):  # "ctx131072" -> "ctx 131K"
                n = int(g.replace("ctx", ""))
                return f"ctx {n//1024}K"
            for g in sorted(s["groups"], key=lambda x: int(x.replace("ctx", ""))):
                gg = s["groups"][g]
                scale.append(dict(b=budget_label(g), steps=round(gg["steps"], 1),
                                  ctx=round(gg["ctx"]), hit=round(gg["hit"]*100, 1)))

    if not ds:
        print("No stats found. Run: python analysis_engine.py <key> for each dataset first."); return
    html = open(TEMPLATE).read()
    html = html.replace("__DS__", json.dumps(ds)).replace("__SCALE__", json.dumps(scale))
    open(OUT, "w").write(html)
    print(f"wrote {OUT}  ({len(ds)} datasets, {len(scale)} scaling rows)")
    print("Open it in a browser, or publish it as a Claude artifact.")

if __name__ == "__main__":
    main()
