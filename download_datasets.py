#!/usr/bin/env python3
"""Download all trace datasets into ./agentic_traces/<name>/.
Usage: python download_datasets.py [name ...]   (default: all)
Datasets are large (~7 GB total) and git-ignored; this script recreates them.
"""
import os, sys
from huggingface_hub import snapshot_download, hf_hub_download

BASE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(BASE, "agentic_traces")

# HF snapshot datasets: (repo, local-subdir, allow-patterns)
SNAPSHOTS = {
    "axt-swe-agent-lm-32b": ("AxT-dev/swe-agent-lm-32b-r2e-gym-trajectories", ["*.jsonl"]),
    "nebius-swe-agent":     ("nebius/SWE-agent-trajectories",                 ["*.parquet"]),
    "r2egym-sft":           ("R2E-Gym/R2EGym-SFT-Trajectories",               ["*.parquet"]),
    "r2egym-verifier":      ("R2E-Gym/R2EGym-Verifier-Trajectories",          ["*.parquet"]),
    "swegym-openhands":     ("SWE-Gym/OpenHands-Sampled-Trajectories",        ["*.parquet"]),
}
# single-file / custom pulls
SINGLE = {
    "sharegpt-v3": ("anon8231489123/ShareGPT_Vicuna_unfiltered",
                    ["ShareGPT_V3_unfiltered_cleaned_split.json"]),
    # Moatless ships zipped run-dumps; grab two representative bench-lite 32B releases
    "swegym-moatless": ("SWE-Gym/MoatlessTools-Sampled-Trajectories",
                        ["32b/release-32b-zero-bench-lite-temp_0.0-fp8_0.zip",
                         "32b/release-32b-it1-v2-bench-lite-temp_0.0-fp8_0.zip",
                         "32b/README.md"]),
}

def dl(name):
    dst = os.path.join(DEST, name); os.makedirs(dst, exist_ok=True)
    if name in SNAPSHOTS:
        repo, patterns = SNAPSHOTS[name]
        print(f"[{name}] snapshot {repo} …", flush=True)
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dst,
                          allow_patterns=patterns, max_workers=8)
    elif name in SINGLE:
        repo, files = SINGLE[name]
        for f in files:
            print(f"[{name}] {repo} :: {f}", flush=True)
            hf_hub_download(repo_id=repo, repo_type="dataset", filename=f, local_dir=dst)
    else:
        print(f"[{name}] unknown dataset", flush=True); return
    print(f"[{name}] done", flush=True)

if __name__ == "__main__":
    names = sys.argv[1:] or (list(SNAPSHOTS) + list(SINGLE))
    for n in names:
        dl(n)
    print("ALL DOWNLOADS COMPLETE")
