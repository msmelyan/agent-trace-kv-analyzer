#!/usr/bin/env python3
"""Trace analysis engine: ISL/OSL, steps, context growth, tool mix, prefix-cache modeling.
Run: analysis_engine.py <dataset_key>   (writes agentic_traces/<key>/<key>.md)
"""
import os, sys, re, json, math, glob, zipfile, random
from collections import Counter, defaultdict

import tiktoken
ENC = tiktoken.get_encoding("o200k_base")
def ntok(s):
    if not s: return 0
    return len(ENC.encode(s, disallowed_special=()))

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "agentic_traces")   # dataset dirs; each holds its own <key>.md report
BLOCK = 16  # vLLM default KV block size (tokens)

# KV bytes/token = 2(K,V) * layers * kv_heads * head_dim * dtype_bytes
def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * dtype_bytes
QWEN32B = dict(name="Qwen2.5-Coder-32B", layers=64, kv_heads=8, head_dim=128)
QWEN7B  = dict(name="Qwen2.5-Coder-7B",  layers=28, kv_heads=4, head_dim=128)
LLAMA8B = dict(name="Llama-3.1-8B",       layers=32, kv_heads=8, head_dim=128)

FUNC_RE = re.compile(r"<function=([A-Za-z0-9_\-\.]+)")
BASH_HINT = re.compile(r"```bash|<command>|\bexecute_bash\b")

def pct(v, p):
    if not v: return 0
    v=sorted(v); k=(len(v)-1)*p; f=math.floor(k); c=math.ceil(k)
    return v[int(k)] if f==c else v[f]*(c-k)+v[c]*(k-f)
def summ(v):
    if not v: return dict(n=0)
    s=sorted(v)
    return dict(n=len(s), mean=sum(s)/len(s), p50=pct(s,.5), p90=pct(s,.9),
               p95=pct(s,.95), p99=pct(s,.99), mx=s[-1], mn=s[0])

# ---- prefix-cache model over one session (list of steps: dict(isl=,osl=,tool=)) ----
def session_cache(steps):
    """steps: ordered; isl = cumulative input tokens to that call, osl = output tokens."""
    prev_cum = None
    prefill_nc = 0   # no-cache prefill = sum isl
    prefill_uc = 0   # unique (uncached/new) prefill = sum nISL
    compactions = 0
    peak_ctx = 0
    for i, st in enumerate(steps):
        isl, osl = st["isl"], st["osl"]
        prefill_nc += isl
        if prev_cum is None:
            nisl = isl
        elif isl >= prev_cum:
            cisl = prev_cum
            nisl = isl - cisl
        else:  # context shrank -> compaction / history rewrite
            compactions += 1
            nisl = isl              # conservative: treat as cache reset
        prefill_uc += nisl
        peak_ctx = max(peak_ctx, isl + osl)
        prev_cum = isl + osl
    hit = 1 - prefill_uc/prefill_nc if prefill_nc else 0.0
    return dict(prefill_nc=prefill_nc, prefill_uc=prefill_uc, hit=hit,
                compactions=compactions, peak_ctx=peak_ctx)

# ============================ LOADERS ============================
# Each yields sessions as: dict(steps=[dict(isl,osl,tool)], reward=?, meta=?)

def load_axt(limit_per_file=700):
    files = sorted(glob.glob(f"{D}/axt-swe-agent-lm-32b/*.jsonl"))
    for fp in files:
        ctx = re.search(r"ctx(\d+)", fp).group(1)
        attempt = re.search(r"attempt-(\d)", fp).group(1)
        n=0
        with open(fp) as f:
            for line in f:
                if n>=limit_per_file: break
                try: r=json.loads(line)
                except: continue
                tsteps=r.get("trajectory_steps") or []
                if not tsteps: continue
                steps=[]
                for s in tsteps:
                    isl=s.get("token_usage_prompt") or 0
                    osl=s.get("token_usage_completion") or 0
                    act=s.get("action") or ""
                    m=FUNC_RE.search(act); tool=m.group(1) if m else ("bash" if BASH_HINT.search(act) else "other")
                    steps.append(dict(isl=isl, osl=osl, tool=tool,
                                      env_t=s.get("env_exec_time") or 0.0,
                                      llm_t=s.get("llm_exec_time") or 0.0))
                if steps:
                    yield dict(steps=steps, reward=r.get("reward"),
                               group=f"ctx{ctx}", attempt=attempt,
                               exit=r.get("exit_reason"))
                    n+=1

def _msg_sessions_parquet(pattern, role_field, content_getter, is_assistant,
                          tool_getter, sample=None, extra=None):
    """Generic: read parquet rows, each row -> one session from a message list."""
    import duckdb
    con=duckdb.connect()
    files=sorted(glob.glob(pattern))
    # count total rows
    total=con.sql(f"SELECT count(*) FROM read_parquet({files!r})").fetchone()[0]
    q=f"SELECT * FROM read_parquet({files!r})"
    if sample and sample<total:
        q=f"SELECT * FROM read_parquet({files!r}) USING SAMPLE {sample} ROWS (reservoir, 42)"
    rel=con.sql(q)
    cols=[c for c in rel.columns]
    for row in rel.fetchall():
        d=dict(zip(cols,row))
        yield d
    con.close()

def load_msglist(key, pattern, msg_col, role_map, tool_from, sample, reward_col=None):
    """role_map: dict mapping raw role->'user'/'assistant'/'system'/'tool'.
       tool_from: fn(msg_dict)->tool name or None (for assistant msgs)."""
    import duckdb
    con=duckdb.connect()
    files=sorted(glob.glob(pattern))
    total=con.sql(f"SELECT count(*) FROM read_parquet({files!r})").fetchone()[0]
    sel=f"SELECT * FROM read_parquet({files!r})"
    if sample and sample<total:
        sel=f"SELECT * FROM read_parquet({files!r}) USING SAMPLE {sample} ROWS (reservoir, 42)"
    rel=con.sql(sel); cols=rel.columns
    used=0
    for row in rel.fetchall():
        d=dict(zip(cols,row))
        msgs=d.get(msg_col) or []
        cum=0; steps=[]
        for m in msgs:
            role=role_map.get(m.get("role"), m.get("role"))
            # content text
            txt = m.get("content") or m.get("text") or ""
            tks = ntok(txt)
            # tool-call args (openhands)
            tcs = m.get("tool_calls")
            tool=None
            if role=="assistant":
                # this message is a model output => a step
                osl=tks
                if tcs:
                    for tc in tcs:
                        fn=tc.get("function") or {}
                        osl+=ntok(fn.get("arguments") or "")
                        tool=tool or fn.get("name")
                if tool is None:
                    tool=tool_from(txt)
                steps.append(dict(isl=cum, osl=osl, tool=tool or "none"))
                cum += osl
            else:
                cum += tks
                if tcs:
                    for tc in tcs:
                        fn=tc.get("function") or {}
                        cum+=ntok(fn.get("arguments") or "")
        if steps:
            rew = d.get(reward_col) if reward_col else None
            yield dict(steps=steps, reward=rew, group=key)
            used+=1
    con.close()
    sys.stderr.write(f"[{key}] sampled {used}/{total} sessions\n")

def load_sharegpt(sample=4000):
    """ShareGPT V3 chat. Session=conversation; step=each assistant(gpt) turn.
       ISL=cumulative context tokens before the turn, OSL=that turn's tokens."""
    fp=f"{D}/sharegpt-v3/ShareGPT_V3_unfiltered_cleaned_split.json"
    data=json.load(open(fp))
    USER={"human","user"}; ASST={"gpt","chatgpt","bard","bing"}
    idxs=list(range(len(data)))
    if sample and sample<len(idxs):
        random.Random(42).shuffle(idxs); idxs=idxs[:sample]
    for i in idxs:
        convs=data[i].get("conversations") or []
        cum=0; steps=[]
        for c in convs:
            role=c.get("from"); val=c.get("value") or ""; t=ntok(val)
            if role in ASST:
                steps.append(dict(isl=cum, osl=t, tool="chat"))
                cum+=t
            else:
                cum+=t
        if steps:
            yield dict(steps=steps, reward=None, group="sharegpt")

def load_moatless(sample_instances=None):
    zips=sorted(glob.glob(f"{D}/swegym-moatless/32b/*.zip"))
    for zp in zips:
        tag=re.search(r"release-32b-([a-z0-9\-]+?)-bench", os.path.basename(zp))
        tag=tag.group(1) if tag else os.path.basename(zp)[:20]
        z=zipfile.ZipFile(zp)
        # group prompt_logs/<instance>/NN_*.json by instance
        byinst=defaultdict(list)
        for n in z.namelist():
            if "/prompt_logs/" in n and n.endswith(".json"):
                inst=n.split("/prompt_logs/")[1].split("/")[0]
                byinst[inst].append(n)
        insts=list(byinst)
        if sample_instances and len(insts)>sample_instances:
            random.Random(42).shuffle(insts); insts=insts[:sample_instances]
        for inst in insts:
            calls=sorted(byinst[inst], key=lambda p: p.rsplit("/",1)[1])
            steps=[]
            for c in calls:
                try: obj=json.loads(z.read(c))
                except: continue
                msgs=obj.get("messages") or []
                comp=obj.get("completion") or ""
                if isinstance(comp,dict): comp=json.dumps(comp)
                isl=sum(ntok(m.get("content") or "") for m in msgs if isinstance(m,dict))
                osl=ntok(comp if isinstance(comp,str) else str(comp))
                actm=re.search(r"\d+_([A-Za-z]+)\.json$", c)
                tool=actm.group(1) if actm else "call"
                steps.append(dict(isl=isl, osl=osl, tool=tool))
            if steps:
                yield dict(steps=steps, reward=None, group=tag)

# tool extractors for plain-text assistant messages
def tool_func(txt):
    m=FUNC_RE.search(txt or "")
    if m: return m.group(1)
    if BASH_HINT.search(txt or ""): return "bash"
    return "none"

FENCE_RE = re.compile(r"```(?:bash)?\s*\n?(.+?)```", re.S)
def tool_aci(txt):
    """SWE-agent classic ACI: command is first token inside a fenced code block."""
    if not txt: return "none"
    m=FENCE_RE.search(txt)
    if m:
        body=m.group(1).strip()
        if body:
            first=body.splitlines()[0].strip()
            t=re.match(r"([A-Za-z_][\w\-]*)", first)
            if t: return t.group(1)
    m2=FUNC_RE.search(txt)
    if m2: return m2.group(1)
    return "none"

# ============================ AGGREGATE + REPORT ============================
def analyze(key, sessions_iter, model_cfg, notes):
    isl_all=[]; osl_all=[]; steps_per=[]; peak_ctx=[]; hits=[]
    prefill_nc_tot=0; prefill_uc_tot=0; compaction_sessions=0
    tools=Counter(); rewards=[]; groups=Counter(); group_ctx=defaultdict(list)
    group_hit=defaultdict(list); group_steps=defaultdict(list)
    nsess=0
    for s in sessions_iter:
        st=s["steps"]; nsess+=1
        steps_per.append(len(st))
        for x in st:
            isl_all.append(x["isl"]); osl_all.append(x["osl"]); tools[x["tool"]]+=1
        c=session_cache(st)
        peak_ctx.append(c["peak_ctx"]); hits.append(c["hit"])
        prefill_nc_tot+=c["prefill_nc"]; prefill_uc_tot+=c["prefill_uc"]
        if c["compactions"]: compaction_sessions+=1
        g=s.get("group","all"); groups[g]+=1
        group_ctx[g].append(c["peak_ctx"]); group_hit[g].append(c["hit"]); group_steps[g].append(len(st))
        if s.get("reward") is not None:
            try: rewards.append(float(s["reward"]))
            except: pass
    bpt=kv_bytes_per_token(model_cfg["layers"], model_cfg["kv_heads"], model_cfg["head_dim"])
    write_md(key, dict(nsess=nsess, isl=summ(isl_all), osl=summ(osl_all),
        steps=summ(steps_per), peak=summ(peak_ctx), hit=summ(hits),
        prefill_nc=prefill_nc_tot, prefill_uc=prefill_uc_tot,
        compaction_sessions=compaction_sessions, tools=tools, rewards=rewards,
        groups=groups, group_ctx=group_ctx, group_hit=group_hit, group_steps=group_steps,
        bpt=bpt, model=model_cfg), notes)
    # machine-readable stats consumed by build_dashboard.py
    def gmean(l): return sum(l)/len(l) if l else 0
    resolved=[r for r in rewards if r>=1]
    stats=dict(key=key, nsess=nsess, nsteps=len(isl_all),
        isl=summ(isl_all), osl=summ(osl_all), steps=summ(steps_per), peak=summ(peak_ctx),
        hit=summ(hits), hit_agg=(1-prefill_uc_tot/prefill_nc_tot if prefill_nc_tot else 0),
        compaction_sessions=compaction_sessions,
        reward=(dict(mean=gmean(rewards), resolved_frac=len(resolved)/len(rewards), n=len(rewards)) if rewards else None),
        tools=[[t,c] for t,c in tools.most_common(12)],
        native_model=model_cfg["name"], native_kib_per_tok=bpt/1024,
        groups={g:dict(n=groups[g], steps=gmean(group_steps[g]), ctx=gmean(group_ctx[g]),
                       hit=gmean(group_hit[g])) for g in groups})
    dst_dir=os.path.join(D, key); os.makedirs(dst_dir, exist_ok=True)
    json.dump(stats, open(os.path.join(dst_dir, f"{key}.stats.json"),"w"), indent=1)
    print(f"[{key}] {nsess} sessions, {len(isl_all)} steps analyzed -> {D}/{key}/ (.md + .stats.json)")

def fmt(n):
    if n is None: return "-"
    if n>=1e9: return f"{n/1e9:.2f}B"
    if n>=1e6: return f"{n/1e6:.2f}M"
    if n>=1e3: return f"{n/1e3:.1f}k"
    return f"{n:.0f}" if n==int(n) else f"{n:.1f}"
def gib(b): return b/(1024**3)
def mib(b): return b/(1024**2)

def write_md(key, a, notes):
    L=[]
    L.append(f"# {key} — trace analysis\n")
    L.append("## About this dataset\n")
    L.append(notes.strip()+"\n")
    isl,osl,stp,pk,ht=a["isl"],a["osl"],a["steps"],a["peak"],a["hit"]
    L.append(f"**Sessions analyzed:** {a['nsess']:,}  ·  **assistant steps (LLM calls):** {isl['n']:,}\n")
    if a["rewards"]:
        r=a["rewards"]; L.append(f"**Resolve/reward:** mean {sum(r)/len(r):.3f} over {len(r):,} scored sessions ({100*sum(1 for x in r if x>=1)/len(r):.1f}% fully resolved)\n")

    L.append("\n## Conversation shape\n")
    L.append("| metric | mean | p50 | p90 | p95 | p99 | max |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    L.append(f"| steps / session | {stp['mean']:.1f} | {fmt(stp['p50'])} | {fmt(stp['p90'])} | {fmt(stp['p95'])} | {fmt(stp['p99'])} | {fmt(stp['mx'])} |")

    L.append("\n## Per-step ISL / OSL (tokens)\n")
    L.append("| metric | mean | p50 | p90 | p95 | p99 | max |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    L.append(f"| **ISL** (input / call) | {fmt(isl['mean'])} | {fmt(isl['p50'])} | {fmt(isl['p90'])} | {fmt(isl['p95'])} | {fmt(isl['p99'])} | {fmt(isl['mx'])} |")
    L.append(f"| **OSL** (output / call) | {fmt(osl['mean'])} | {fmt(osl['p50'])} | {fmt(osl['p90'])} | {fmt(osl['p95'])} | {fmt(osl['p99'])} | {fmt(osl['mx'])} |")

    L.append("\n## Prefix-cache modeling\n")
    nc,uc=a["prefill_nc"],a["prefill_uc"]
    agg_hit=1-uc/nc if nc else 0
    L.append(f"Computed per session from the exact cumulative-context token counts. "
             f"cISL (cache hit) = prior cumulative prompt; nISL (new) = growth since last call; "
             f"context shrink ⇒ counted as a **compaction** (cache reset).\n")
    L.append(f"- **Aggregate prefix-cache hit ratio** = `1 − Σ nISL / Σ ISL` = **{100*agg_hit:.1f}%** "
             f"(Σ input {fmt(nc)} tok → only {fmt(uc)} tok actually need prefill).")
    L.append(f"- **Per-session hit ratio:** mean {100*ht['mean']:.1f}% · p50 {100*ht['p50']:.1f}% · p90 {100*ht['p90']:.1f}%")
    comp_rate=a['compaction_sessions']/max(a['nsess'],1)
    L.append(f"- **Sessions with a compaction / context shrink:** {a['compaction_sessions']:,} "
             f"({100*comp_rate:.1f}%)\n")
    if comp_rate>0.5:
        L.append(f"> ⚠️ **Non-append-only agent.** {100*comp_rate:.0f}% of sessions shrink context between "
                 f"calls — this agent rebuilds prompts per action rather than appending, so the append-only "
                 f"cISL arithmetic is only approximate here. The high shrink rate is itself the finding: this "
                 f"architecture is **prefix-cache-hostile** vs. append-only ReAct agents (90%+ hit).\n")

    L.append("### Prefix-cache *size* (peak context per session)\n")
    m=a["model"]; bpt=a["bpt"]
    L.append(f"KV-cache footprint uses **{m['name']}** config "
             f"(layers={m['layers']}, kv_heads={m['kv_heads']}, head_dim={m['head_dim']}, fp16) → "
             f"**{bpt/1024:.0f} KiB/token** (= 2·L·H·d·2B), vLLM block = {BLOCK} tok = {bpt*BLOCK/1024/1024:.2f} MiB.\n")
    L.append("| peak-context percentile | tokens | KV blocks (÷16) | KV cache size |")
    L.append("|---|--:|--:|--:|")
    for lab,val in [("mean",pk['mean']),("p50",pk['p50']),("p90",pk['p90']),("p95",pk['p95']),("p99",pk['p99']),("max",pk['mx'])]:
        L.append(f"| {lab} | {fmt(val)} | {fmt(math.ceil(val/BLOCK))} | {mib(val*bpt):.0f} MiB |")

    L.append("\n## Tool / action mix\n")
    tot=sum(a["tools"].values())
    L.append("| tool | calls | share |")
    L.append("|---|--:|--:|")
    for t,c in a["tools"].most_common(12):
        L.append(f"| `{t}` | {c:,} | {100*c/tot:.1f}% |")

    if len([g for g in a["groups"] if g])>1:
        L.append("\n## By group (context budget / attempt / release)\n")
        L.append("| group | sessions | mean steps | mean peak-ctx (tok) | mean cache-hit |")
        L.append("|---|--:|--:|--:|--:|")
        for g,n in a["groups"].most_common():
            gc=a["group_ctx"][g]; gh=a["group_hit"][g]; gs=a["group_steps"][g]
            L.append(f"| {g} | {n:,} | {sum(gs)/len(gs):.1f} | {fmt(sum(gc)/len(gc))} | {100*sum(gh)/len(gh):.1f}% |")

    # write the report inside the dataset's own trace dir
    dst_dir=os.path.join(D, key); os.makedirs(dst_dir, exist_ok=True)
    open(os.path.join(dst_dir, f"{key}.md"),"w").write("\n".join(L)+"\n")

# ============================ DISPATCH ============================
def main():
    key=sys.argv[1]
    if key=="axt":
        analyze("axt-swe-agent-lm-32b", load_axt(700), QWEN32B,
          "**`AxT-dev/swe-agent-lm-32b-r2e-gym-trajectories`** — trajectories from running **SWE-agent-LM-32B** "
          "(the open, Qwen2.5-32B-based SWE agent model from the SWE-agent/SWE-smith line) against **R2E-Gym**, a "
          "library of executable real-world software-engineering task environments (the same lineage as DeepSWE, "
          "which trains on a 4.5K-problem R2E-Gym subset). The agent runs a ReAct loop — read the issue, explore "
          "`/testbed`, edit files, run tests — until it submits a patch. This release repeats the run at **4 context "
          "budgets (16K / 32K / 98K / 131K) × 2 attempts**, which is what makes it uniquely useful for studying how "
          "context length shapes cISL and cache behavior.\n\n"
          "**Format:** JSONL, one trajectory per line (`trajectory_steps[]` with `thought`/`action`/`observation`). "
          "**Token counts are MEASURED** here (`token_usage_prompt`/`token_usage_completion`), not tiktoken estimates. "
          "Also carries per-step timing (`llm_exec_time`, `env_exec_time`) and a final `reward`. Sampled 700 sessions/file.")
    elif key=="nebius":
        analyze("nebius-swe-agent",
          load_msglist("nebius", f"{D}/nebius-swe-agent/data/*.parquet","trajectory",
            {"ai":"assistant","user":"user","system":"system"}, tool_aci, sample=2500),
          QWEN7B,
          "**`nebius/SWE-agent-trajectories`** — a large (~80K-trajectory) corpus of **SWE-agent** runs released by "
          "Nebius, collected to train/fine-tune open SWE agents (their SWE-agent-7B/32B efforts). Uses the classic "
          "**SWE-agent ACI** (Agent-Computer Interface): the model emits a short discussion followed by a single "
          "command in a code fence — `edit`, `open`, `search_dir`, `create`, `python`, `ls` — and the environment "
          "returns the result. Each row is one issue-resolution session with a `generated_patch` and `exit_status`.\n\n"
          "**Format:** parquet ×12, `trajectory[]` of `{role: system/user/ai, text}`. No token counts, so ISL/OSL are "
          "estimated with tiktoken `o200k_base`. Sampled 2,500 sessions. (KV sizing uses the 7B config, matching the "
          "smaller models these traces target.)")
    elif key=="r2egym-sft":
        analyze("r2egym-sft",
          load_msglist("r2egym-sft", f"{D}/r2egym-sft/data/*.parquet","messages",
            {"assistant":"assistant","user":"user","system":"system"}, tool_func, sample=None),
          QWEN32B,
          "**`R2E-Gym/R2EGym-SFT-Trajectories`** — the **supervised fine-tuning** set from **R2E-Gym** (COLM 2025), "
          "the project that procedurally generates executable SWE environments and trains open-weight SWE agents. "
          "These are successful/curated agent trajectories used as SFT data to teach the ReAct tool-use loop. Actions "
          "use the `<function=name>…</function>` tool-call convention (`file_editor`, `execute_bash`, `search`, `finish`).\n\n"
          "**Format:** parquet, `messages[]` of `{role, content}` (OpenAI-style). No token counts → tiktoken "
          "`o200k_base`. All 3,231 sessions analyzed.")
    elif key=="r2egym-verifier":
        analyze("r2egym-verifier",
          load_msglist("r2egym-verifier", f"{D}/r2egym-verifier/data/*.parquet","messages",
            {"assistant":"assistant","user":"user","system":"system"}, tool_func, sample=3000, reward_col="rewards"),
          QWEN32B,
          "**`R2E-Gym/R2EGym-Verifier-Trajectories`** — data for training R2E-Gym's **verifier** (an outcome reward "
          "model used for best-of-n selection, one of the paper's headline results). Each row feeds a *whole* agent "
          "trajectory to a judge that emits a single short verdict, with a `rewards` label. So unlike the others this "
          "is **not a multi-turn agent** — it's a **one-shot classification** workload: exactly 1 LLM call/session, "
          "large ISL (reads the full trajectory ~15k tok), tiny OSL (~9 tok), and no prefix reuse.\n\n"
          "**Format:** parquet, `messages[]` + `docker_images` + `rewards`. Tokens via tiktoken `o200k_base`. "
          "Sampled 3,000 sessions. Included as a deliberate *contrast* to the agent traces.")
    elif key=="openhands":
        analyze("swegym-openhands",
          load_msglist("openhands", f"{D}/swegym-openhands/data/*.parquet","messages",
            {"assistant":"assistant","user":"user","system":"system","tool":"tool"}, tool_func,
            sample=2500, reward_col=None),
          QWEN32B,
          "**`SWE-Gym/OpenHands-Sampled-Trajectories`** — sampled rollouts of the **OpenHands** agent (the CodeAct "
          "architecture) on **SWE-Gym** (ICML 2025), the first publicly-usable environment of 2.4K real SWE tasks with "
          "executable test harnesses. Uses native OpenAI **tool-calling** (`str_replace_editor`, `execute_bash`, "
          "`finish`) rather than text-parsed actions, with explicit `tool` role messages carrying results. Each row "
          "has a `resolved` flag and full `test_result`. OpenHands can also condense/summarize long histories, which "
          "is why a few sessions show non-monotonic context.\n\n"
          "**Format:** parquet ×3, rich `messages[]` (`role`, `content`, `tool_calls[]`, `tool_call_id`). Tokens via "
          "tiktoken `o200k_base`; **OSL includes tool-call arguments**. Sampled 2,500 sessions.")
    elif key=="moatless":
        analyze("swegym-moatless", load_moatless(sample_instances=None), QWEN32B,
          "**`SWE-Gym/MoatlessTools-Sampled-Trajectories`** — rollouts of the **Moatless Tools** agent on SWE-Gym "
          "(SWE-Gym project, Qwen 7B/32B releases). Moatless is architecturally different from the others: a "
          "**structured state machine** that moves through explicit phases — `SearchCode` → `IdentifyCode` → "
          "`DecideRelevance` → `PlanToCode` → `EditCode` — **rebuilding a focused prompt for each phase** instead of "
          "appending to one growing transcript. That design is the reason its prefix-cache hit is far lower than the "
          "append-only ReAct/CodeAct agents (see below) — a clean natural experiment in how agent architecture drives "
          "GPU cost.\n\n"
          "**Format:** zip dumps of full eval runs; I sampled 2 bench-lite 32B releases (`zero`, `it1-v2`). Each "
          "`prompt_logs/<instance>/NN_<Action>.json` = one LLM call (`messages` + `completion`), so ISL/OSL are read "
          "per call and tokenized with tiktoken `o200k_base`. Grouped by release.")
    elif key=="sharegpt":
        analyze("sharegpt-v3", load_sharegpt(sample=4000), LLAMA8B,
          "**`ShareGPT_V3_unfiltered_cleaned_split`** — ~94K real human↔ChatGPT **web-chat** conversations shared to "
          "sharegpt.com; the canonical dataset vLLM-style benchmarks (and the *Frontier* paper's SharedGPT trace) replay "
          "for realistic serving workloads. Included here as a **non-agentic baseline**: general chat + coding help, "
          "**no tools, no environment**. Modeled in the multi-turn serving view — each assistant (`gpt`) turn is one LLM "
          "call whose ISL is the accumulated conversation.\n\n"
          "**Format:** JSON list of `{id, conversations:[{from, value}]}`. Tokens via tiktoken `o200k_base`. Sampled 4,000 "
          "conversations. KV sizing uses **Llama-3.1-8B** (a typical chat-serving model; model-agnostic data). "
          "Contrast the others: chat is decode-balanced and short-session, agents are prefill-heavy and long.")
    else:
        print("unknown key"); sys.exit(1)

if __name__=="__main__":
    main()
