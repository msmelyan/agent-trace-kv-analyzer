const thresholds = [
  { label: "32K", requests: 90.1, sessions: 100 },
  { label: "64K", requests: 75.0, sessions: 98.2 },
  { label: "128K", requests: 53.3, sessions: 79.1 },
  { label: "256K", requests: 30.2, sessions: 44.0 },
  { label: "512K", requests: 11.2, sessions: 17.8 },
  { label: "1M", requests: 0, sessions: 0 },
];

const reuseByTurn = [
  { label: "1", value: 24.8 },
  { label: "2", value: 80.1 },
  { label: "3–4", value: 92.8 },
  { label: "5–8", value: 94.9 },
  { label: "9–16", value: 96.9 },
  { label: "17–32", value: 98.1 },
  { label: "33–64", value: 98.8 },
  { label: "65+", value: 99.1 },
];

const representatives = [
  ["7e370a…947ac", "Short-context", "Near p10 session maximum context"],
  ["a26302…170", "Median-context", "Near corpus median maximum context"],
  ["4e6875…0d79", "Long-context", "Near p90 session maximum context"],
  ["61c4d4…07dc7", "Context ceiling", "Largest observed context: 996,579 tokens"],
  ["5b6e13…9004", "Most requests", "4,114 model requests in one session"],
  ["439e5f…6903f", "Parallel fan-out", "Strongest explicit subagent fan-out"],
  ["5cbd63…f9dc", "Peak concurrency", "23 simultaneous model requests"],
  ["851f59…a9ab", "Highest reuse", "Highest weighted theoretical prefix reuse"],
  ["04dba6…5c48", "Lowest reuse", "Strongest context discontinuity"],
];

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <p className="eyebrow">{children}</p>;
}

export default function Home() {
  return (
    <main>
      <nav className="nav shell" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="AgentX Trace Atlas home">
          <span className="brandMark" aria-hidden="true">AX</span>
          <span>Trace Atlas</span>
        </a>
        <div className="navLinks">
          <a href="#shape">Traffic shape</a>
          <a href="#reuse">KV reuse</a>
          <a href="#agents">Subagents</a>
          <a href="#method">Method</a>
        </div>
      </nav>

      <header className="hero shell" id="top">
        <div className="heroCopy">
          <SectionLabel>SemiAnalysis AgentX · Claude Code traces</SectionLabel>
          <h1>A realistic coding-agent workload is <em>long, iterative, and branching.</em></h1>
          <p className="lede">
            A reproducible reconstruction of 393 sessions and 98,827 model requests,
            grounded in AIPerf&apos;s WekaTrace semantics.
          </p>
          <div className="heroActions">
            <a className="button primary" href="#findings">Explore findings <span>↓</span></a>
            <a className="button quiet" href="https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/tree/main">Source corpus ↗</a>
          </div>
        </div>
        <div className="heroFigure" aria-label="Abstract prefix reuse and subagent branching diagram">
          <div className="lane parentLane"><span className="node" /><i className="reuse r1" /><i className="reuse r2" /><i className="reuse r3" /><i className="reuse r4" /><i className="fresh f1" /><i className="fresh f2" /></div>
          <div className="branch b1" />
          <div className="branch b2" />
          <div className="lane childLane one"><span className="node" /><i className="reuse r1" /><i className="reuse r2" /><i className="reuse r3" /><i className="fresh f1" /></div>
          <div className="lane childLane two"><span className="node" /><i className="reuse r1" /><i className="reuse r2" /><i className="reuse r3" /><i className="reuse r4" /><i className="fresh f1" /><i className="fresh f2" /></div>
          <div className="figureLegend"><span><b className="teal" /> reusable prefix</span><span><b className="coral" /> new KV</span></div>
        </div>
      </header>

      <section className="statRail" id="findings">
        <div className="shell stats">
          <div><strong>142K</strong><span>median input tokens</span></div>
          <div><strong>444</strong><span>median output tokens</span></div>
          <div><strong>98.3%</strong><span>aggregate theoretical prefix</span></div>
          <div><strong>42.5%</strong><span>requests from subagents</span></div>
          <div><strong>10.6d</strong><span>longest session</span></div>
        </div>
      </section>

      <section className="section shell" id="shape">
        <div className="sectionIntro">
          <div>
            <SectionLabel>01 · Traffic shape</SectionLabel>
            <h2>Prefill dominates the token shape.</h2>
          </div>
          <p>Median input is 320× median output. The p99 request carries 863K input tokens but generates 9.3K output tokens.</p>
        </div>

        <div className="twoCol">
          <article className="panel contextPanel">
            <div className="panelHead"><span>Requests above context threshold</span><small>request % / session %</small></div>
            <div className="thresholdChart" role="img" aria-label="Fraction of requests and sessions exceeding context thresholds">
              {thresholds.map((d) => (
                <div className="thresholdRow" key={d.label}>
                  <span>{d.label}</span>
                  <div className="track"><i className="requestBar" style={{ width: `${d.requests}%` }} /><i className="sessionTick" style={{ left: `${Math.min(d.sessions, 99.5)}%` }} /></div>
                  <b>{d.requests.toFixed(d.requests === 0 ? 0 : 1)}%</b>
                </div>
              ))}
            </div>
            <div className="chartLegend"><span><i className="barKey" /> requests</span><span><i className="tickKey" /> sessions</span></div>
          </article>

          <article className="panel metricMatrix">
            <div className="panelHead"><span>Sequence length percentiles</span><small>tokens</small></div>
            <div className="matrixHead"><span /><span>p50</span><span>p90</span><span>p99</span><span>max</span></div>
            <div className="matrixRow"><b>ISL</b><span>142K</span><span>550K</span><span>863K</span><span>990K</span></div>
            <div className="matrixRow"><b>OSL</b><span>444</span><span>2.7K</span><span>9.3K</span><span>59.9K</span></div>
            <div className="matrixRow"><b>Context</b><span>143K</span><span>551K</span><span>865K</span><span>997K</span></div>
            <div className="ratioCallout"><span>Median input : output</span><strong>320 : 1</strong></div>
          </article>
        </div>

        <div className="imagePanel">
          <img src="/plots/joint_isl_osl.png" alt="Hexbin plot of input and output sequence length showing long inputs and comparatively short outputs" />
          <div className="imageCaption"><span>Joint ISL / OSL distribution</span><small>Log-scaled axes reveal the dominant long-input, short-output regime.</small></div>
        </div>
      </section>

      <section className="section darkBand" id="reuse">
        <div className="shell">
          <div className="sectionIntro">
            <div><SectionLabel>02 · Prefix / KV reuse</SectionLabel><h2>Most input tokens are old news.</h2></div>
            <p>Following WekaTrace, ordered 64-token <code>hash_ids</code> expose leading-prefix identity within each trace.</p>
          </div>
          <div className="reuseGrid">
            <article className="reuseHero">
              <div className="reuseNumber">98.3<span>%</span></div>
              <p>aggregate input tokens theoretically reusable as prefix</p>
              <div className="tokenStrip" aria-label="98.3 percent reusable prefix and 1.7 percent new KV"><i /><b /></div>
              <div className="stripLabels"><span>21.27B reusable</span><span>365.9M new</span></div>
            </article>
            <article className="turnChart">
              <div className="panelHead"><span>Reuse matures with stream age</span><small>mean reusable input</small></div>
              <div className="turnBars">
                {reuseByTurn.map((d) => <div key={d.label}><i style={{ height: `${d.value}%` }}><b>{d.value.toFixed(1)}%</b></i><span>{d.label}</span></div>)}
              </div>
              <p className="axisLabel">turn within reconstructed request stream</p>
            </article>
          </div>
          <aside className="caveat">
            <span>Important distinction</span>
            <p>This is theoretical reuse available from trace structure—not an observed runtime KV-cache hit rate. Residency, eviction, routing, and cache policy are not recorded.</p>
          </aside>
        </div>
      </section>

      <section className="section shell" id="agents">
        <div className="sectionIntro">
          <div><SectionLabel>03 · Agent behavior</SectionLabel><h2>One session is a small concurrent system.</h2></div>
          <p>Explicit SPAWN/JOIN groups coexist with independent chains detected from leading hash-prefix continuity.</p>
        </div>
        <div className="fourUp">
          <article><strong>44.5%</strong><span>sessions use subagents</span><small>175 of 393</small></article>
          <article><strong>1,697</strong><span>explicit subagent groups</span><small>median 16 requests each</small></article>
          <article><strong>10</strong><span>max simultaneous subagents</span><small>p90 = 4</small></article>
          <article><strong>23</strong><span>max in-flight requests</span><small>p99 = 16</small></article>
        </div>

        <figure className="timelinePanel">
          <img src="/plots/representative_complex_timeline.png" alt="Timeline of parent and subagent model requests in a representative high-fan-out session" />
          <figcaption><span>Representative complex session</span><code>439e5f9ed3159bd826529d8ae5bd1536903f</code><small>Parent and child request intervals; shaded regions show explicit subagent lifetimes.</small></figcaption>
        </figure>

        <div className="behaviorGrid">
          <article className="panel">
            <div className="panelHead"><span>Session cadence</span><small>recorded timing</small></div>
            <dl className="kvList">
              <div><dt>Dependent idle time</dt><dd>3.7s <small>p50</small></dd></div>
              <div><dt>Think time</dt><dd>2.3s <small>p50</small></dd></div>
              <div><dt>Arrival gap</dt><dd>9.2s <small>p50</small></dd></div>
              <div><dt>Requests within 10s</dt><dd>52.7%</dd></div>
              <div><dt>Gaps over 60s</dt><dd>11.1%</dd></div>
            </dl>
          </article>
          <article className="panel">
            <div className="panelHead"><span>Session scale</span><small>median / tail</small></div>
            <dl className="kvList">
              <div><dt>Requests</dt><dd>86 <small>p90 618</small></dd></div>
              <div><dt>Parent turns</dt><dd>45 <small>p90 152</small></dd></div>
              <div><dt>Duration</dt><dd>2.8h <small>p90 1.96d</small></dd></div>
              <div><dt>Generated tokens</dt><dd>96K <small>p90 735K</small></dd></div>
              <div><dt>Maximum context</dt><dd>226K <small>p90 696K</small></dd></div>
            </dl>
          </article>
        </div>
      </section>

      <section className="section implicationBand">
        <div className="shell">
          <SectionLabel>04 · Serving implications</SectionLabel>
          <h2>Design for memory continuity, then absorb bursty branches.</h2>
          <div className="implicationGrid">
            <article><span>01</span><h3>Prefix caching</h3><p>The 98.3% theoretical token reuse makes prefix-aware routing and long-lived cache residency first-order concerns.</p></article>
            <article><span>02</span><h3>KV memory</h3><p>Median per-session peak context is 226K; 17.8% of sessions cross 512K. Capacity planning must live in the tail.</p></article>
            <article><span>03</span><h3>Prefill / decode</h3><p>The workload is profoundly prefill-heavy by raw tokens, while decode remains short and latency-sensitive.</p></article>
            <article><span>04</span><h3>Long attention</h3><p>More than half of requests exceed 128K context. Efficient long-context attention is core-path infrastructure.</p></article>
            <article><span>05</span><h3>Batching</h3><p>Burst arrivals and subagent fan-out create short concurrency windows inside otherwise interactive sessions.</p></article>
            <article><span>06</span><h3>P/D disaggregation</h3><p>Asymmetric ISL/OSL and reusable prefill suggest separate resource envelopes—but locality costs must be measured.</p></article>
          </div>
          <div className="specNote"><strong>Speculative decoding:</strong> these traces define traffic shape only. Acceptance rates require separate, model-specific calibration such as SPEED-Bench.</div>
        </div>
      </section>

      <section className="section shell representatives">
        <div className="sectionIntro"><div><SectionLabel>05 · Trace field guide</SectionLabel><h2>Nine sessions worth keeping.</h2></div><p>Selected anchors for short, median, long, fan-out, concurrency, and reuse experiments.</p></div>
        <div className="traceTable" role="table" aria-label="Representative trace identifiers">
          {representatives.map(([id, label, reason]) => <div className="traceRow" role="row" key={id}><code>{id}</code><strong>{label}</strong><span>{reason}</span></div>)}
        </div>
      </section>

      <section className="section methodBand" id="method">
        <div className="shell methodGrid">
          <div><SectionLabel>06 · Reproducibility</SectionLabel><h2>Implementation, not intuition.</h2><p>The analysis first inspects raw entries, then mirrors AIPerf&apos;s chain detector and global prefix prepass. Every input length was verified against its 64-token hash block count.</p></div>
          <dl className="provenance">
            <div><dt>Dataset revision</dt><dd><code>23f152f6f0f9399a85901b89a6458def0ef16729</code></dd></div>
            <div><dt>AIPerf revision</dt><dd><code>c26fe88bd88d32740c83b4637c1e573e06df5ecc</code></dd></div>
            <div><dt>Requests verified</dt><dd>98,827 / 98,827</dd></div>
            <div><dt>Hash scope / block</dt><dd>local / 64 tokens</dd></div>
          </dl>
        </div>
      </section>

      <footer className="footer shell">
        <div><span className="brandMark">AX</span><p>AgentX Trace Atlas<br /><small>Measured August 2026 · timestamps remain session-relative</small></p></div>
        <div className="footerLinks"><a href="https://github.com/ai-dynamo/aiperf">AIPerf source ↗</a><a href="https://docs.nvidia.com/aiperf/dev/benchmark-modes/semi-analysis-agent-x-how-the-benchmark-works-faq">WekaTrace semantics ↗</a></div>
      </footer>
    </main>
  );
}
