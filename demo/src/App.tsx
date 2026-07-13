import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ACE_PLAYBOOK,
  BENCHMARK,
  GEPA_TRAJECTORY,
  TEST_ACC,
} from "./data.ts";

// The replay timeline: one tick reveals the next training event. GEPA rewrites
// its instruction; ACE appends the next playbook bullet. Baseline never moves.
const TICKS = Math.max(GEPA_TRAJECTORY.length, ACE_PLAYBOOK.length + 1);

// How far each optimizer has progressed at tick `t` (0..TICKS-1 -> 0..1).
const frac = (t: number) => (TICKS <= 1 ? 1 : t / (TICKS - 1));

// Accuracy shown at tick t: interpolate seed -> measured final along progress.
function accAt(method: "gepa" | "ace", t: number): number | null {
  const seed = 0.652; // shared un-optimized program (test)
  const final = TEST_ACC[method];
  if (final === null) return null;
  return seed + (final - seed) * frac(t);
}

const SPEEDS = [
  { label: "0.5×", ms: 1500 },
  { label: "1×", ms: 800 },
  { label: "2×", ms: 400 },
  { label: "4×", ms: 180 },
];

export default function App() {
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1); // index into SPEEDS
  const timer = useRef<number | null>(null);

  const done = tick >= TICKS - 1;

  const stop = useCallback(() => {
    if (timer.current !== null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
    setPlaying(false);
  }, []);

  const play = useCallback(() => {
    if (done) setTick(0);
    setPlaying(true);
  }, [done]);

  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => {
      setTick((t) => {
        if (t >= TICKS - 1) {
          if (timer.current !== null) window.clearInterval(timer.current);
          timer.current = null;
          setPlaying(false);
          return t;
        }
        return t + 1;
      });
    }, SPEEDS[speed].ms);
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current);
    };
  }, [playing, speed]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>
            <span className="accent">ACE</span> vs GEPA vs Baseline
          </h1>
          <p>
            Replay how three DSPy optimizers train the <em>same</em> program on{" "}
            {BENCHMARK.name} ({BENCHMARK.subtitle}). Baseline never trains; GEPA
            rewrites its instruction; ACE grows a playbook.
          </p>
          <div className="bench">
            <span>
              <b>model</b> {BENCHMARK.model}
            </span>
            <span>
              <b>test</b> {BENCHMARK.sizes.test} rows
            </span>
            <span>
              <b>metric</b> tag-F1
            </span>
          </div>
        </div>

        <div className="controls">
          <button
            className={"play" + (playing ? " running" : "")}
            onClick={playing ? stop : play}
          >
            <span className="glyph">{playing ? "❚❚" : done ? "↻" : "▶"}</span>
            {playing ? "Pause" : done ? "Replay" : tick === 0 ? "Play" : "Resume"}
          </button>
          <button className="btn" onClick={() => { stop(); setTick(0); }}>
            Reset
          </button>
          <div className="speed" role="group" aria-label="Speed">
            {SPEEDS.map((s, i) => (
              <button
                key={s.label}
                className={i === speed ? "on" : ""}
                onClick={() => setSpeed(i)}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* scrubber: fast-forward / rewind the whole training */}
      <div className="scrub">
        <input
          type="range"
          min={0}
          max={TICKS - 1}
          value={tick}
          onChange={(e) => { stop(); setTick(Number(e.target.value)); }}
          aria-label="Training step"
        />
        <span className="stepno">
          step {tick} / {TICKS - 1}
        </span>
      </div>

      <div className="columns">
        <BaselineCol />
        <GepaCol tick={tick} />
        <AceCol tick={tick} />
      </div>

      <footer>
        real numbers — DeepSeek-V3.1 on the ACE paper's FiNER benchmark ·{" "}
        <a href="https://github.com/ppaiboi/ace-optimizer">ppaiboi/ace-optimizer</a>
      </footer>
    </div>
  );
}

/* -- columns --------------------------------------------------------------- */

function Acc({ value, mc }: { value: number | null; mc: string }) {
  return (
    <>
      <div className="acc">
        {value === null ? (
          <span className="num pending">running…</span>
        ) : (
          <span className="num" style={{ color: mc }}>
            {(value * 100).toFixed(1)}
            <span style={{ fontSize: 15 }}>%</span>
          </span>
        )}
      </div>
      <div className={"accbar" + (value === null ? " pending" : "")}>
        {value !== null && (
          <span style={{ width: `${value * 100}%`, background: mc }} />
        )}
      </div>
    </>
  );
}

function BaselineCol() {
  return (
    <section className="col baseline">
      <div className="col-head">
        <div className="col-title">
          <span className="name">Baseline</span>
          <span className="tag">no training</span>
        </div>
        <p className="col-sub">The raw program, run as-is.</p>
        <Acc value={TEST_ACC.baseline} mc="var(--baseline)" />
      </div>
      <div className="artifact">
        <p className="artifact-label">program</p>
        <pre className="code">
{`dspy.Predict(
  "question -> answer"
)`}
        </pre>
        <p className="empty">Nothing to optimize — this is the control.</p>
      </div>
    </section>
  );
}

function GepaCol({ tick }: { tick: number }) {
  // reveal accepted instructions as training proceeds
  const shown = Math.min(
    GEPA_TRAJECTORY.length,
    Math.max(1, Math.round(frac(tick) * GEPA_TRAJECTORY.length)),
  );
  const current = GEPA_TRAJECTORY[shown - 1];
  return (
    <section className="col gepa">
      <div className="col-head">
        <div className="col-title">
          <span className="name">GEPA</span>
          <span className="tag">evolves instruction</span>
        </div>
        <p className="col-sub">Reflects on failures, rewrites the prompt.</p>
        <Acc value={accAt("gepa", tick)} mc="var(--gepa)" />
      </div>
      <div className="artifact">
        <p className="artifact-label">
          <span>current instruction</span>
          <span>iter {current.iter}</span>
        </p>
        <div className="instruction">{current.instruction}</div>
        <div className="iterlog">
          {GEPA_TRAJECTORY.slice(0, shown).map((s) => (
            <div
              key={s.iter}
              className={"iterrow" + (s === current ? " hit" : "")}
            >
              <span>iter {String(s.iter).padStart(2, "0")}</span>
              <span className="v">val {(s.val * 100).toFixed(1)}%</span>
              <span className="up">↑ accepted</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function AceCol({ tick }: { tick: number }) {
  const shown = Math.min(
    ACE_PLAYBOOK.length,
    Math.round(frac(tick) * ACE_PLAYBOOK.length),
  );
  const bullets = useMemo(() => ACE_PLAYBOOK.slice(0, shown), [shown]);

  // group revealed bullets by section, preserving first-seen order
  const groups = useMemo(() => {
    const g: { section: string; items: typeof bullets }[] = [];
    for (const b of bullets) {
      let cur = g.find((x) => x.section === b.section);
      if (!cur) {
        cur = { section: b.section, items: [] };
        g.push(cur);
      }
      cur.items.push(b);
    }
    return g;
  }, [bullets]);

  return (
    <section className="col ace">
      <div className="col-head">
        <div className="col-title">
          <span className="name">ACE</span>
          <span className="tag">grows playbook</span>
        </div>
        <p className="col-sub">
          Accumulates reusable tactics — {shown}/{ACE_PLAYBOOK.length} bullets.
        </p>
        <Acc value={accAt("ace", tick)} mc="var(--ace)" />
      </div>
      <div className="artifact">
        <p className="artifact-label">playbook.md</p>
        {groups.length === 0 ? (
          <p className="empty">empty — press Play to build it</p>
        ) : (
          <div className="bullets">
            {groups.map((g) => (
              <div key={g.section}>
                <div className="section-head">## {g.section}</div>
                {g.items.map((b) => (
                  <div className="bullet" key={b.id}>
                    <div className="bid">
                      <span>[{b.id}]</span>
                      <span className="ctr">helpful+</span>
                    </div>
                    <div className="btext">{b.content}</div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
