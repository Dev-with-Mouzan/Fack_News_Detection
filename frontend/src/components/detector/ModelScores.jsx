import { VERDICT_STYLES, classify } from "../../utils/verdicts.jsx";
import { CpuIcon, GlobeIcon } from "../ui/icons.jsx";

function ScoreCard({ icon: Icon, header, label, sub }) {
  const cls = classify(label);
  const style = VERDICT_STYLES[cls];

  return (
    <div className="rounded-xl border border-line bg-surface2 p-4">
      <p className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-faint">
        <Icon size={13} />
        {header}
      </p>
      <p className={`mt-1.5 text-xl font-extrabold tracking-tight ${style.text}`}>{label}</p>
      <p className="mt-0.5 font-mono text-xs tabular-nums text-faint">{sub}</p>
    </div>
  );
}

/** Side-by-side per-engine scores (combined mode only). */
export default function ModelScores({ ml, ai }) {
  if (!ml && !ai) return null;

  const pct = (c) => {
    const v = Math.round((c || 0) * 100);
    return v > 0 ? `${v}%` : "n/a";
  };

  return (
    <section aria-label="Per-engine results" className="grid gap-3 sm:grid-cols-2">
      {ml && (
        <ScoreCard
          icon={CpuIcon}
          header="ML model"
          label={ml.label}
          sub={`${ml.model} · ${pct(ml.confidence)}`}
        />
      )}
      {ai && (
        <ScoreCard
          icon={GlobeIcon}
          header="AI verification"
          label={ai.label}
          sub={`${pct(ai.confidence)} confidence`}
        />
      )}
    </section>
  );
}
