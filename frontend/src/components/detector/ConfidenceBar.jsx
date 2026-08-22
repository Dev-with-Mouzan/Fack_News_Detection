import { motion } from "framer-motion";

/** Animated horizontal confidence meter. */
export default function ConfidenceBar({ pct, cls }) {
  const fillColor = { real: "bg-ok", fake: "bg-bad", uncertain: "bg-warn" }[cls];

  return (
    <section aria-label="Overall confidence">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-xs font-bold uppercase tracking-[0.14em] text-faint">
          Overall confidence
        </span>
        <span className="font-mono text-lg font-semibold tabular-nums">{pct}%</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-surface2">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1], delay: 0.15 }}
          className={`h-full rounded-full ${fillColor}`}
        />
      </div>
    </section>
  );
}
