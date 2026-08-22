import { motion } from "framer-motion";

import { CpuIcon, GlobeIcon, LayersIcon } from "../ui/icons.jsx";

const MODES = [
  {
    id: "combined",
    icon: LayersIcon,
    label: "Combined",
    desc: "Both engines",
  },
  {
    id: "ml",
    icon: CpuIcon,
    label: "ML only",
    desc: "Instant, offline",
  },
  {
    id: "ai",
    icon: GlobeIcon,
    label: "AI only",
    desc: "Web fact-check",
  },
];

/**
 * Segmented control for picking the analysis engine.
 * The active pill slides between tabs via layoutId.
 */
export default function ModeSelector({ mode, onChange, disabled }) {
  return (
    <div
      role="tablist"
      aria-label="Analysis mode"
      className="relative grid grid-cols-3 gap-1.5 rounded-2xl border border-line bg-surface2 p-1.5"
    >
      {MODES.map((m) => {
        const active = mode === m.id;
        return (
          <button
            key={m.id}
            role="tab"
            aria-selected={active}
            type="button"
            disabled={disabled}
            onClick={() => onChange(m.id)}
            className={`relative flex items-center justify-center gap-2 rounded-xl px-2 py-3 text-sm font-semibold transition-colors duration-200 disabled:opacity-50 ${
              active ? "text-ink" : "text-muted hover:text-ink"
            }`}
          >
            {active && (
              <motion.span
                layoutId="mode-pill"
                transition={{ type: "spring", stiffness: 400, damping: 32 }}
                className="absolute inset-0 rounded-xl bg-surface shadow-card"
              />
            )}
            <m.icon size={16} className={`relative z-10 ${active ? "text-accent" : ""}`} />
            <span className="relative z-10 hidden sm:inline">{m.label}</span>
            {/* Compact label on tiny screens */}
            <span className="relative z-10 sm:hidden">{m.label.split(" ")[0]}</span>
          </button>
        );
      })}
    </div>
  );
}
