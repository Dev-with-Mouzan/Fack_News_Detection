import { AnimatePresence, motion } from "framer-motion";

import { CheckIcon, SpinnerIcon } from "../ui/icons.jsx";

function StepIndicator({ state }) {
  if (state === "done") {
    return (
      <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-ok text-white">
        <CheckIcon size={11} />
      </span>
    );
  }
  if (state === "active") {
    return (
      <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full text-accent">
        <SpinnerIcon size={16} />
      </span>
    );
  }
  return <span className="h-6 w-6 shrink-0 rounded-full border-2 border-line" />;
}

/**
 * Vertical progress list. Each step is idle → active → done.
 * @param {{steps: {title:string, desc:string}[], current: number}} props
 * current = index of the active step; steps before it are "done".
 */
export default function LoadingSteps({ steps, current }) {
  return (
    <motion.section
      aria-label="Analysis progress"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      className="mt-5 space-y-2"
    >
      {steps.map((step, i) => {
        const state = i < current ? "done" : i === current ? "active" : "idle";
        return (
          <motion.div
            key={step.title}
            layout
            className={`flex items-center gap-3.5 rounded-xl border p-4 transition-all duration-300 ${
              state === "active"
                ? "border-accent/60 bg-surface shadow-glow"
                : state === "done"
                  ? "border-ok/25 bg-surface opacity-60"
                  : "border-line bg-surface opacity-40"
            }`}
          >
            <StepIndicator state={state} />
            <div>
              <p className={`text-sm font-semibold ${state === "idle" ? "text-faint" : "text-ink"}`}>
                {step.title}
              </p>
              <p className="text-xs text-muted">{step.desc}</p>
            </div>
          </motion.div>
        );
      })}
    </motion.section>
  );
}
