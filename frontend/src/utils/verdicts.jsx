// Normalize the three backend response shapes into one display model,
// and map verdicts to semantic classes used across result components.

export function classify(verdict) {
  const v = String(verdict || "").toLowerCase();
  if (v === "real" || v === "true") return "real";
  if (v === "fake" || v === "false") return "fake";
  return "uncertain";
}

/**
 * @param {"combined"|"ml"|"ai"} mode
 * @param {object} data raw API response
 * @returns unified: { mode, label, cls, confidence, explanation, sources, ml, ai }
 */
export function normalizeResult(mode, data) {
  if (mode === "combined") {
    return {
      mode,
      label: data.final_verdict || "Uncertain",
      cls: classify(data.final_verdict),
      confidence: data.confidence ?? 0,
      explanation: data.explanation || "",
      sources: data.sources || [],
      ml: data.ml_prediction
        ? {
            label: data.ml_prediction.label,
            confidence: data.ml_prediction.confidence ?? 0,
            model: data.ml_prediction.model || "XGBoost",
          }
        : null,
      ai: data.ai_verification
        ? {
            label: data.ai_verification.verdict,
            confidence: data.ai_verification.confidence ?? 0,
          }
        : null,
    };
  }

  if (mode === "ml") {
    return {
      mode,
      label: data.label || "Uncertain",
      cls: classify(data.label),
      confidence: data.confidence ?? 0,
      explanation: data.explanation || "",
      sources: [],
      ml: { label: data.label, confidence: data.confidence ?? 0, model: data.model || "XGBoost" },
      ai: null,
    };
  }

  // ai
  return {
    mode,
    label: data.verdict || "Uncertain",
    cls: classify(data.verdict),
    confidence: data.confidence ?? 0,
    explanation: data.explanation || "",
    sources: data.sources || [],
    ml: null,
    ai: { label: data.verdict, confidence: data.confidence ?? 0 },
  };
}

// Tailwind class bundles per verdict class.
export const VERDICT_STYLES = {
  real: {
    text: "text-ok",
    chip: "bg-ok/10 text-ok border-ok/25",
    bannerBg: "bg-ok/[0.07]",
    iconBg: "bg-ok/12 text-ok border-ok/25",
    bar: "bg-ok",
  },
  fake: {
    text: "text-bad",
    chip: "bg-bad/10 text-bad border-bad/25",
    bannerBg: "bg-bad/[0.07]",
    iconBg: "bg-bad/12 text-bad border-bad/25",
    bar: "bg-bad",
  },
  uncertain: {
    text: "text-warn",
    chip: "bg-warn/10 text-warn border-warn/25",
    bannerBg: "bg-warn/[0.07]",
    iconBg: "bg-warn/12 text-warn border-warn/25",
    bar: "bg-warn",
  },
};

/** Render backend markdown bold (**text**) as <strong> segments. */
export function renderBold(text) {
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i} className="font-semibold text-ink">
        {part.slice(2, -2)}
      </strong>
    ) : (
      part
    )
  );
}
