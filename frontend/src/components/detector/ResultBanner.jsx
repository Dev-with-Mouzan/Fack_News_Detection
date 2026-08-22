import { motion } from "framer-motion";

import { VERDICT_STYLES } from "../../utils/verdicts.jsx";
import { CheckIcon, HelpCircleIcon, XCircleIcon } from "../ui/icons.jsx";

const ICONS = {
  real: CheckIcon,
  fake: XCircleIcon,
  uncertain: HelpCircleIcon,
};

const SUBTITLES = {
  combined: "Combined engine verdict",
  ml: "XGBoost model prediction",
  ai: "AI verification verdict",
};

/** Color-coded verdict banner at the top of a result card. */
export default function ResultBanner({ cls, label, confidencePct, mode }) {
  const style = VERDICT_STYLES[cls];
  const Icon = ICONS[cls];

  return (
    <div className={`border-b border-line p-6 sm:p-7 ${style.bannerBg}`}>
      <div className="flex items-center gap-4 sm:gap-5">
        <motion.span
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", stiffness: 300, damping: 18, delay: 0.1 }}
          className={`grid h-14 w-14 shrink-0 place-items-center rounded-2xl border ${style.iconBg}`}
        >
          <Icon size={26} />
        </motion.span>
        <div>
          <h2 className={`text-2xl font-extrabold tracking-tight sm:text-3xl ${style.text}`}>
            {label}
          </h2>
          <p className="mt-1 text-sm text-muted">
            {SUBTITLES[mode] || "Verdict"}
            {confidencePct > 0 && (
              <>
                {" · "}
                <span className="font-mono font-semibold tabular-nums">{confidencePct}%</span>{" "}
                confidence
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
