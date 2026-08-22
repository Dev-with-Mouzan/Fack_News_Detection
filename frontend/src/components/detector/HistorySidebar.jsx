import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";

import { useHistoryStore } from "../../hooks/useHistory.jsx";
import { VERDICT_STYLES, classify } from "../../utils/verdicts.jsx";
import { HistoryIcon, TrashIcon, XIcon } from "../ui/icons.jsx";

function HistoryItem({ item }) {
  const navigate = useNavigate();
  const { setOpen } = useHistoryStore();
  const cls = classify(item.result?.label);
  const style = VERDICT_STYLES[cls];
  const conf = Math.round((item.result?.confidence ?? 0) * 100);
  const pillText = conf > 0 ? `${item.result?.label} · ${conf}%` : item.result?.label;

  const open = () => {
    setOpen(false);
    navigate("/detector", { state: { restoreId: item.id } });
  };

  return (
    <button
      type="button"
      onClick={open}
      className="w-full rounded-xl border border-line bg-surface2 p-3.5 text-left transition-all duration-200 hover:border-accent/50 hover:bg-accent/[0.04]"
    >
      <div className="flex items-center justify-between gap-2">
        <span className={`rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide ${style.chip}`}>
          {pillText}
        </span>
        <time className="shrink-0 font-mono text-[11px] text-faint">
          {new Date(item.ts).toLocaleString([], {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })}
        </time>
      </div>
      <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-muted">{item.text}</p>
    </button>
  );
}

/** Slide-over panel listing past predictions (stored in localStorage). */
export default function HistorySidebar() {
  const { items, open, setOpen, clearAll } = useHistoryStore();

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-[60] bg-black/45 backdrop-blur-sm"
          />
          <motion.aside
            role="dialog"
            aria-label="Prediction history"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 320, damping: 34 }}
            className="fixed right-0 top-0 z-[70] flex h-dvh w-[380px] max-w-[92vw] flex-col border-l border-line bg-surface shadow-card-lg"
          >
            <header className="flex items-center justify-between border-b border-line p-5">
              <h3 className="flex items-center gap-2 font-bold tracking-tight">
                <HistoryIcon size={17} className="text-accent" />
                Prediction history
              </h3>
              <div className="flex items-center gap-1">
                {items.length > 0 && (
                  <button
                    type="button"
                    onClick={clearAll}
                    title="Clear all history"
                    className="icon-btn !border-transparent hover:!bg-bad/10 hover:text-bad"
                  >
                    <TrashIcon size={16} />
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => setOpen(false)}
                  aria-label="Close history"
                  className="icon-btn"
                >
                  <XIcon size={16} />
                </button>
              </div>
            </header>

            <div className="scroll-thin flex-1 space-y-2.5 overflow-y-auto p-4">
              {items.length === 0 ? (
                <div className="flex flex-col items-center gap-3 py-20 text-center">
                  <HistoryIcon size={28} className="text-faint" />
                  <p className="text-sm text-faint">
                    No predictions yet. Analyze an article and it will show up here.
                  </p>
                </div>
              ) : (
                items.map((item) => <HistoryItem key={item.id} item={item} />)
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
