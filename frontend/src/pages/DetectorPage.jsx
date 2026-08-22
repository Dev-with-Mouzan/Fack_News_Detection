import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";

import ModeSelector from "../components/detector/ModeSelector.jsx";
import NewsInput from "../components/detector/NewsInput.jsx";
import LoadingSteps from "../components/detector/LoadingSteps.jsx";
import ResultBanner from "../components/detector/ResultBanner.jsx";
import ConfidenceBar from "../components/detector/ConfidenceBar.jsx";
import ModelScores from "../components/detector/ModelScores.jsx";
import SourcesList from "../components/detector/SourcesList.jsx";

import { predictAI, predictCombined, predictML } from "../api/client.js";
import { useHistoryStore } from "../hooks/useHistory.jsx";
import { useSettings } from "../hooks/useSettings.jsx";
import { normalizeResult, renderBold } from "../utils/verdicts.jsx";
import { AlertTriangleIcon } from "../components/ui/icons.jsx";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const AI_LABELS = { gpt: "GPT", google: "Gemini" };

function getStepDefs(provider) {
  const aiName = AI_LABELS[provider] || "AI";
  return {
    ml: [
      { title: "Preprocessing text", desc: "Cleaning and tokenizing input" },
      { title: "Running ML model", desc: "XGBoost classification in progress" },
    ],
    ai: [
      { title: "Preprocessing text", desc: "Cleaning and tokenizing input" },
      { title: "Searching the web", desc: "Finding related coverage via DuckDuckGo" },
      { title: `Analyzing with ${aiName}`, desc: `${aiName} fact-checking in progress` },
    ],
    combined: [
      { title: "Preprocessing text", desc: "Cleaning and tokenizing input" },
      { title: "Running ML model", desc: "XGBoost classification in progress" },
      { title: "AI verification", desc: `Web search + ${aiName} fact-checking` },
    ],
  };
}

export default function DetectorPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { save } = useHistoryStore();
  const { provider, activeKey } = useSettings();

  const [mode, setMode] = useState("combined");
  const [text, setText] = useState("");
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [step, setStep] = useState(0);
  const textareaRef = useRef(null);

  const steps = getStepDefs(provider)[mode];
  const confidencePct = result ? Math.round((result.confidence ?? 0) * 100) : 0;

  // Restore a history entry passed via navigation state.
  useEffect(() => {
    const restoreId = location.state?.restoreId;
    if (!restoreId) return;
    try {
      const items = JSON.parse(localStorage.getItem("truthshield_history")) || [];
      const item = items.find((it) => it.id === restoreId);
      if (item?.result) {
        setMode(item.mode || "combined");
        setText(item.text || "");
        setResult(item.result);
        setStatus("done");
        setError("");
      }
    } catch {
      /* ignore malformed history */
    }
    // Clear the state so refresh/re-nav doesn't re-trigger.
    navigate(location.pathname, { replace: true, state: null });
  }, [location.state, location.pathname, navigate]);

  async function run() {
    const trimmed = text.trim();
    if (!trimmed) {
      setError("Please paste some news text first.");
      textareaRef.current?.focus();
      return;
    }

    setStatus("loading");
    setError("");
    setResult(null);
    setStep(0);

    try {
      let raw;
      // User-configured AI credentials are forwarded when set (AI + combined).
      const aiOpts = activeKey ? { provider, apiKey: activeKey } : {};

      if (mode === "ml") {
        setStep(1);
        [raw] = await Promise.all([predictML(trimmed), sleep(500)]);
        setStep(2);
      } else if (mode === "ai") {
        setStep(1);
        await sleep(450);
        setStep(2);
        [raw] = await Promise.all([predictAI(trimmed, aiOpts), sleep(400)]);
        setStep(3);
      } else {
        setStep(1);
        await sleep(500);
        setStep(2);
        [raw] = await Promise.all([predictCombined(trimmed, aiOpts), sleep(300)]);
        setStep(3);
      }

      // Normalize the three backend shapes into one display model.
      const normalized = normalizeResult(mode, raw);

      save({
        id: Date.now(),
        ts: new Date().toISOString(),
        mode,
        text: trimmed,
        result: normalized,
      });
      setResult(normalized);
      setStatus("done");
    } catch (err) {
      setError(err.message || "Something went wrong.");
      setStatus("error");
    }
  }

  function reset() {
    setText("");
    setResult(null);
    setError("");
    setStatus("idle");
    setStep(0);
    textareaRef.current?.focus();
  }

  const showResults = status === "done" && result;

  return (
    <section className="container-x max-w-3xl pb-24 pt-12 sm:pt-16">
      <motion.header
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <p className="eyebrow">Detector</p>
        <h1 className="mt-2 text-3xl font-extrabold tracking-tight sm:text-4xl">
          Check an article
        </h1>
        <p className="mt-2 text-muted">
          Pick an engine, paste the text, get a sourced verdict.
        </p>
      </motion.header>

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.08 }}
      >
        <div className="mt-8">
          <ModeSelector mode={mode} onChange={setMode} disabled={status === "loading"} />
        </div>

        <NewsInput
          value={text}
          onChange={setText}
          onSubmit={run}
          onClear={reset}
          loading={status === "loading"}
        />
      </motion.div>

      {/* Keyboard-accessible live region for errors */}
      <AnimatePresence>
        {status === "error" && error && (
          <motion.div
            role="alert"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-4 flex items-start gap-3 rounded-xl border border-bad/30 bg-bad/[0.07] p-4 text-sm text-bad"
          >
            <AlertTriangleIcon size={17} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Progress steps while a request is in flight */}
      <AnimatePresence>
        {status === "loading" && <LoadingSteps key={mode} steps={steps} current={step} />}
      </AnimatePresence>

      {/* Result card */}
      <AnimatePresence>
        {showResults && (
          <motion.section
            aria-label="Analysis result"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
            className="card mt-5 overflow-hidden"
          >
            <ResultBanner
              cls={result.cls}
              label={result.label}
              confidencePct={confidencePct}
              mode={result.mode}
            />

            <div className="space-y-6 p-6 sm:p-7">
              <ModelScores ml={result.ml} ai={result.ai} />
              {result.confidence > 0 && <ConfidenceBar pct={confidencePct} cls={result.cls} />}

              <section aria-label="Analysis explanation">
                <h3 className="text-xs font-bold uppercase tracking-[0.14em] text-faint">
                  Analysis
                </h3>
                <p className="mt-2.5 text-[15px] leading-relaxed text-muted">
                  {renderBold(result.explanation)}
                </p>
              </section>

              <SourcesList sources={result.sources} />
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      {/* Reset affordance under results */}
      {showResults && (
        <button type="button" onClick={reset} className="btn-ghost mt-4 w-full">
          Check another article
        </button>
      )}
    </section>
  );
}
