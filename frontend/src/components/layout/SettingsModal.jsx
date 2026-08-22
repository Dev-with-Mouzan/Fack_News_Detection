import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import { PROVIDERS, useSettings } from "../../hooks/useSettings.jsx";
import {
  CheckIcon,
  EyeIcon,
  EyeOffIcon,
  GemIcon,
  KeyIcon,
  SparklesIcon,
  XIcon,
} from "../ui/icons.jsx";

const PROVIDER_ICONS = { gpt: SparklesIcon, google: GemIcon };

function ProviderCard({ provider, active, onSelect }) {
  const Icon = PROVIDER_ICONS[provider.id];
  return (
    <button
      type="button"
      onClick={() => onSelect(provider.id)}
      aria-pressed={active}
      className={`flex w-full items-center gap-3.5 rounded-xl border p-4 text-left transition-all duration-200 ${
        active
          ? "border-brand-2 bg-brand/25 shadow-glow"
          : "border-line bg-surface2 hover:border-line hover:bg-surface2/70"
      }`}
    >
      <span
        className={`grid h-10 w-10 shrink-0 place-items-center rounded-lg transition-colors ${
          active ? "bg-brand text-white" : "bg-surface text-muted"
        }`}
      >
        <Icon size={18} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-bold tracking-tight">{provider.name}</span>
        <span className="block truncate text-xs text-muted">{provider.desc}</span>
      </span>
      {active && (
        <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-accent text-bg">
          <CheckIcon size={11} />
        </span>
      )}
    </button>
  );
}

export default function SettingsModal({ open, onClose }) {
  const { provider, keys, setProvider, setKey } = useSettings();
  const [draftKey, setDraftKey] = useState(keys[provider] || "");
  const [showKey, setShowKey] = useState(false);
  const [saved, setSaved] = useState(false);

  // Sync the draft when the modal opens or provider switches.
  useEffect(() => {
    if (open) {
      setDraftKey(keys[provider] || "");
      setSaved(false);
    }
  }, [open, provider]); // eslint-disable-line react-hooks/exhaustive-deps

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  function handleSave() {
    setKey(provider, draftKey);
    setSaved(true);
    setTimeout(onClose, 700);
  }

  const meta = PROVIDERS[provider];

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-[80] bg-black/55 backdrop-blur-sm"
          />
          <div className="pointer-events-none fixed inset-0 z-[90] grid place-items-center p-4">
            <motion.div
              role="dialog"
              aria-label="AI settings"
              initial={{ opacity: 0, scale: 0.95, y: 14 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.97, y: 8 }}
              transition={{ type: "spring", stiffness: 340, damping: 28 }}
              className="card pointer-events-auto w-full max-w-md overflow-hidden !rounded-3xl border border-line shadow-card-lg"
            >
              {/* Header */}
              <header className="flex items-center justify-between border-b border-line px-6 py-4">
                <h2 className="text-base font-bold tracking-tight">AI settings</h2>
                <button
                  type="button"
                  onClick={onClose}
                  aria-label="Close settings"
                  className="icon-btn"
                >
                  <XIcon size={16} />
                </button>
              </header>

              <div className="space-y-4 px-6 py-5">
                {/* Provider choice */}
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.14em] text-faint">
                    AI provider
                  </p>
                  <div className="mt-2.5 space-y-2">
                    {Object.values(PROVIDERS).map((p) => (
                      <ProviderCard
                        key={p.id}
                        provider={p}
                        active={provider === p.id}
                        onSelect={(id) => {
                          setProvider(id);
                          setDraftKey(keys[id] || "");
                          setSaved(false);
                        }}
                      />
                    ))}
                  </div>
                </div>

                {/* API key */}
                <div>
                  <label
                    htmlFor="api-key-input"
                    className="text-xs font-bold uppercase tracking-[0.14em] text-faint"
                  >
                    {meta.name} API key
                  </label>
                  <div className="relative mt-2">
                    <KeyIcon
                      size={15}
                      className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-faint"
                    />
                    <input
                      id="api-key-input"
                      type={showKey ? "text" : "password"}
                      value={draftKey}
                      onChange={(e) => {
                        setDraftKey(e.target.value);
                        setSaved(false);
                      }}
                      placeholder={meta.keyHint}
                      autoComplete="off"
                      spellCheck={false}
                      className="input-field !py-3 pl-10 pr-11 font-mono text-xs"
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey((s) => !s)}
                      aria-label={showKey ? "Hide API key" : "Show API key"}
                      className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md p-1 text-faint transition-colors hover:text-ink"
                    >
                      {showKey ? <EyeOffIcon size={16} /> : <EyeIcon size={16} />}
                    </button>
                  </div>

                  <p className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-faint">
                    <span>Stored only in this browser.</span>
                    <a
                      href={meta.keyUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-semibold text-accent hover:underline"
                    >
                      Get a key ↗
                    </a>
                  </p>
                </div>
              </div>

              {/* Footer */}
              <footer className="flex items-center justify-between gap-3 border-t border-line px-6 py-4">
                <p aria-live="polite" className="text-xs font-semibold text-ok">
                  {saved && "Saved"}
                </p>
                <div className="flex gap-2.5">
                  <button type="button" onClick={onClose} className="btn-ghost">
                    Cancel
                  </button>
                  <button type="button" onClick={handleSave} className="btn-primary">
                    Save
                  </button>
                </div>
              </footer>
            </motion.div>
          </div>
        </>
      )}
    </AnimatePresence>
  );
}
