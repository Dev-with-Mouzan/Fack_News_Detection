import { createContext, useCallback, useContext, useMemo, useState } from "react";

const SettingsContext = createContext(null);
const STORAGE_KEY = "ts_settings";

export const PROVIDERS = {
  gpt: {
    id: "gpt",
    name: "OpenAI GPT",
    desc: "GPT-4o mini fact-checking",
    keyUrl: "https://platform.openai.com/api-keys",
    keyHint: "sk-…",
  },
  google: {
    id: "google",
    name: "Google Gemini",
    desc: "Gemini fact-checking via AI Studio",
    keyUrl: "https://aistudio.google.com/app/apikey",
    keyHint: "AIza…",
  },
};

function load() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch {
    return {};
  }
}

export function SettingsProvider({ children }) {
  // shape: { provider: "gpt" | "google", keys: { gpt: "", google: "" } }
  const [settings, setSettings] = useState(() => ({
    provider: "gpt",
    keys: { gpt: "", google: "" },
    ...load(),
  }));

  const update = useCallback((next) => {
    setSettings(next);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      /* ignore */
    }
  }, []);

  const setProvider = useCallback(
    (provider) => update({ ...settings, provider }),
    [settings, update]
  );

  const setKey = useCallback(
    (provider, key) =>
      update({ ...settings, keys: { ...settings.keys, [provider]: key.trim() } }),
    [settings, update]
  );

  const value = useMemo(() => {
    const activeKey = settings.keys[settings.provider] || "";
    return {
      provider: settings.provider,
      keys: settings.keys,
      activeKey,
      hasKey: activeKey.length > 0,
      setProvider,
      setKey,
    };
  }, [settings, setProvider, setKey]);

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used inside SettingsProvider");
  return ctx;
}
