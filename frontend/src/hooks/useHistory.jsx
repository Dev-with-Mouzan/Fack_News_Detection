import { createContext, useCallback, useContext, useMemo, useState } from "react";

const HistoryContext = createContext(null);
const STORAGE_KEY = "truthshield_history";
const MAX_ITEMS = 50;

function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function HistoryProvider({ children }) {
  const [items, setItems] = useState(load);
  const [open, setOpen] = useState(false);

  const persist = useCallback((next) => {
    setItems(next);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      /* ignore quota errors */
    }
  }, []);

  const save = useCallback(
    (entry) => {
      persist([entry, ...load()].slice(0, MAX_ITEMS));
    },
    [persist]
  );

  const clearAll = useCallback(() => persist([]), [persist]);

  const value = useMemo(
    () => ({ items, open, setOpen, save, clearAll }),
    [items, open, save, clearAll]
  );

  return <HistoryContext.Provider value={value}>{children}</HistoryContext.Provider>;
}

export function useHistoryStore() {
  const ctx = useContext(HistoryContext);
  if (!ctx) throw new Error("useHistoryStore must be used inside HistoryProvider");
  return ctx;
}
