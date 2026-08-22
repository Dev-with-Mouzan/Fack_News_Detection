import { useEffect, useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";

import { useHistoryStore } from "../../hooks/useHistory.jsx";
import { useSettings } from "../../hooks/useSettings.jsx";
import SettingsModal from "./SettingsModal.jsx";
import {
  MenuIcon,
  HistoryIcon,
  XIcon,
  ZapIcon,
  SettingsIcon,
} from "../ui/icons.jsx";

const TABS = [
  { to: "/", label: "Home", end: true },
  { to: "/detector", label: "Detector" },
  { to: "/features", label: "Features" },
  { to: "/about", label: "About" },
];

export default function Navbar() {
  const { setOpen } = useHistoryStore();
  const { hasKey } = useSettings();
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const location = useLocation();

  // Elevate the bar once the page scrolls.
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Close the mobile menu whenever the route changes.
  useEffect(() => setMenuOpen(false), [location.pathname]);

  return (
    <>
      <header
        className={`sticky top-0 z-50 border-b backdrop-blur-xl transition-colors duration-300 ${
          scrolled ? "border-line bg-bg/80" : "border-transparent bg-bg/60"
        }`}
      >
        <nav className="container-x flex h-16 items-center justify-between gap-4">
          {/* Brand */}
          <Link to="/" className="group flex shrink-0 items-center gap-2.5" aria-label="FakeNews Detector home">
            <img
              src="/logo.png"
              alt="FakeNews Detector logo"
              width={36}
              height={36}
              className="h-9 w-9 rounded-xl object-cover shadow-glow ring-1 ring-line transition-transform duration-200 group-hover:scale-105"
            />
            <span className="flex flex-col leading-none">
              <span className="text-[1.05rem] font-bold tracking-tight">FakeNews Detector</span>
              <span className="mt-0.5 hidden text-[10px] font-semibold uppercase tracking-[0.18em] text-faint sm:block">
                News verification
              </span>
            </span>
          </Link>

          {/* Desktop tabs */}
          <div className="hidden items-center gap-1 rounded-full border border-line bg-surface2/70 p-1 md:flex">
            {TABS.map((tab) => (
              <NavLink
                key={tab.to}
                to={tab.to}
                end={tab.end}
                className={({ isActive }) =>
                  `rounded-full px-4 py-1.5 text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? "bg-surface text-ink shadow-card"
                      : "text-muted hover:text-ink"
                  }`
                }
              >
                {tab.label}
              </NavLink>
            ))}
          </div>

          {/* Right cluster */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="icon-btn"
              title="Prediction history"
              aria-label="Open prediction history"
              onClick={() => setOpen(true)}
            >
              <HistoryIcon size={17} />
            </button>

            {/* Settings — LLM provider & API key */}
            <button
              type="button"
              className="icon-btn"
              onClick={() => setSettingsOpen(true)}
              title="AI settings (provider & API key)"
              aria-label="Open AI settings"
            >
              <SettingsIcon size={17} />
              {hasKey && (
                <span
                  aria-hidden="true"
                  className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-accent ring-2 ring-bg"
                />
              )}
            </button>

            <Link to="/detector" className="btn-primary hidden !px-4 !py-2 lg:inline-flex">
              <ZapIcon size={15} />
              Try Detector
            </Link>

            {/* Mobile hamburger */}
            <button
              type="button"
              className="icon-btn md:hidden"
              onClick={() => setMenuOpen((o) => !o)}
              aria-expanded={menuOpen}
              aria-label="Toggle navigation menu"
            >
              {menuOpen ? <XIcon size={17} /> : <MenuIcon size={17} />}
            </button>
          </div>
        </nav>

        {/* Mobile menu panel */}
        <AnimatePresence>
          {menuOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25, ease: "easeInOut" }}
              className="overflow-hidden border-t border-line bg-bg/95 backdrop-blur-xl md:hidden"
            >
              <div className="container-x flex flex-col gap-1 py-4">
                {TABS.map((tab) => (
                  <NavLink
                    key={tab.to}
                    to={tab.to}
                    end={tab.end}
                    className={({ isActive }) =>
                      `rounded-xl px-4 py-3 text-sm font-semibold transition-colors ${
                        isActive ? "bg-brand/30 text-accent" : "text-muted hover:bg-surface2 hover:text-ink"
                      }`
                    }
                  >
                    {tab.label}
                  </NavLink>
                ))}
                <div className="mt-2 flex gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setMenuOpen(false);
                      setSettingsOpen(true);
                    }}
                    className="btn-ghost flex-1"
                  >
                    <SettingsIcon size={15} />
                    AI settings
                  </button>
                  <Link to="/detector" className="btn-primary flex-1">
                    <ZapIcon size={15} />
                    Detector
                  </Link>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      <SettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </>
  );
}
