import { useEffect } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence, motion, MotionConfig } from "framer-motion";

import Navbar from "./components/layout/Navbar.jsx";
import Footer from "./components/layout/Footer.jsx";
import HistorySidebar from "./components/detector/HistorySidebar.jsx";
import BackgroundFX from "./components/ui/BackgroundFX.jsx";
import Landing from "./pages/Landing.jsx";
import DetectorPage from "./pages/DetectorPage.jsx";
import FeaturesPage from "./pages/FeaturesPage.jsx";
import AboutPage from "./pages/AboutPage.jsx";

import { HistoryProvider } from "./hooks/useHistory.jsx";
import { SettingsProvider } from "./hooks/useSettings.jsx";
/** Fade/slide wrapper applied to every routed page. */
function Page({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.28, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
}

export default function App() {
  const location = useLocation();

  // Scroll to top on navigation (instant — smooth scrolling stays for anchors).
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [location.pathname]);

  return (
    <MotionConfig reducedMotion="user">
      <SettingsProvider>
        <HistoryProvider>
          <a
            href="#main"
            className="fixed left-4 top-4 z-[100] -translate-y-24 rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-white transition-transform focus-visible:translate-y-0"
          >
            Skip to content
          </a>

          <BackgroundFX />

          <div className="relative z-10 flex min-h-dvh flex-col">
            <Navbar />

            <main id="main" className="flex-1">
              <AnimatePresence mode="wait" initial={false}>
                <Routes location={location} key={location.pathname}>
                  <Route path="/" element={<Page><Landing /></Page>} />
                  <Route path="/detector" element={<Page><DetectorPage /></Page>} />
                  <Route path="/features" element={<Page><FeaturesPage /></Page>} />
                  <Route path="/about" element={<Page><AboutPage /></Page>} />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </AnimatePresence>
            </main>

            <Footer />
          </div>

          <HistorySidebar />
        </HistoryProvider>
      </SettingsProvider>
    </MotionConfig>
  );
}
