import { Link } from "react-router-dom";
import { motion } from "framer-motion";

import { ArrowRightIcon, CheckIcon, ShieldIcon } from "../ui/icons.jsx";

/* Staggered entrance for the hero copy */
const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.09, delayChildren: 0.05 } },
};
const item = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] } },
};

/* Deterministic particle field — CSS animates each dot upward on a loop */
const PARTICLES = [
  { left: "6%", size: 3, dist: "48vh", drift: "2rem", dur: "11s", delay: "0s" },
  { left: "14%", size: 4, dist: "38vh", drift: "-1.5rem", dur: "13s", delay: "3s" },
  { left: "24%", size: 3, dist: "52vh", drift: "1rem", dur: "10s", delay: "6s" },
  { left: "38%", size: 5, dist: "44vh", drift: "-2.5rem", dur: "15s", delay: "1.5s" },
  { left: "52%", size: 3, dist: "50vh", drift: "1.8rem", dur: "12s", delay: "7.5s" },
  { left: "64%", size: 4, dist: "40vh", drift: "-1rem", dur: "14s", delay: "4.5s" },
  { left: "76%", size: 3, dist: "54vh", drift: "2.2rem", dur: "11.5s", delay: "9s" },
  { left: "88%", size: 4, dist: "42vh", drift: "-1.8rem", dur: "13.5s", delay: "2s" },
  { left: "95%", size: 3, dist: "46vh", drift: "1.2rem", dur: "10.5s", delay: "5.2s" },
];

function Particles() {
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
      {PARTICLES.map((p, i) => (
        <span
          key={i}
          className="fx-particle"
          style={{
            left: p.left,
            width: p.size,
            height: p.size,
            "--dist": p.dist,
            "--drift": p.drift,
            animationDuration: p.dur,
            animationDelay: p.delay,
          }}
        />
      ))}
    </div>
  );
}

/** Animated product mockup: a fake article being scanned, then a verdict. */
function DetectionCard() {
  const lines = ["92%", "78%", "85%", "64%"];

  return (
    <div className="relative">
      {/* Sonar rings pulse outward from behind the card */}
      <div
        aria-hidden="true"
        className="fx-radar absolute inset-0 grid place-items-center"
      >
        <span />
        <span />
        <span />
      </div>

      {/* Ambient glow */}
      <div className="absolute -inset-10 rounded-full bg-accent/15 blur-3xl" aria-hidden="true" />

      <div className="card relative -rotate-1 p-5 transition-transform duration-300 hover:rotate-0">
        {/* Window chrome */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-bad/60" />
            <span className="h-2.5 w-2.5 rounded-full bg-warn/60" />
            <span className="h-2.5 w-2.5 rounded-full bg-ok/60" />
          </div>
          <span className="font-mono text-[11px] text-faint">article.txt</span>
        </div>

        {/* Article body + scanning beam */}
        <div className="relative overflow-hidden rounded-xl border border-line bg-surface2 p-4">
          <div className="space-y-3">
            <div className="h-3 w-[70%] rounded bg-ink/15" />
            {lines.map((w) => (
              <div key={w} className="h-2.5 rounded bg-ink/[0.08]" style={{ width: w }} />
            ))}
            {/* Flagged claim */}
            <div className="rounded-lg border-l-2 border-warn bg-warn/10 px-3 py-2">
              <div className="h-2.5 w-[88%] rounded bg-warn/30" />
              <p className="mt-1.5 text-[10px] font-semibold uppercase tracking-wide text-warn">
                Unverified claim detected
              </p>
            </div>
            <div className="h-2.5 w-[58%] rounded bg-ink/[0.08]" />
          </div>

          {/* Scan beam â€” loops top to bottom */}
          <motion.div
            aria-hidden="true"
            className="pointer-events-none absolute inset-x-0 h-14 bg-gradient-to-b from-transparent via-accent/20 to-transparent"
            initial={{ y: -60 }}
            animate={{ y: 260 }}
            transition={{ duration: 2.6, repeat: Infinity, ease: "easeInOut", repeatDelay: 0.8 }}
          />
        </div>

        {/* Engine readouts */}
        <div className="mt-4 grid grid-cols-2 gap-3">
          <div className="rounded-xl border border-line bg-surface2 p-3">
            <p className="text-[10px] font-bold uppercase tracking-wide text-faint">ML model</p>
            <p className="mt-1 flex items-center gap-1.5 font-mono text-sm font-semibold text-ok">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-ok" />
              Real Â· 94%
            </p>
          </div>
          <div className="rounded-xl border border-line bg-surface2 p-3">
            <p className="text-[10px] font-bold uppercase tracking-wide text-faint">AI check</p>
            <p className="mt-1 flex items-center gap-1.5 font-mono text-sm font-semibold text-accent">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent" />
              True Â· 91%
            </p>
          </div>
        </div>
      </div>

      {/* Verdict badge pops in over the card corner */}
      <motion.div
        initial={{ scale: 0, rotate: -6 }}
        animate={{ scale: 1, rotate: 3 }}
        transition={{ type: "spring", stiffness: 260, damping: 16, delay: 0.9 }}
        className="absolute -bottom-5 right-4 flex items-center gap-2 rounded-full bg-gradient-to-r from-brand to-brand-2 px-4 py-2 text-xs font-bold uppercase tracking-wide text-white shadow-card-lg"
      >
        <CheckIcon size={13} />
        Verdict: Real
      </motion.div>
    </div>
  );
}

export default function Hero() {
  return (
    <section className="relative overflow-hidden pb-24 pt-16 sm:pt-24">
      <Particles />

      {/* Background glows */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-32 right-[-10%] h-96 w-96 rounded-full bg-accent/10 blur-3xl"
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute bottom-[-20%] left-[-10%] h-96 w-96 rounded-full bg-brand/20 blur-3xl"
      />

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="container-x grid items-center gap-16 lg:grid-cols-2"
      >
        {/* Copy */}
        <div>
          <motion.div variants={item}>
            <span className="inline-flex items-center gap-2 rounded-full border border-accent/25 bg-accent/10 px-3.5 py-1.5 text-xs font-semibold text-accent">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
              </span>
              Dual-engine verification
            </span>
          </motion.div>

          <motion.h1
            variants={item}
            className="mt-6 max-w-xl text-4xl font-extrabold leading-[1.06] tracking-tight sm:text-5xl xl:text-6xl"
          >
            Detect fake news in{" "}
            <span className="text-gradient">seconds.</span>
          </motion.h1>

          <motion.p variants={item} className="mt-5 max-w-lg text-lg leading-relaxed text-muted">
            FakeNews Detector cross-checks any article with a local XGBoost classifier
            and GPT-powered web fact-checking â€” so you know what to trust
            before you share it.
          </motion.p>

          <motion.div variants={item} className="mt-8 flex flex-wrap items-center gap-3">
            <Link to="/detector" className="btn-primary !px-7 !py-3.5 !text-base">
              Try the detector
              <ArrowRightIcon size={17} />
            </Link>
            <a href="#how-it-works" className="btn-ghost !px-6 !py-3.5 !text-base">
              See how it works
            </a>
          </motion.div>

          <motion.ul
            variants={item}
            className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-faint"
          >
            {["~99% ML accuracy", "No signup required", "Sources cited"].map((t) => (
              <li key={t} className="flex items-center gap-1.5">
                <ShieldIcon size={13} className="text-accent" />
                {t}
              </li>
            ))}
          </motion.ul>
        </div>

        {/* Product mockup */}
        <motion.div
          initial={{ opacity: 0, y: 28 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1], delay: 0.25 }}
          className="hidden lg:block"
        >
          <DetectionCard />
        </motion.div>
      </motion.div>
    </section>
  );
}
