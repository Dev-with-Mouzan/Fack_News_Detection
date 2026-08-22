import { Link } from "react-router-dom";
import { motion } from "framer-motion";

import {
  AlertTriangleIcon,
  ArrowRightIcon,
  CpuIcon,
  GlobeIcon,
  LockIcon,
  MessageIcon,
  SearchIcon,
  ShieldIcon,
} from "../components/ui/icons.jsx";

const STACK = [
  { icon: CpuIcon, name: "XGBoost + TF-IDF", role: "Local ML classifier trained on 44K+ labeled articles" },
  { icon: GlobeIcon, name: "DuckDuckGo Search", role: "Live retrieval of related coverage for evidence" },
  { icon: MessageIcon, name: "GPT-4o mini", role: "Reasons over search results to verify claims" },
  { icon: SearchIcon, name: "LangChain", role: "Orchestrates the search-and-verify pipeline" },
  { icon: ShieldIcon, name: "FastAPI", role: "Serves the prediction API and the built frontend" },
  { icon: LockIcon, name: "Browser storage", role: "History stays on your device — no tracking" },
];

const LIMITATIONS = [
  "Verdicts are AI-assisted estimates, not legal or editorial determinations.",
  "The ML model reflects patterns in its training data and may inherit its biases.",
  "AI verification depends on external services; availability can vary.",
  "Very short text (a few words) gives the engines little to work with.",
];

export default function AboutPage() {
  return (
    <>
      <section className="container-x pb-16 pt-16 sm:pt-24">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55 }}
          className="max-w-2xl"
        >
          <p className="eyebrow">About</p>
          <h1 className="mt-3 text-4xl font-extrabold tracking-tight sm:text-5xl">
            Built for a web where{" "}
            <span className="text-gradient">seeing isn&rsquo;t believing</span>
          </h1>
          <p className="mt-5 text-lg leading-relaxed text-muted">
            FakeNews Detector started as a simple question: before I share this
            article, is there any fast, honest way to check it? Manual
            fact-checking is slow; gut feeling is worse. So we combined two
            independent engines — one statistical, one reasoning-based — and
            made them show their work.
          </p>
        </motion.div>
      </section>

      {/* Tech stack */}
      <section className="container-x pb-20">
        <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">Under the hood</h2>
        <div className="mt-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {STACK.map((tech, i) => (
            <motion.div
              key={tech.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.5, delay: i * 0.06 }}
              className="card flex items-start gap-4 p-6"
            >
              <span className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-accent/10 text-accent">
                <tech.icon size={20} />
              </span>
              <div>
                <h3 className="font-bold tracking-tight">{tech.name}</h3>
                <p className="mt-1 text-sm leading-relaxed text-muted">{tech.role}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Limitations */}
      <section className="container-x pb-20">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.6 }}
          className="card border-warn/30 p-7 sm:p-9"
        >
          <div className="flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-xl bg-warn/10 text-warn">
              <AlertTriangleIcon size={19} />
            </span>
            <h2 className="text-xl font-bold tracking-tight">Honest limitations</h2>
          </div>
          <ul className="mt-5 space-y-3">
            {LIMITATIONS.map((limit) => (
              <li key={limit} className="flex gap-3 text-sm leading-relaxed text-muted">
                <span aria-hidden="true" className="mt-[9px] h-1.5 w-1.5 shrink-0 rounded-full bg-warn" />
                {limit}
              </li>
            ))}
          </ul>
        </motion.div>
      </section>

      {/* Contact */}
      <section className="container-x pb-24">
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.55 }}
          className="card flex flex-col items-start justify-between gap-6 p-8 sm:flex-row sm:items-center"
        >
          <div>
            <h2 className="text-xl font-bold tracking-tight">Questions or feedback?</h2>
            <p className="mt-1.5 text-sm text-muted">
              Found a verdict that looks wrong? We want to hear about it.
            </p>
          </div>
          <div className="flex shrink-0 gap-3">
            <a href="mailto:hello@fakenewsdetector.app" className="btn-ghost">
              Email us
            </a>
            <Link to="/detector" className="btn-primary">
              Try the detector
              <ArrowRightIcon size={15} />
            </Link>
          </div>
        </motion.div>
      </section>
    </>
  );
}
