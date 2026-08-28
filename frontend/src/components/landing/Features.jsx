import { motion } from "framer-motion";

import {
  ClockIcon,
  CpuIcon,
  GlobeIcon,
  LockIcon,
  MessageIcon,
  SearchIcon,
  WifiOffIcon,
  ZapIcon,
} from "../ui/icons.jsx";

/** Core trio shown on the landing page. */
export const CORE_FEATURES = [
  {
    icon: CpuIcon,
    title: "Dual-engine verification",
    desc: "A local XGBoost classifier and GPT fact-checking run side by side. When both engines agree, confidence gets a boost.",
  },
  {
    icon: SearchIcon,
    title: "Live web evidence",
    desc: "The AI engine searches DuckDuckGo for related coverage, then reasons over real sources â€” every verdict comes with citations.",
  },
  {
    icon: LockIcon,
    title: "Private by design",
    desc: "ML-only mode runs entirely on the server with no external API calls. History stays in your browser, never in a database.",
  },
];

/** Extended set for the dedicated Features page. */
export const MORE_FEATURES = [
  {
    icon: MessageIcon,
    title: "Explainable verdicts",
    desc: "Every result ships with a plain-language explanation of why the text was classified that way.",
  },
  {
    icon: ClockIcon,
    title: "Built-in history",
    desc: "Past analyses are saved locally. Reopen any previous check to restore its full verdict instantly.",
  },
  {
    icon: ZapIcon,
    title: "Fast by default",
    desc: "ML-only mode returns a verdict in under a second â€” no internet or API key needed.",
  },
  {
    icon: WifiOffIcon,
    title: "Works offline",
    desc: "The classifier is bundled with the app, so basic checks keep working even when external services are down.",
  },
];

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: (i) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.55, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] },
  }),
};

/**
 * @param {{items?: object[], id?: string}} props
 */
export default function Features({ items = CORE_FEATURES, id }) {
  return (
    <section id={id} className="py-24">
      <div className="container-x">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.55 }}
        >
          <div className="mx-auto max-w-2xl text-center">
            <p className="eyebrow">Why FakeNews Detector</p>
            <h2 className="mt-3 text-3xl font-extrabold tracking-tight sm:text-4xl">
              Two engines are better than one
            </h2>
            <p className="mt-4 leading-relaxed text-muted">
              Single-source fact-checking is fragile. FakeNews Detector combines
              statistical pattern matching with live evidence gathering, so a
              verdict only sticks when the signals line up.
            </p>
          </div>
        </motion.div>

        <div className="mt-12 grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {items.map((feature, i) => (
            <motion.article
              key={feature.title}
              custom={i % 3}
              variants={fadeUp}
              initial="hidden"
              whileInView="show"
              viewport={{ once: true, margin: "-60px" }}
              className="card group p-7 transition-all duration-300 hover:-translate-y-1 hover:shadow-card-lg"
            >
              <span className="mb-5 grid h-12 w-12 place-items-center rounded-xl bg-accent/10 text-accent transition-colors duration-300 group-hover:bg-gradient-to-br group-hover:from-brand group-hover:to-brand-2 group-hover:text-white">
                <feature.icon size={22} />
              </span>
              <h3 className="text-lg font-bold tracking-tight">{feature.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">{feature.desc}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
