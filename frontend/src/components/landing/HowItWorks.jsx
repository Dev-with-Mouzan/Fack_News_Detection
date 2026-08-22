import { motion } from "framer-motion";

import { FileTextIcon, LayersIcon, GlobeIcon, CheckIcon } from "../ui/icons.jsx";

const STEPS = [
  {
    icon: FileTextIcon,
    title: "Paste the article",
    desc: "Drop in a headline or a full article â€” any text you want verified.",
  },
  {
    icon: LayersIcon,
    title: "Pick an engine",
    desc: "Combined for maximum confidence, ML-only for speed, or AI-only for claim-level fact-checking.",
  },
  {
    icon: GlobeIcon,
    title: "Engines analyze",
    desc: "XGBoost scores the text while GPT searches the web for corroborating coverage.",
  },
  {
    icon: CheckIcon,
    title: "Get your verdict",
    desc: "A clear Real/Fake call with a confidence score, explanation, and cited sources.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="border-y border-line bg-surface/50 py-24">
      <div className="container-x">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.55 }}
          className="mx-auto max-w-2xl text-center"
        >
          <p className="eyebrow">How it works</p>
          <h2 className="mt-3 text-3xl font-extrabold tracking-tight sm:text-4xl">
            From paste to verdict in four steps
          </h2>
        </motion.div>

        <ol className="relative mt-16 grid gap-10 lg:grid-cols-4 lg:gap-6">
          {STEPS.map((step, i) => (
            <motion.li
              key={step.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.5, delay: i * 0.12 }}
              className="relative"
            >
              {/* Connector line to the next step (desktop only) */}
              {i < STEPS.length - 1 && (
                <span
                  aria-hidden="true"
                  className="absolute left-[calc(50%+3rem)] right-[calc(-50%+3rem)] top-8 hidden h-px bg-line lg:block"
                />
              )}

              <div className="flex flex-col items-center text-center">
                <span className="relative grid h-16 w-16 place-items-center rounded-2xl bg-gradient-to-br from-brand to-brand-2 text-white shadow-glow">
                  {/* Ghost number behind icon */}
                  <span className="absolute -right-2 -top-2 grid h-6 w-6 place-items-center rounded-full border border-line bg-surface font-mono text-[11px] font-semibold text-muted">
                    {i + 1}
                  </span>
                  <step.icon size={24} />
                </span>
                <h3 className="mt-5 font-bold tracking-tight">{step.title}</h3>
                <p className="mt-2 max-w-xs text-sm leading-relaxed text-muted">{step.desc}</p>
              </div>
            </motion.li>
          ))}
        </ol>
      </div>
    </section>
  );
}
