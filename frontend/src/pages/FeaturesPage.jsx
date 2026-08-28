import { motion } from "framer-motion";
import Features, { CORE_FEATURES, MORE_FEATURES } from "../components/landing/Features.jsx";
import HowItWorks from "../components/landing/HowItWorks.jsx";
import Stats from "../components/landing/Stats.jsx";
import CTA from "../components/landing/CTA.jsx";
import { CheckIcon, XIcon } from "../components/ui/icons.jsx";

const ALL_FEATURES = [...CORE_FEATURES, ...MORE_FEATURES];

/** Capability comparison across the three engines. */
const COMPARISON = [
  { method: "ML only", speed: "< 1s", internet: false, apiKey: false, best: "Quick triage" },
  { method: "AI only", speed: "~10–20s", internet: true, apiKey: true, best: "Claim-level fact-check" },
  { method: "Combined", speed: "~15–25s", internet: true, apiKey: false, best: "Highest confidence" },
];

function Cell({ value }) {
  return value ? (
    <span className="inline-flex items-center gap-1.5 text-ok">
      <CheckIcon size={13} /> Yes
    </span>
  ) : (
    <span className="inline-flex items-center gap-1.5 text-faint">
      <XIcon size={12} /> No
    </span>
  );
}

export default function FeaturesPage() {
  return (
    <>
      {/* Page header */}
      <section className="container-x pb-4 pt-16 sm:pt-24">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55 }}
          className="mx-auto max-w-2xl text-center"
        >
          <p className="eyebrow">Features</p>
          <h1 className="mt-3 text-4xl font-extrabold tracking-tight sm:text-5xl">
            Everything you need to fight{" "}
            <span className="text-gradient">misinformation</span>
          </h1>
          <p className="mt-5 text-lg leading-relaxed text-muted">
            Three engines, cited evidence, and a verdict you can actually
            understand — here&rsquo;s the full toolbox.
          </p>
        </motion.div>
      </section>

      <Features items={ALL_FEATURES} />

      {/* Engine comparison */}
      <section className="pb-24">
        <div className="container-x">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-80px" }}
            transition={{ duration: 0.6 }}
            className="card overflow-hidden"
          >
            <div className="border-b border-line px-7 py-6">
              <h2 className="text-xl font-bold tracking-tight">Engine comparison</h2>
              <p className="mt-1 text-sm text-muted">Pick the right mode for the job.</p>
            </div>
            <div className="overflow-x-auto scroll-thin">
              <table className="w-full min-w-[640px] text-sm">
                <thead>
                  <tr className="border-b border-line text-left">
                    {["Method", "Speed", "Internet", "API key", "Best for"].map((h) => (
                      <th
                        key={h}
                        className="px-7 py-3.5 text-xs font-bold uppercase tracking-wider text-faint"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {COMPARISON.map((row) => (
                    <tr
                      key={row.method}
                      className={`border-b border-line/60 transition-colors last:border-0 hover:bg-surface2 ${
                        row.method === "Combined" ? "bg-accent/[0.04]" : ""
                      }`}
                    >
                      <td className="px-7 py-4 font-semibold">{row.method}</td>
                      <td className="px-7 py-4 font-mono text-xs text-muted">{row.speed}</td>
                      <td className="px-7 py-4"><Cell value={row.internet} /></td>
                      <td className="px-7 py-4"><Cell value={row.apiKey} /></td>
                      <td className="px-7 py-4 text-muted">{row.best}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        </div>
      </section>

      <HowItWorks />
      <Stats />
      <CTA />
    </>
  );
}
