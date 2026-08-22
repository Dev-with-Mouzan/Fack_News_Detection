import { Link } from "react-router-dom";
import { motion } from "framer-motion";

import { ArrowRightIcon } from "../ui/icons.jsx";

export default function CTA() {
  return (
    <section className="py-24">
      <div className="container-x">
        <motion.div
          initial={{ opacity: 0, scale: 0.97 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-brand to-brand-2 px-8 py-16 text-center text-white sm:px-16"
        >
          {/* Decorative blurred circles */}
          <span
            aria-hidden="true"
            className="pointer-events-none absolute -left-20 -top-20 h-64 w-64 rounded-full bg-white/15 blur-3xl"
          />
          <span
            aria-hidden="true"
            className="pointer-events-none absolute -bottom-24 -right-16 h-72 w-72 rounded-full bg-white/10 blur-3xl"
          />
          {/* Light streak sweeping across the panel */}
          <span aria-hidden="true" className="fx-shimmer" />

          <h2 className="relative text-3xl font-extrabold tracking-tight sm:text-4xl">
            Ready to find out what&rsquo;s real?
          </h2>
          <p className="relative mx-auto mt-4 max-w-xl leading-relaxed text-white/85">
            Paste an article and get a sourced verdict in seconds. Free, no
            account needed.
          </p>
          <Link
            to="/detector"
            className="relative mt-8 inline-flex items-center gap-2 rounded-full bg-white px-8 py-3.5 font-bold text-bg shadow-xl transition-transform duration-200 hover:scale-[1.03] active:scale-95"
          >
            Open the detector
            <ArrowRightIcon size={17} />
          </Link>
        </motion.div>
      </div>
    </section>
  );
}
