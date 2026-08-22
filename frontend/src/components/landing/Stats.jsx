import { useEffect, useRef } from "react";
import { animate, motion, useInView } from "framer-motion";

/** Count-up number that triggers once when scrolled into view. */
function CountUp({ to, suffix = "", prefix = "" }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });

  useEffect(() => {
    if (!inView || !ref.current) return;
    const controls = animate(0, to, {
      duration: 1.6,
      ease: "easeOut",
      onUpdate: (v) => {
        if (ref.current) ref.current.textContent = `${prefix}${Math.round(v)}${suffix}`;
      },
    });
    return () => controls.stop();
  }, [inView, to, prefix, suffix]);

  return (
    <span ref={ref}>
      {prefix}0{suffix}
    </span>
  );
}

const STATS = [
  { value: 99, suffix: "%", label: "ML classification accuracy" },
  { value: 3, suffix: "", label: "Analysis engines" },
  { value: 44, suffix: "K+", label: "Training articles" },
  { value: 5, prefix: "<", suffix: "s", label: "Average verdict time" },
];

export default function Stats() {
  return (
    <section className="border-y border-line bg-surface/60">
      <div className="container-x grid grid-cols-2 gap-x-6 gap-y-10 py-14 lg:grid-cols-4">
        {STATS.map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.5, delay: i * 0.08 }}
            className="text-center"
          >
            <p className="text-gradient text-4xl font-extrabold tracking-tight sm:text-5xl">
              <CountUp to={stat.value} prefix={stat.prefix || ""} suffix={stat.suffix} />
            </p>
            <p className="mt-2 text-sm text-muted">{stat.label}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
