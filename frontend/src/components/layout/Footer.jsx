import { Link } from "react-router-dom";

const PRODUCT_LINKS = [
  { to: "/", label: "Home" },
  { to: "/detector", label: "Detector" },
  { to: "/features", label: "Features" },
  { to: "/about", label: "About" },
];

const TECH_LINKS = [
  { href: "https://fastapi.tiangolo.com/", label: "FastAPI" },
  { href: "https://xgboost.readthedocs.io/", label: "XGBoost + TF-IDF" },
  { href: "https://openai.com/", label: "GPT-4o mini" },
  { href: "https://python.langchain.com/", label: "LangChain" },
];

const RESOURCE_LINKS = [
  {
    href: "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
    label: "Training dataset",
  },
  { href: "https://duckduckgo.com/", label: "Web search source" },
  { href: "https://github.com/", label: "GitHub" },
];

function LinkColumn({ title, links, ...rest }) {
  return (
    <div {...rest}>
      <h3 className="text-xs font-bold uppercase tracking-[0.16em] text-faint">{title}</h3>
      <ul className="mt-4 space-y-2.5">
        {links.map((l) => (
          <li key={l.label}>
            {"to" in l ? (
              <Link to={l.to} className="text-sm text-muted transition-colors hover:text-accent">
                {l.label}
              </Link>
            ) : (
              <a
                href={l.href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-muted transition-colors hover:text-accent"
              >
                {l.label}
              </a>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="container-x grid gap-10 py-14 sm:grid-cols-2 lg:grid-cols-[1.5fr_1fr_1fr_1fr]">
        {/* Brand */}
        <div className="max-w-xs">
          <Link to="/" className="flex items-center gap-2.5">
            <img
              src="/icon-192.png"
              alt=""
              width={32}
              height={32}
              className="h-8 w-8 rounded-lg object-cover ring-1 ring-line"
            />
            <span className="font-bold tracking-tight">FakeNews Detector</span>
          </Link>
          <p className="mt-4 text-sm leading-relaxed text-muted">
            Dual-engine news verification. Local machine learning meets AI
            fact-checking with cited web sources.
          </p>
        </div>

        <LinkColumn title="Product" links={PRODUCT_LINKS} />
        <LinkColumn title="Technology" links={TECH_LINKS} />
        <LinkColumn title="Resources" links={RESOURCE_LINKS} />
      </div>

      <div className="border-t border-line">
        <div className="container-x flex flex-col items-center justify-between gap-2 py-5 text-xs text-faint sm:flex-row">
          <p>&copy; {new Date().getFullYear()} FakeNews Detector. All rights reserved.</p>
          <p>Results are AI-generated and should not be treated as definitive fact.</p>
        </div>
      </div>
    </footer>
  );
}
