import { LinkIcon } from "../ui/icons.jsx";

function isUrl(s) {
  return /^https?:\/\//i.test(s);
}

/** Cited evidence list; URLs become external links, plain strings stay text. */
export default function SourcesList({ sources }) {
  if (!sources?.length) return null;

  return (
    <section aria-label="Sources consulted">
      <h3 className="text-xs font-bold uppercase tracking-[0.14em] text-faint">
        Sources consulted
      </h3>
      <ul className="mt-3 space-y-2">
        {sources.map((source, i) => (
          <li key={i}>
            {isUrl(source) ? (
              <a
                href={source}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-start gap-2 break-all rounded-lg border border-line bg-surface2 px-3.5 py-2.5 text-sm text-muted transition-colors hover:border-accent/50 hover:text-accent"
              >
                <LinkIcon size={14} className="mt-1 shrink-0" />
                <span className="font-mono text-xs leading-relaxed">{source}</span>
              </a>
            ) : (
              <p className="rounded-lg border border-line bg-surface2 px-3.5 py-2.5 text-sm leading-relaxed text-muted">
                {source}
              </p>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
