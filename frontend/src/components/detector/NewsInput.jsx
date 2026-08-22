import { SearchIcon, SpinnerIcon } from "../ui/icons.jsx";

/**
 * Article input card: labeled textarea, live char count,
 * keyboard hint, and Clear / Analyze actions.
 */
export default function NewsInput({
  value,
  onChange,
  onSubmit,
  onClear,
  loading,
}) {
  const empty = value.trim().length === 0;

  return (
    <section aria-label="Article input" className="card mt-5 p-5 sm:p-6">
      <div className="mb-3 flex items-baseline justify-between">
        <label htmlFor="news-text" className="text-xs font-bold uppercase tracking-[0.14em] text-faint">
          Article text
        </label>
        <span className="font-mono text-xs tabular-nums text-faint">
          {value.length.toLocaleString()} chars
        </span>
      </div>

      <textarea
        id="news-text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            if (!empty && !loading) onSubmit();
          }
        }}
        placeholder="Paste a headline or a full article here…"
        rows={7}
        spellCheck
        className="w-full resize-y rounded-xl border border-line bg-surface2 p-4 text-sm leading-relaxed outline-none transition-all duration-200 placeholder:text-faint focus:border-accent focus:bg-surface focus:ring-4 focus:ring-accent/15"
      />

      <div className="mt-4 flex flex-col-reverse items-stretch justify-between gap-3 sm:flex-row sm:items-center">
        <p className="flex items-center justify-center gap-1.5 text-xs text-faint sm:justify-start">
          <kbd className="kbd">Ctrl</kbd> + <kbd className="kbd">Enter</kbd> to analyze
        </p>

        <div className="flex gap-2.5">
          <button type="button" onClick={onClear} disabled={empty || loading} className="btn-ghost flex-1 sm:flex-none">
            Clear
          </button>
          <button type="button" onClick={onSubmit} disabled={empty || loading} className="btn-primary flex-1 sm:flex-none">
            {loading ? (
              <>
                <SpinnerIcon size={15} />
                Analyzing…
              </>
            ) : (
              <>
                <SearchIcon size={15} />
                Analyze
              </>
            )}
          </button>
        </div>
      </div>
    </section>
  );
}
