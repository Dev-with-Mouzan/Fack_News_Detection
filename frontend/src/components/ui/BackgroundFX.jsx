/**
 * Global ambient backdrop rendered once behind every page:
 * a slow-panning blueprint grid + drifting aurora blobs in brand greens.
 * Pure CSS keyframes (transform-only), disabled under prefers-reduced-motion.
 */
export default function BackgroundFX() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-0 overflow-hidden"
    >
      <div className="fx-grid" />
      <div className="fx-blob fx-blob-a" />
      <div className="fx-blob fx-blob-b" />
      <div className="fx-blob fx-blob-c" />
    </div>
  );
}
