import logging
import os
from pathlib import Path

# Serverless filesystems are read-only except /tmp. Point NLTK at a writable
# dir before any NLTK-dependent module (services.ml_predictor) is imported.
os.environ.setdefault("NLTK_DATA", "/tmp/nltk_data")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from api.routes import router  # noqa: E402
from config import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Serve the built frontend (backend/public). Vite writes the production
# bundle here (see frontend/vite.config.js) so the assets ship inside the
# Vercel serverless function. Falls back to the legacy frontend/dist path
# for local runs done before the migration.
_PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"
_FRONTEND_ROOT = Path(__file__).resolve().parent.parent.parent / "frontend"

if _PUBLIC_DIR.exists():
    FRONTEND_DIR = _PUBLIC_DIR
elif (_FRONTEND_ROOT / "dist").exists():
    FRONTEND_DIR = _FRONTEND_ROOT / "dist"
else:
    FRONTEND_DIR = None

app = FastAPI(
    title="News Predictor API",
    version="2.0.0",
    description="Fake news detection combining ML (XGBoost) and AI (GPT + web search).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/api")
async def api_root():
    return {"message": "News Predictor API v2 is running", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}


from fastapi.responses import FileResponse

# Serve the embedded frontend. All non-API paths fall back to index.html so
# client-side routes (/about, /features, /detector, ...) work on refresh and
# on direct links. Real asset files (js/css/png) are resolved from disk.
if FRONTEND_DIR is not None and FRONTEND_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIR / "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        candidate = (FRONTEND_DIR / full_path).resolve()
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIR / "index.html")
