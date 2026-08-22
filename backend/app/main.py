import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import router
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Serve the built frontend (frontend/dist). Falls back to frontend/ if the
# app hasn't been built yet, so the backend still runs pre-migration.
_FRONTEND_ROOT = Path(__file__).resolve().parent.parent.parent / "frontend"
FRONTEND_DIR = _FRONTEND_ROOT / "dist" if (_FRONTEND_ROOT / "dist").exists() else _FRONTEND_ROOT

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


# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
