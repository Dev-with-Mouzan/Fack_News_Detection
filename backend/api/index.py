"""Vercel serverless entrypoint (@vercel/python, ASGI).

Boots the FastAPI app defined in app/main.py and exposes it as `app`
(the variable name Vercel's Python runtime looks for).

The application code uses absolute imports relative to the `app/`
directory (e.g. `from api.routes import router`, `from config import
settings`), so that directory must be FIRST on sys.path — inserted here
before any app import so bundled modules always win over same-named
files living next to this entrypoint.
"""

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

# Serverless filesystems are read-only except /tmp — NLTK data must be
# downloaded there at cold-start (see services/ml_predictor.py).
os.environ.setdefault("NLTK_DATA", "/tmp/nltk_data")

from main import app  # noqa: E402  (FastAPI instance from app/main.py)

__all__ = ["app"]
