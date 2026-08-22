import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "app"
sys.path.insert(0, str(APP_DIR))

import uvicorn

from config import settings

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        app_dir=str(APP_DIR),
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.APP_DEBUG,
    )
