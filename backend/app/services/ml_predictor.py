import logging
import os
import re
import string
from pathlib import Path

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from models.schemas import MLPredictionResponse

logger = logging.getLogger(__name__)

# Paths to saved model files
_MODELS_DIR = Path(__file__).resolve().parent.parent / "ml_models"
_MODEL_PATH = _MODELS_DIR / "final_model(XGBoost).pkl"
_VECTORIZER_PATH = _MODELS_DIR / "vectorizer.pkl"

# On read-only serverless filesystems (e.g. Vercel), NLTK_DATA points to a
# writable /tmp directory — create it before any download attempt.
_NLTK_DIR = os.environ.get("NLTK_DATA")
if _NLTK_DIR:
    try:
        Path(_NLTK_DIR).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# Download required NLTK data (idempotent)
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(_pkg, quiet=True)

# Load model and vectorizer once at startup
_model = None
_vectorizer = None


def _load_models():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        try:
            _model = joblib.load(_MODEL_PATH)
            _vectorizer = joblib.load(_VECTORIZER_PATH)
            logger.info("ML model and vectorizer loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load ML model or vectorizer: %s", e)
            raise RuntimeError(f"Could not load ML models: {e}") from e


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


def _preprocess(text: str) -> str:
    cleaned = _clean_text(text)
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(cleaned)
    return " ".join(w for w in tokens if w.lower() not in stop_words)


async def predict_ml(text: str) -> MLPredictionResponse:
    """Run the local XGBoost model on the given news text."""
    _load_models()

    cleaned = _preprocess(text)
    vectorized = _vectorizer.transform([cleaned])
    prediction = int(_model.predict(vectorized)[0])

    label = "Real" if prediction == 1 else "Fake"

    return MLPredictionResponse(
        label=label,
        confidence=0.0,  # XGBoost doesn't expose predict_proba by default here
        explanation=(
            f"The ML model classified this article as **{label}** news "
            "based on textual patterns learned from a dataset of real and fake news articles."
        ),
        model="XGBoost",
    )
