from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    TRUE = "True"
    FALSE = "False"
    UNCERTAIN = "Uncertain"


class NewsRequest(BaseModel):
    text: str = Field(..., min_length=1, description="News article text or URL")
    # Optional per-request AI credentials (set from the frontend settings panel).
    provider: Literal["gpt", "google"] | None = Field(
        default=None, description="AI provider: 'gpt' (OpenAI) or 'google' (Gemini)"
    )
    api_key: str | None = Field(default=None, description="API key for the chosen provider")

class NewsResponse(BaseModel):
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list)


class MLPredictionResponse(BaseModel):
    label: str = Field(..., description="Real or Fake")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = Field(..., min_length=1)
    model: str = Field(default="XGBoost")


class CombinedRequest(BaseModel):
    text: str = Field(..., min_length=1, description="News article text")
    provider: Literal["gpt", "google"] | None = None
    api_key: str | None = None


class CombinedResponse(BaseModel):
    ml_prediction: MLPredictionResponse
    ai_verification: NewsResponse | None = None
    final_verdict: str = Field(..., description="Combined verdict")
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., min_length=1)
    sources: list[str] = Field(default_factory=list)
