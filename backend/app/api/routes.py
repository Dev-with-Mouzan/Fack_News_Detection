import logging

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_predictor
from models.schemas import (
    CombinedRequest,
    CombinedResponse,
    MLPredictionResponse,
    NewsRequest,
    NewsResponse,
)
from services.ml_predictor import predict_ml

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict/ai",
    response_model=NewsResponse,
    summary="AI-powered fact check (GPT + web search)",
    description="Searches the web and uses GPT to verify a news article.",
)
async def predict_ai(
    request: NewsRequest,
    predictor=Depends(get_predictor),
) -> NewsResponse:
    try:
        return await predictor(request.text, request.provider, request.api_key)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        # Missing/invalid API key or provider package — client-fixable.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("AI prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction service unavailable. Please try again later.",
        )


@router.post(
    "/predict/ml",
    response_model=MLPredictionResponse,
    summary="ML model prediction (local XGBoost)",
    description="Uses a pre-trained XGBoost model to classify news as Real or Fake.",
)
async def predict_ml_endpoint(request: NewsRequest) -> MLPredictionResponse:
    try:
        return await predict_ml(request.text)
    except Exception as e:
        logger.exception("ML prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ML prediction failed: {e}",
        )


@router.post(
    "/predict",
    response_model=CombinedResponse,
    summary="Combined ML + AI prediction",
    description="Runs the local ML model and optionally the AI fact-checker for a comprehensive verdict.",
)
async def predict_combined(
    request: CombinedRequest,
    predictor=Depends(get_predictor),
) -> CombinedResponse:
    try:
        # Step 1: ML prediction
        ml_result = await predict_ml(request.text)

        # Step 2: AI verification (best-effort — won't fail if the LLM is down)
        ai_result = None
        try:
            ai_result = await predictor(request.text, request.provider, request.api_key)
        except Exception as e:
            logger.warning("AI verification skipped (unavailable): %s", e)

        # Step 3: Combine verdicts
        ml_label = ml_result.label.lower()

        if ai_result is not None:
            ai_verdict = ai_result.verdict.value.lower()
            # If both agree
            if (ml_label == "real" and ai_verdict == "true") or (
                ml_label == "fake" and ai_verdict == "false"
            ):
                final_verdict = ml_result.label
                confidence = max(ai_result.confidence, 0.85)
                explanation = (
                    f"Both the ML model and AI verification agree: this is **{final_verdict}** news. "
                    f"{ai_result.explanation}"
                )
                sources = ai_result.sources
            elif ai_verdict == "uncertain":
                final_verdict = ml_result.label
                confidence = 0.6
                explanation = (
                    f"The ML model classifies this as **{ml_result.label}** news, "
                    f"but the AI was unable to verify independently. "
                    f"{ai_result.explanation}"
                )
                sources = ai_result.sources
            else:
                # Disagreement — trust AI more for False verdicts
                if ai_verdict == "false":
                    final_verdict = "Fake"
                    confidence = ai_result.confidence
                    explanation = (
                        f"AI verification contradicts the ML model: the article appears to be **Fake**. "
                        f"{ai_result.explanation}"
                    )
                else:
                    final_verdict = ml_result.label
                    confidence = 0.7
                    explanation = (
                        f"ML says **{ml_result.label}**, AI says **{ai_verdict}**. "
                        f"Leaning towards the ML model. {ai_result.explanation}"
                    )
                sources = ai_result.sources
        else:
            # AI unavailable — ML only
            final_verdict = ml_result.label
            confidence = 0.6
            explanation = (
                f"AI verification is currently unavailable. "
                f"Based on the ML model alone, this is classified as **{final_verdict}** news. "
                f"{ml_result.explanation}"
            )
            sources = []

        return CombinedResponse(
            ml_prediction=ml_result,
            ai_verification=ai_result,
            final_verdict=final_verdict,
            confidence=confidence,
            explanation=explanation,
            sources=sources,
        )
    except Exception as e:
        logger.exception("Combined prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction service unavailable. Please try again later.",
        )
