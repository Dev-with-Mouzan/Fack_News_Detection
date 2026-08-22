from collections.abc import AsyncGenerator

from services.predictor import predict_news


async def get_predictor() -> AsyncGenerator:
    yield predict_news
