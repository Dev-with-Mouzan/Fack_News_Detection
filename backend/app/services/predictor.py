import logging

from models.schemas import NewsResponse

logger = logging.getLogger(__name__)


async def predict_news(
    text: str,
    provider: str | None = None,
    api_key: str | None = None,
) -> NewsResponse:
    """Run the full AI pipeline: search + LLM fact-check.

    provider/api_key come from the frontend AI settings; when omitted the
    server falls back to its env-configured credentials.
    """
    # Import here to avoid circular imports and allow lazy LLM init
    from core.llm import get_llm, SYSTEM_PROMPT
    from services.searcher import search_news

    try:
        llm = get_llm(provider, api_key)
    except RuntimeError as e:
        raise RuntimeError(str(e)) from e

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        (
            "user",
            "News article:\n{text}\n\n"
            "Internet search results:\n{search_results}\n\n"
            "Based on the search results above, determine if the news is TRUE or FALSE.",
        ),
    ])

    try:
        search_results, sources = await search_news(text, max_results=5)
        chain = prompt | llm.with_structured_output(NewsResponse)
        result: NewsResponse = await chain.ainvoke({
            "text": text,
            "search_results": search_results,
        })
        result.sources = sources
        return result
    except Exception as e:
        logger.exception("AI prediction failed")
        raise
