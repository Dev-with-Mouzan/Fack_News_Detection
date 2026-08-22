import asyncio
import logging
from functools import partial

from ddgs import DDGS

logger = logging.getLogger(__name__)


def _search_sync(query: str, max_results: int) -> list[dict]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No relevant search results found."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        href = r.get("href", "").strip()
        lines.append(f"{i}. {title}\n   {body}\n   Source: {href}")
    return "\n\n".join(lines)


def _extract_sources(results: list[dict]) -> list[str]:
    return [r.get("href", "").strip() for r in results if r.get("href")]


async def search_news(query: str, max_results: int = 5) -> tuple[str, list[str]]:
    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(
            None, partial(_search_sync, query, max_results)
        )
        return _format_results(results), _extract_sources(results)
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return "[Search unavailable]", []
