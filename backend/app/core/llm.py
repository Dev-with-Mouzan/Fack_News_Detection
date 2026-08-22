import logging

from config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a professional fact-checking assistant with live internet search capability. "
    "Analyze the given news article and the provided web search results to determine "
    "if the news is likely TRUE or FALSE.\n\n"
    "Use the search results as evidence to support your verdict. "
    "If the search results contain authoritative sources confirming the claim, lean TRUE. "
    "If they contradict or debunk it, lean FALSE. "
    "If there is insufficient information, return Uncertain.\n\n"
    "Return your assessment with:\n"
    "- verdict: \"True\", \"False\", or \"Uncertain\"\n"
    "- confidence: a number between 0.0 and 1.0\n"
    "- explanation: a concise 2-3 sentence explanation referencing the search evidence"
)

# Cache of initialized LLMs keyed by (provider, api_key) so repeated requests
# with the same credentials reuse one client instance.
_llm_cache: dict[tuple[str, str], object] = {}

DEFAULT_PROVIDER = "gpt"


def _resolve(provider: str | None, api_key: str | None) -> tuple[str, str | None]:
    """Fall back to env-configured defaults when a request omits credentials."""
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider == "google":
        key = api_key or settings.GOOGLE_API_KEY
        if not key:
            raise RuntimeError(
                "No Google API key provided. Set one in the app's AI settings, "
                "or in backend/.env as GOOGLE_API_KEY."
            )
        return "google", key

    key = api_key or settings.OPENAI_API_KEY
    if not key:
        raise RuntimeError(
            "No OpenAI API key provided. Set one in the app's AI settings, "
            "or in backend/.env as OPENAI_API_KEY."
        )
    return "gpt", key


def get_llm(provider: str | None = None, api_key: str | None = None):
    """Return an LLM client for the requested provider and credentials.

    provider: 'gpt' (OpenAI ChatOpenAI) or 'google' (Gemini).
    Falls back to keys from backend/.env when api_key is omitted.
    Raises RuntimeError when no usable key is available.
    """
    resolved_provider, resolved_key = _resolve(provider, api_key)
    cache_key = (resolved_provider, resolved_key)

    if cache_key not in _llm_cache:
        if resolved_provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as e:
                raise RuntimeError(
                    "Google provider requires langchain-google-genai. "
                    "Install it with: pip install langchain-google-genai"
                ) from e
            _llm_cache[cache_key] = ChatGoogleGenerativeAI(
                model=settings.GOOGLE_MODEL,
                temperature=0.3,
                max_output_tokens=500,
                google_api_key=resolved_key,
            )
            logger.info("Gemini LLM initialized (model=%s)", settings.GOOGLE_MODEL)
        else:
            from langchain_openai import ChatOpenAI

            _llm_cache[cache_key] = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=0.3,
                max_tokens=500,
                api_key=resolved_key,
            )
            logger.info("OpenAI LLM initialized (model=%s)", settings.OPENAI_MODEL)

    return _llm_cache[cache_key]
