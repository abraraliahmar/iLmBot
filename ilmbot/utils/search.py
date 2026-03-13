"""
utils/search.py
───────────────
Live web search via the Tavily API.
"""

from config.config import TAVILY_API_KEY


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Search the web using Tavily and return structured results.

    Args:
        query: The search query string.
        max_results: Number of results to return.

    Returns:
        List of dicts with keys: title, url, snippet.
        Returns an empty list on failure.
    """
    if not TAVILY_API_KEY:
        return [{"title": "⚠️ Tavily API key not configured",
                 "url": "", "snippet": "Set TAVILY_API_KEY in your .env file."}]

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, max_results=max_results)
        results: list[dict] = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            })
        return results
    except ImportError:
        return [{"title": "⚠️ tavily-python not installed",
                 "url": "", "snippet": "pip install tavily-python"}]
    except Exception as exc:
        return [{"title": "⚠️ Web search failed",
                 "url": "", "snippet": str(exc)}]
