"""web_search: search the web through Tavily."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from goalsmith.harness.tools import Tool, ToolContext

_SNIPPET_CHARS = 500


class SearchClient(Protocol):
    """Anything with Tavily's `search` shape. Tests pass a fake; production passes Tavily."""

    async def search(self, query: str, *, max_results: int) -> dict[str, Any]: ...


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=1, description="What to search for, in plain words")
    max_results: int = Field(default=5, ge=1, le=10, description="How many results to return")


def tavily_client(api_key: str) -> SearchClient:
    from tavily import AsyncTavilyClient  # imported here so tests never need it

    client: SearchClient = AsyncTavilyClient(api_key=api_key)
    return client


def web_search_tool(client: SearchClient) -> Tool:
    async def search(args: WebSearchArgs, ctx: ToolContext) -> str:
        response = await client.search(args.query, max_results=args.max_results)
        results = response.get("results") or []
        if not results:
            return f"No results for {args.query!r}."
        return "\n\n".join(_format(i, r) for i, r in enumerate(results, start=1))

    return Tool(
        name="web_search",
        description=(
            "Search the web. Returns numbered results with title, URL and a snippet. "
            "Cite the URL of any result you rely on."
        ),
        args_schema=WebSearchArgs,
        risk="read",
        fn=search,
    )


def _format(index: int, result: dict[str, Any]) -> str:
    title = result.get("title") or "(untitled)"
    url = result.get("url") or ""
    snippet = str(result.get("content") or "").strip()
    if len(snippet) > _SNIPPET_CHARS:
        snippet = snippet[:_SNIPPET_CHARS].rstrip() + "..."
    return f"[{index}] {title}\nURL: {url}\n{snippet}"
