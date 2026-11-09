"""search_docs: hybrid search over ingested documents, with citable source ids."""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from goalsmith.capabilities.rag.models import SearchHit
from goalsmith.capabilities.rag.retrieve import HybridRetriever
from goalsmith.capabilities.rag.store import StoreError
from goalsmith.harness.tools import Tool, ToolContext, ToolError


def search_docs_tool(
    retriever: HybridRetriever,
    *,
    default_collection: str = "docs",
    collections: Sequence[str] = (),
) -> Tool:
    """`collections` (if known) are listed in the description so the model can pick one."""

    class SearchDocsArgs(BaseModel):
        query: str = Field(min_length=1, description="What to look for, in plain words")
        collection: str = Field(
            default=default_collection, description="Which document collection to search"
        )
        k: int = Field(default=5, ge=1, le=10, description="How many passages to return")

    async def search(args: SearchDocsArgs, ctx: ToolContext) -> str:
        try:
            hits = await retriever.search(args.query, args.collection, k=args.k)
        except StoreError as exc:
            raise ToolError(str(exc)) from None
        if not hits:
            return f"No passages found for {args.query!r} in {args.collection!r}."
        return "\n\n".join(_format(hit) for hit in hits)

    return Tool(
        name="search_docs",
        description=(
            "Search the user's ingested documents (handbooks, specs, reports). Returns "
            "passages, each starting with a source id in square brackets, e.g. "
            "[handbook.md#3]. Cite that id after every statement that relies on it."
            + (f" Collections: {', '.join(collections)}." if collections else "")
        ),
        args_schema=SearchDocsArgs,
        risk="read",
        fn=search,
    )


def _format(hit: SearchHit) -> str:
    chunk = hit.chunk
    return f"[{chunk.source_id}] {chunk.location()}\n{chunk.text}"
