"""remember and recall on a LangGraph Store.

A Store is a key-value database organised by *namespace* (a tuple, like a folder
path). Each user's memories live under ("users", <user_id>, "memories"). With an
`index` config, the Store also embeds each memory's text, so `recall` can search
by meaning: "what does the user like to drink?" finds "prefers green tea".

Checkpoints (M4) remember one run. The Store remembers across runs.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from langchain_core.embeddings import Embeddings
from langgraph.store.base import BaseStore, IndexConfig
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from pydantic import BaseModel


class MemoryStoreError(RuntimeError):
    """The saved memories were embedded with a different vector size than the current model."""


class MemoryItem(BaseModel):
    id: str
    text: str
    created_at: str
    score: float | None = None  # similarity to the query, for recall results


class Memory:
    """One user's memories."""

    def __init__(self, store: BaseStore, user_id: str) -> None:
        self._store = store
        self.namespace = ("users", user_id, "memories")

    async def remember(self, text: str) -> MemoryItem:
        key = uuid.uuid4().hex
        await self._store.aput(self.namespace, key, {"text": text})
        item = await self._store.aget(self.namespace, key)
        created = item.created_at.isoformat() if item else ""
        return MemoryItem(id=key, text=text, created_at=created)

    async def recall(self, query: str, k: int = 5) -> list[MemoryItem]:
        results = await self._store.asearch(self.namespace, query=query, limit=k)
        return [
            MemoryItem(
                id=r.key,
                text=str(r.value.get("text", "")),
                created_at=r.created_at.isoformat(),
                score=r.score,
            )
            for r in results
        ]


def index_config(embeddings: Embeddings, dims: int) -> IndexConfig:
    """Embed the "text" field of every memory so it can be searched by meaning."""
    return {"dims": dims, "embed": embeddings, "fields": ["text"]}


async def embedding_dims(embeddings: Embeddings) -> int:
    """The vector size, found by embedding a probe (pgvector needs it up front)."""
    return len(await embeddings.aembed_query("dimension probe"))


def in_memory_store(embeddings: Embeddings, dims: int) -> InMemoryStore:
    return InMemoryStore(index=index_config(embeddings, dims))


@asynccontextmanager
async def postgres_store(
    conn_string: str, embeddings: Embeddings, dims: int
) -> AsyncIterator[AsyncPostgresStore]:
    """A Store in Postgres; vectors use the pgvector extension. Creates tables if needed."""
    existing = await stored_vector_dims(conn_string)
    if existing is not None and existing != dims:
        # pgvector fixes the size when the table is created, so this would fail on
        # the first `remember` with a cryptic error. Fail early and clearly instead.
        raise MemoryStoreError(
            f"Saved memories use {existing}-dim vectors but the embedding model makes "
            f"{dims}-dim ones. Use the original embedding model, or a fresh database."
        )
    async with AsyncPostgresStore.from_conn_string(
        conn_string, index={"dims": dims, "embed": embeddings, "fields": ["text"]}
    ) as store:
        await store.setup()
        yield store


async def stored_vector_dims(conn_string: str) -> int | None:
    """Vector size of the existing memory table, or None if it does not exist yet.
    (For a pgvector column, Postgres keeps the size in `atttypmod`.)"""
    query = (
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = to_regclass('store_vectors') AND attname = 'embedding'"
    )
    async with await psycopg.AsyncConnection.connect(conn_string) as conn:
        row = await (await conn.execute(query)).fetchone()
    return int(row[0]) if row else None
