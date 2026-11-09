"""Where runs and memories are saved: Postgres when available, else this process.

- Checkpointer: LangGraph saves each run's state after every node, keyed by
  thread_id (our run_id). A paused run can be resumed from there, even after a restart.
- Store: long-term memory shared across runs (see memory/store.py).

Without Postgres everything still works, but only until the process exits.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import psycopg
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore

from goalsmith.core.serde import checkpoint_serializer
from goalsmith.memory.store import embedding_dims, in_memory_store, postgres_store


class PersistenceError(RuntimeError):
    """Durable storage was required but Postgres is not configured or not reachable."""


@dataclass(frozen=True)
class Persistence:
    checkpointer: BaseCheckpointSaver[Any]
    store: BaseStore
    durable: bool  # True: saved in Postgres and survives restarts


@asynccontextmanager
async def open_persistence(
    postgres_url: str | None, embeddings: Embeddings, *, require_durable: bool = False
) -> AsyncIterator[Persistence]:
    dims = await embedding_dims(embeddings)
    if postgres_url and await postgres_reachable(postgres_url):
        async with (
            AsyncPostgresSaver.from_conn_string(
                postgres_url, serde=checkpoint_serializer()
            ) as saver,
            postgres_store(postgres_url, embeddings, dims) as store,
        ):
            await saver.setup()  # creates the checkpoint tables on first use
            yield Persistence(checkpointer=saver, store=store, durable=True)
        return

    if require_durable:
        raise PersistenceError(
            "Postgres is not reachable. Start it with `make up` (or set GOALSMITH_POSTGRES_URL)."
        )
    yield Persistence(
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        store=in_memory_store(embeddings, dims),
        durable=False,
    )


async def postgres_reachable(url: str, timeout_s: int = 3) -> bool:
    try:
        conn = await psycopg.AsyncConnection.connect(url, connect_timeout=timeout_s)
    except psycopg.OperationalError:
        return False
    await conn.close()
    return True
