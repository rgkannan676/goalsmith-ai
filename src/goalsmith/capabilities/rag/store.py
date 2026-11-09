"""DocStore: the only code that talks to Qdrant.

Each Goalsmith collection is one Qdrant collection with two vectors per point:
- "dense": the embedding, compared by cosine similarity;
- "bm25": a sparse BM25 vector; Qdrant applies IDF at query time (see sparse.py).
The chunk itself is stored as the point's payload.
"""

import re
import uuid
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient, models

from goalsmith.capabilities.rag.models import Chunk, SearchHit
from goalsmith.config import RagSettings

DENSE = "dense"
SPARSE = "bm25"

_COLLECTION_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
# Same source id -> same point id, so re-ingesting overwrites instead of duplicating.
_POINT_NAMESPACE = uuid.UUID("6f1c2a8e-3b7d-4c55-9a0e-5d2f8b1e7c40")


class StoreError(ValueError):
    """A collection is missing, misnamed, or built with a different embedding size."""


def open_client(settings: RagSettings) -> AsyncQdrantClient:
    """Qdrant in Docker by default, or embedded in this process if `qdrant_path` is set."""
    if settings.qdrant_path is not None:
        return AsyncQdrantClient(path=str(settings.qdrant_path))
    return AsyncQdrantClient(url=settings.qdrant_url, timeout=10)


def check_collection_name(name: str) -> str:
    if not _COLLECTION_NAME.match(name):
        raise StoreError(
            f"Invalid collection name {name!r}: use lowercase letters, digits, '-' and '_'."
        )
    return name


class DocStore:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def collections(self) -> list[str]:
        response = await self._client.get_collections()
        return sorted(c.name for c in response.collections)

    async def has_collection(self, name: str) -> bool:
        return await self._client.collection_exists(check_collection_name(name))

    async def ensure_collection(self, name: str, *, dim: int) -> None:
        """Create the collection if needed. An existing one must use the same vector size."""
        if await self.has_collection(name):
            info = await self._client.get_collection(name)
            vectors = info.config.params.vectors
            size = vectors[DENSE].size if isinstance(vectors, dict) else None
            if size != dim:
                raise StoreError(
                    f"Collection {name!r} holds {size}-dim vectors but the embedding model "
                    f"makes {dim}-dim ones. Ingest into a new collection, or delete this one."
                )
            return
        await self._client.create_collection(
            name,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )

    async def replace_source(
        self,
        collection: str,
        source: str,
        chunks: Sequence[Chunk],
        dense: Sequence[Sequence[float]],
        sparse: Sequence[models.SparseVector],
    ) -> None:
        """Delete every chunk of `source`, then write the new ones."""
        await self._client.delete(collection, points_selector=_source_filter(source), wait=True)
        if not chunks:
            return
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_POINT_NAMESPACE, f"{collection}/{chunk.source_id}")),
                vector={DENSE: list(d), SPARSE: s},
                payload=chunk.model_dump(),
            )
            for chunk, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        await self._client.upsert(collection, points=points, wait=True)

    async def dense_search(
        self, collection: str, vector: Sequence[float], limit: int
    ) -> list[SearchHit]:
        return await self._query(collection, list(vector), DENSE, limit)

    async def sparse_search(
        self, collection: str, vector: models.SparseVector, limit: int
    ) -> list[SearchHit]:
        if not vector.indices:  # a query of only stopwords matches nothing
            return []
        return await self._query(collection, vector, SPARSE, limit)

    async def _query(
        self, collection: str, query: list[float] | models.SparseVector, using: str, limit: int
    ) -> list[SearchHit]:
        response = await self._client.query_points(
            collection, query=query, using=using, limit=limit, with_payload=True
        )
        return [
            SearchHit(chunk=Chunk.model_validate(point.payload), score=point.score)
            for point in response.points
        ]


def _source_filter(source: str) -> models.FilterSelector:
    return models.FilterSelector(
        filter=models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
        )
    )
