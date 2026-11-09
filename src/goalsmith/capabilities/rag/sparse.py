"""BM25 as sparse vectors, so keyword search runs inside Qdrant next to dense search.

BM25 scores a document d for a query q as

    sum over query terms t:   IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avg))

where tf is how often t appears in d, |d| is the length of d, and avg is the
average document length.

Split it in two:
- the term-frequency part depends only on the document, so it is computed here, at
  ingest time, and stored as the document's sparse vector (term id -> weight);
- IDF depends on the whole collection, so Qdrant computes it at query time
  (the collection's sparse vector is created with `modifier=IDF`).

A query vector is just 1.0 for each of its terms. Qdrant's dot product of the two
vectors, with IDF applied, is the BM25 score.
"""

import re
import zlib
from collections import Counter

from qdrant_client import models

K1 = 1.2  # how quickly repeated terms stop adding score
B = 0.75  # how much long chunks are penalised
AVG_LEN = 200  # rough average chunk length in tokens (1200 chars / ~6 chars per token)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORD_TEXT = """
a an and are as at be by can do does for from has have how i if in into is it its of on or
our that the their there these this to was we what when where which who why will with you your
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def term_id(token: str) -> int:
    """A stable id per term. crc32 fits Qdrant's uint32 indices; collisions are rare."""
    return zlib.crc32(token.encode())


def encode_document(text: str) -> models.SparseVector:
    tokens = tokenize(text)
    length_norm = 1 - B + B * len(tokens) / AVG_LEN
    weights = {
        term_id(term): tf * (K1 + 1) / (tf + K1 * length_norm)
        for term, tf in Counter(tokens).items()
    }
    return _sparse(weights)


def encode_query(text: str) -> models.SparseVector:
    return _sparse({term_id(term): 1.0 for term in set(tokenize(text))})


def _sparse(weights: dict[int, float]) -> models.SparseVector:
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[weights[i] for i in indices])
