"""Offline stand-ins for the RAG models, plus a tiny PDF writer for parser tests."""

import math
import zlib
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from goalsmith.capabilities.rag.sparse import tokenize


class HashEmbeddings(Embeddings):
    """Bag-of-words vectors hashed into `dim` buckets.

    Not semantic, but deterministic and offline, and texts that share words get
    similar vectors, which is enough to test the plumbing around real embeddings.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.queries: list[str] = []

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            vector[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._embed(text)


class KeywordReranker:
    """Scores 1.0 for documents containing `keyword`, else 0.0."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.calls: list[tuple[str, int]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, len(documents)))
        return [1.0 if self.keyword in doc else 0.0 for doc in documents]


def make_pdf(pages: Sequence[Sequence[str]], title: str | None = None) -> bytes:
    """A minimal valid PDF: one text line per string, one list of lines per page."""

    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = []
    page_ids = [4 + 2 * i for i in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for pid, lines in zip(page_ids, pages, strict=True):
        body = "".join(f"({escape(line)}) Tj T* " for line in lines)
        stream = f"BT /F1 11 Tf 14 TL 72 760 Td {body}ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {pid + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    info_id = None
    if title is not None:
        objects.append(f"<< /Title ({escape(title)}) >>".encode())
        info_id = len(objects)

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, obj)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    info = f" /Info {info_id} 0 R" if info_id else ""
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R{info} >>\n".encode()
    out += b"startxref\n%d\n%%%%EOF\n" % xref
    return bytes(out)
