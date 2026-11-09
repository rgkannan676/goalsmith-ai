"""Ingest: files -> structured sections -> chunks -> Qdrant.

Chunking is *structure-aware*: a chunk never spans two headings (or two PDF pages),
so each chunk is about one thing and carries its heading path as metadata.
"""

import asyncio
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from pypdf import PdfReader

from goalsmith.capabilities.rag.models import Chunk, Document, Section
from goalsmith.capabilities.rag.sparse import encode_document
from goalsmith.capabilities.rag.store import DocStore


class IngestError(ValueError):
    """A file cannot be ingested (missing, unsupported type, duplicate name)."""


# --- parsing ---------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = ("```", "~~~")


class _Outline:
    """Tracks the current heading path while a document is read top to bottom."""

    def __init__(self) -> None:
        self.title: str | None = None
        self._stack: list[tuple[int, str]] = []
        self._buffer: list[str] = []
        self.sections: list[Section] = []

    def heading(self, level: int, text: str) -> None:
        self.flush()
        if level == 1 and self.title is None:
            self.title = text  # the first H1 is the document title, not a section
            return
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, text))

    def add(self, text: str) -> None:
        self._buffer.append(text)

    def flush(self) -> None:
        text = "\n".join(self._buffer).strip()
        self._buffer = []
        if text:
            headings = tuple(name for _, name in self._stack)
            self.sections.append(Section(headings=headings, text=text))


def parse_markdown(text: str, source: str) -> Document:
    outline = _Outline()
    in_fence = False
    for line in _strip_front_matter(text).splitlines():
        if line.lstrip().startswith(_FENCE):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            outline.heading(len(match.group(1)), match.group(2))
        else:
            outline.add(line)
    outline.flush()
    return Document(
        source=source, title=outline.title or Path(source).stem, sections=tuple(outline.sections)
    )


def _strip_front_matter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :]
    return text


_HTML_BLOCKS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "td", "th"]
_HTML_NOISE = ["script", "style", "noscript", "nav", "footer", "template"]


def parse_html(html: str, source: str) -> Document:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_HTML_NOISE):
        tag.decompose()

    outline = _Outline()
    root = soup.body or soup
    for element in root.find_all(_HTML_BLOCKS):
        # A <p> inside an <li> is already part of the <li>'s text.
        if not isinstance(element, Tag) or element.find_parent(_HTML_BLOCKS):
            continue
        if element.name[0] == "h":
            outline.heading(int(element.name[1]), element.get_text(" ", strip=True))
        else:
            separator = "" if element.name == "pre" else " "
            text = element.get_text(separator, strip=element.name != "pre")
            outline.add(text + "\n")  # blank line between blocks = paragraph break
    outline.flush()

    page_title = soup.title.get_text(strip=True) if soup.title else ""
    title = outline.title or page_title or Path(source).stem
    return Document(source=source, title=title, sections=tuple(outline.sections))


def parse_pdf(path: Path, source: str) -> Document:
    """One section per page. PDFs rarely keep real heading structure, but pages are
    stable and are what a reader expects in a citation."""
    reader = PdfReader(path)
    sections = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(Section(headings=(), text=text, page=number))
    title = (reader.metadata.title if reader.metadata else None) or Path(source).stem
    return Document(source=source, title=title, sections=tuple(sections))


def parse_text(text: str, source: str) -> Document:
    section = Section(headings=(), text=text.strip())
    return Document(source=source, title=Path(source).stem, sections=(section,))


_PARSERS: dict[str, Callable[[Path], Document]] = {
    ".md": lambda p: parse_markdown(p.read_text(encoding="utf-8"), p.name),
    ".markdown": lambda p: parse_markdown(p.read_text(encoding="utf-8"), p.name),
    ".html": lambda p: parse_html(p.read_text(encoding="utf-8"), p.name),
    ".htm": lambda p: parse_html(p.read_text(encoding="utf-8"), p.name),
    ".pdf": lambda p: parse_pdf(p, p.name),
    ".txt": lambda p: parse_text(p.read_text(encoding="utf-8"), p.name),
}
SUPPORTED_SUFFIXES = frozenset(_PARSERS)


def load_document(path: Path) -> Document:
    parser = _PARSERS.get(path.suffix.lower())
    if parser is None:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise IngestError(f"Unsupported file type {path.name!r}. Supported: {supported}")
    return parser(path)


# --- chunking --------------------------------------------------------------------


def chunk_document(doc: Document, collection: str, *, max_chars: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        for text in split_text(section.text, max_chars=max_chars, overlap=overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    source_id=f"{doc.source}#{index}",
                    source=doc.source,
                    collection=collection,
                    index=index,
                    title=doc.title,
                    section=" > ".join(section.headings),
                    page=section.page,
                    text=text,
                )
            )
    return chunks


def split_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Pack paragraphs into chunks of at most `max_chars`.

    Long paragraphs are split into sentences, and very long sentences into word
    windows, so no unit is larger than a chunk. Consecutive chunks share trailing
    units totalling at most `overlap` characters, so a fact on a boundary is not lost.
    """
    chunks: list[str] = []
    current: list[str] = []
    for unit in _units(text, max_chars):
        if current and len(_join([*current, unit])) > max_chars:
            chunks.append(_join(current))
            tail = _tail(current, overlap)
            current = tail if len(_join([*tail, unit])) <= max_chars else []
        current.append(unit)
    if current:
        chunks.append(_join(current))
    return chunks


def _join(units: Sequence[str]) -> str:
    return "\n\n".join(units)


def _units(text: str, max_chars: int) -> Iterable[str]:
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            yield paragraph
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            yield from _windows(sentence, max_chars)


def _windows(sentence: str, max_chars: int) -> Iterable[str]:
    words: list[str] = []
    for word in sentence.split():
        if words and len(" ".join([*words, word])) > max_chars:
            yield " ".join(words)
            words = []
        words.append(word[:max_chars])
    if words:
        yield " ".join(words)


def _tail(units: Sequence[str], overlap: int) -> list[str]:
    tail: list[str] = []
    for unit in reversed(units):
        if len(_join([unit, *tail])) > overlap:
            break
        tail.insert(0, unit)
    return tail


# --- the pipeline ----------------------------------------------------------------


class IngestResult(BaseModel):
    source: str
    chunks: int


def find_files(paths: Iterable[Path]) -> list[Path]:
    """Expand folders into the supported files inside them. Explicit files must exist."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
            )
        elif path.is_file():
            files.append(path)
        else:
            raise IngestError(f"No such file or folder: {path}")

    names = [f.name for f in files]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        # The file name is the source id prefix, so it must be unique per ingest.
        raise IngestError(f"Two files share a name: {', '.join(duplicates)}. Rename one.")
    return files


async def ingest_files(
    paths: Iterable[Path],
    *,
    collection: str,
    store: DocStore,
    embeddings: Embeddings,
    max_chars: int,
    overlap: int,
) -> list[IngestResult]:
    """Parse, chunk and index files. Re-ingesting a file replaces its old chunks."""
    results = []
    for path in find_files(paths):
        doc = await asyncio.to_thread(load_document, path)
        chunks = chunk_document(doc, collection, max_chars=max_chars, overlap=overlap)
        texts = [chunk.context_text() for chunk in chunks]
        dense = await embeddings.aembed_documents(texts) if texts else []
        if dense:
            await store.ensure_collection(collection, dim=len(dense[0]))
        if await store.has_collection(collection):
            sparse = [encode_document(text) for text in texts]
            await store.replace_source(collection, doc.source, chunks, dense, sparse)
        results.append(IngestResult(source=doc.source, chunks=len(chunks)))
    return results
