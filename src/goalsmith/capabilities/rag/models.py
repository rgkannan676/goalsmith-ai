"""Data shapes shared by ingest, storage and retrieval."""

from pydantic import BaseModel, ConfigDict


class Section(BaseModel):
    """A piece of a document under one heading (or one PDF page), before chunking."""

    model_config = ConfigDict(frozen=True)

    headings: tuple[str, ...]  # e.g. ("Leave", "Parental leave")
    text: str
    page: int | None = None


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str  # the file name; also the prefix of every source id
    title: str
    sections: tuple[Section, ...]


class Chunk(BaseModel):
    """The unit that is indexed, retrieved and cited."""

    model_config = ConfigDict(frozen=True)

    source_id: str  # "<source>#<index>", e.g. "handbook.md#3". This is what answers cite.
    source: str
    collection: str
    index: int
    title: str
    section: str  # headings joined with " > "
    page: int | None
    text: str

    def context_text(self) -> str:
        """The text that is embedded and BM25-indexed.

        Prefixing the title and section gives a chunk context it lacks on its own:
        "Up to 16 weeks" means little until you know it is under "Parental leave".
        """
        heading = " > ".join(part for part in (self.title, self.section) if part)
        return f"{heading}\n{self.text}" if heading else self.text

    def location(self) -> str:
        """Human-readable place in the document, for tool output."""
        parts = [" > ".join(part for part in (self.title, self.section) if part)]
        if self.page is not None:
            parts.append(f"page {self.page}")
        return ", ".join(part for part in parts if part)


class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
