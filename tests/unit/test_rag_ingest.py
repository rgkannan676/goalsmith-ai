from pathlib import Path

import pytest

from goalsmith.capabilities.rag.ingest import (
    IngestError,
    chunk_document,
    find_files,
    load_document,
    parse_html,
    parse_markdown,
    split_text,
)
from tests.fixtures.fake_rag import make_pdf

MARKDOWN = """\
---
author: someone
---
# Employee Handbook

Welcome to the company.

## Leave

### Parental leave

Up to 16 weeks of paid leave.

```python
# not a heading
x = 1
```

### Sick leave

Ten days per year.

## Expenses

Submit receipts within 30 days.
"""


def test_markdown_sections_follow_headings() -> None:
    doc = parse_markdown(MARKDOWN, "handbook.md")

    assert doc.title == "Employee Handbook"  # first H1 is the title, not a section
    assert [s.headings for s in doc.sections] == [
        (),
        ("Leave", "Parental leave"),
        ("Leave", "Sick leave"),
        ("Expenses",),
    ]
    assert doc.sections[0].text == "Welcome to the company."
    assert "# not a heading" in doc.sections[1].text  # fenced code is kept as text
    assert "author" not in doc.sections[0].text  # front matter is dropped


def test_markdown_without_h1_uses_file_name() -> None:
    assert parse_markdown("## A\n\ntext", "notes.md").title == "notes"


HTML = """\
<html><head><title>Travel Policy</title><script>var x = 1;</script></head>
<body>
<nav><a href="/">Home</a></nav>
<h2>Flights</h2>
<p>Economy class for flights under 6 hours.</p>
<ul><li><p>Book through the portal.</p></li><li>Keep the boarding pass.</li></ul>
<h3>Upgrades</h3>
<p>Business class needs VP approval.</p>
<footer>Copyright</footer>
</body></html>
"""


def test_html_sections_and_noise_removal() -> None:
    doc = parse_html(HTML, "travel.html")

    assert doc.title == "Travel Policy"
    assert [s.headings for s in doc.sections] == [("Flights",), ("Flights", "Upgrades")]
    flights = doc.sections[0].text
    assert "Economy class" in flights
    assert flights.count("Book through the portal.") == 1  # <p> inside <li> not doubled
    full = " ".join(s.text for s in doc.sections)
    assert "var x" not in full and "Home" not in full and "Copyright" not in full


def test_pdf_has_one_section_per_page(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(make_pdf([["Revenue grew 12%."], ["Churn fell to 3%."]], title="Q3 Report"))

    doc = load_document(path)

    assert doc.title == "Q3 Report"
    assert [s.page for s in doc.sections] == [1, 2]
    assert "Churn fell to 3%." in doc.sections[1].text


def test_unsupported_file_type(tmp_path: Path) -> None:
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"")
    with pytest.raises(IngestError, match="Unsupported"):
        load_document(path)


def test_chunks_carry_source_ids_and_metadata() -> None:
    doc = parse_markdown(MARKDOWN, "handbook.md")
    chunks = chunk_document(doc, "docs", max_chars=500, overlap=0)

    assert [c.source_id for c in chunks] == [f"handbook.md#{i}" for i in range(len(chunks))]
    parental = chunks[1]
    assert parental.section == "Leave > Parental leave"
    assert parental.collection == "docs"
    assert parental.context_text().startswith("Employee Handbook > Leave > Parental leave\n")


def test_chunks_never_cross_sections() -> None:
    doc = parse_markdown(MARKDOWN, "handbook.md")
    chunks = chunk_document(doc, "docs", max_chars=5000, overlap=0)
    assert len(chunks) == len(doc.sections)


def test_split_respects_max_chars() -> None:
    paragraphs = [f"Paragraph {i}. " + "word " * 40 for i in range(10)]
    pieces = split_text("\n\n".join(paragraphs), max_chars=500, overlap=0)
    assert len(pieces) > 1
    assert all(len(p) <= 500 for p in pieces)


def test_split_breaks_long_paragraphs_and_words() -> None:
    long_sentence = "x" * 50 + " " + "y " * 400
    pieces = split_text(long_sentence, max_chars=200, overlap=0)
    assert all(len(p) <= 200 for p in pieces)
    assert "".join(pieces).replace(" ", "").count("y") == 400


def test_split_overlaps_trailing_paragraph() -> None:
    text = "\n\n".join(["A" * 300, "B" * 300, "short tail", "C" * 300])
    pieces = split_text(text, max_chars=700, overlap=50)
    assert pieces[0].endswith("short tail")
    assert pieces[1].startswith("short tail")  # the boundary fact appears in both


def test_find_files_expands_folders(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.html").write_text("<p>b</p>")
    (tmp_path / "skip.png").write_bytes(b"")
    assert [p.name for p in find_files([tmp_path])] == ["a.md", "b.html"]


def test_find_files_rejects_missing_and_duplicate_names(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="No such file"):
        find_files([tmp_path / "nope.md"])

    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    (tmp_path / "x" / "same.md").write_text("one")
    (tmp_path / "y" / "same.md").write_text("two")
    with pytest.raises(IngestError, match="share a name"):
        find_files([tmp_path])
