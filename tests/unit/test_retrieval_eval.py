from pathlib import Path

import pytest

from goalsmith.capabilities.rag.models import Chunk, SearchHit
from goalsmith.config import Settings
from goalsmith.evals.retrieval import (
    LabelledQuery,
    Relevant,
    gains,
    hit_rate,
    load_queries,
    ndcg,
    reciprocal_rank,
    render,
    run_eval,
)
from tests.fixtures.fake_rag import HashEmbeddings, KeywordReranker

REPO = Path(__file__).resolve().parents[2]


def hit(source: str, section: str = "", page: int | None = None) -> SearchHit:
    chunk = Chunk(
        source_id=f"{source}#0",
        source=source,
        collection="c",
        index=0,
        title="t",
        section=section,
        page=page,
        text="x",
    )
    return SearchHit(chunk=chunk, score=0.0)


def test_relevant_matches_section_and_page() -> None:
    assert Relevant(source="a.md", section="Leave").matches(hit("a.md", "Policy > Leave").chunk)
    assert not Relevant(source="a.md", section="Leave").matches(hit("b.md", "Leave").chunk)
    assert Relevant(source="r.pdf", page=2).matches(hit("r.pdf", page=2).chunk)
    assert not Relevant(source="r.pdf", page=2).matches(hit("r.pdf", page=3).chunk)


def test_gains_count_each_label_once() -> None:
    labels = [Relevant(source="a.md", section="Leave")]
    hits = [hit("b.md"), hit("a.md", "Leave"), hit("a.md", "Leave")]
    assert gains(hits, labels) == [0, 1, 0]


def test_metrics() -> None:
    assert hit_rate([0, 0, 1]) == 1.0
    assert hit_rate([0, 0, 0]) == 0.0
    assert reciprocal_rank([0, 0, 1, 0]) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 0]) == 0.0
    assert ndcg([1, 0, 0], n_relevant=1) == 1.0
    assert ndcg([0, 1, 0], n_relevant=1) == pytest.approx(1 / 1.585, rel=1e-3)
    assert ndcg([0, 0, 0], n_relevant=1) == 0.0


def test_labelled_queries_file_is_valid() -> None:
    queries = load_queries(REPO / "evals/retrieval/queries.yaml")
    assert len(queries) == 40
    assert len({q.id for q in queries}) == 40
    sources = {p.name for p in (REPO / "evals/retrieval/corpus").iterdir()}
    assert {r.source for q in queries for r in q.relevant} <= sources


async def test_run_eval_end_to_end_offline(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\n## Leave\n\n18 weeks of parental leave.\n")
    (tmp_path / "b.md").write_text("# B\n\n## Codes\n\nError E-1042 means no wifi.\n")
    queries = [
        LabelledQuery(
            id="q1", query="parental leave", relevant=[Relevant(source="a.md", section="Leave")]
        ),
        LabelledQuery(id="q2", query="E-1042", relevant=[Relevant(source="b.md")]),
    ]
    scores = await run_eval(tmp_path, queries, HashEmbeddings(), KeywordReranker("wifi"), k=1)

    by_setup = {s.setup: s for s in scores}
    assert set(by_setup) == {"dense", "bm25", "hybrid (RRF)", "hybrid + rerank"}
    assert by_setup["bm25"].hit_at_k == 1.0
    assert by_setup["hybrid + rerank"].misses == ["q1"]  # the reranker always prefers "wifi"

    report = render(scores, k=1, n_queries=2, settings=Settings(_env_file=None))
    assert "| hybrid + rerank |" in report
    assert "hit@1" in report
