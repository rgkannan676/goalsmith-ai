from goalsmith.capabilities.rag.sparse import encode_document, encode_query, term_id, tokenize


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    assert tokenize("What is the HR-204 form?") == ["hr", "204", "form"]


def test_term_ids_are_stable() -> None:
    assert term_id("leave") == term_id("leave")
    assert term_id("leave") != term_id("leaves")


def weight(vector_text: str, term: str) -> float:
    vector = encode_document(vector_text)
    return dict(zip(vector.indices, vector.values, strict=True))[term_id(term)]


def test_repeated_terms_saturate() -> None:
    w1, w2, w3, w50 = (weight(" ".join(["leave"] * n + ["x"] * 50), "leave") for n in (1, 2, 3, 50))
    assert w1 < w2 < w3 < w50
    assert w3 - w2 < w2 - w1  # each extra repeat adds less (k1)
    assert w50 < 2.2  # never more than k1 + 1


def test_longer_chunks_weigh_a_term_less() -> None:
    assert weight("leave " + "x " * 10, "leave") > weight("leave " + "x " * 400, "leave")


def test_query_vector_is_unit_weights_sorted() -> None:
    vector = encode_query("parental leave leave policy")
    assert vector.values == [1.0, 1.0, 1.0]
    assert vector.indices == sorted(vector.indices)


def test_stopword_only_query_is_empty() -> None:
    assert encode_query("what is the").indices == []
