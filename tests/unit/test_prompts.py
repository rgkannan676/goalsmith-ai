"""Versioned prompt files (src/goalsmith/prompts/)."""

import pytest

from goalsmith.prompts import Prompt, PromptError, available_prompts, load_prompt, parse_prompt


@pytest.mark.parametrize("name", available_prompts())
def test_every_prompt_file_loads(name: str) -> None:
    prompt = load_prompt(name)
    assert prompt.name == name
    assert prompt.version >= 1
    assert prompt.text and not prompt.text.startswith("---")


def test_the_known_prompts_exist() -> None:
    assert {"agent_system", "judge"} <= set(available_prompts())


def test_parse_reads_header_and_text() -> None:
    prompt = parse_prompt("---\nname: demo\nversion: 3\ndescription: d\n---\nHello {who}.\n")
    assert (prompt.name, prompt.version, prompt.text) == ("demo", 3, "Hello {who}.")
    assert prompt.ref == "demo@v3"
    assert prompt.format(who="Ada") == "Hello Ada."


def test_hash_changes_with_the_text_but_not_the_version() -> None:
    a = Prompt(name="p", version=1, text="one")
    assert a.sha == Prompt(name="p", version=2, text="one").sha
    assert a.sha != Prompt(name="p", version=1, text="two").sha
    assert a.trace_metadata() == {"prompt_name": "p", "prompt_version": "1", "prompt_sha": a.sha}


@pytest.mark.parametrize(
    "source",
    [
        "no header at all",
        "---\nname: x\n",  # header never closed
        "---\n- a list\n---\ntext",
        "---\nname: x\nversion: 0\n---\ntext",  # versions start at 1
    ],
)
def test_malformed_prompt_files_are_rejected(source: str) -> None:
    with pytest.raises((PromptError, ValueError)):
        parse_prompt(source)


def test_name_must_match_the_file() -> None:
    with pytest.raises(PromptError, match="declares name"):
        parse_prompt("---\nname: other\nversion: 1\n---\nx", expected_name="judge")


def test_unknown_prompt() -> None:
    with pytest.raises(PromptError, match="No prompt named"):
        load_prompt("does_not_exist")
