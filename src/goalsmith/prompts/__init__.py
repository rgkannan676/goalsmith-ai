"""Versioned prompt templates.

Each prompt is a Markdown file in this folder with a small YAML header:

    ---
    name: agent_system
    version: 2
    description: What the prompt is for.
    ---
    The prompt text...

Bump `version` whenever you change the text, and note why in prompts/CHANGELOG.md.
Every trace records the prompt's name, version and a short hash of its text, so a
change in behaviour can be traced to the prompt that caused it (and an edit
without a version bump shows up as a new hash).

The files live inside the package, not at the repo root, so they ship with it
(e.g. in the Docker image) and load the same way from any working directory.
"""

import hashlib
from functools import cache
from importlib import resources

import yaml
from pydantic import BaseModel, ConfigDict, Field

_DELIMITER = "---"


class PromptError(ValueError):
    """A prompt file is missing or malformed."""


class Prompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: int = Field(ge=1)
    description: str = ""
    text: str

    @property
    def sha(self) -> str:
        """First 12 hex characters of the text's SHA-256."""
        return hashlib.sha256(self.text.encode()).hexdigest()[:12]

    @property
    def ref(self) -> str:
        """e.g. `agent_system@v2`."""
        return f"{self.name}@v{self.version}"

    def format(self, **values: str) -> str:
        """Fill `{placeholders}` in the text."""
        return self.text.format(**values)

    def trace_metadata(self) -> dict[str, str]:
        return {
            "prompt_name": self.name,
            "prompt_version": str(self.version),
            "prompt_sha": self.sha,
        }


def parse_prompt(source: str, *, expected_name: str | None = None) -> Prompt:
    parts = source.split(_DELIMITER, 2)
    if len(parts) != 3 or parts[0].strip():
        raise PromptError("A prompt file must start with a '---' YAML header and end it with '---'")
    header = yaml.safe_load(parts[1]) or {}
    if not isinstance(header, dict):
        raise PromptError("The prompt header must be a YAML mapping")
    prompt = Prompt.model_validate({**header, "text": parts[2].strip()})
    if expected_name and prompt.name != expected_name:
        raise PromptError(f"File {expected_name}.md declares name {prompt.name!r}")
    return prompt


@cache
def load_prompt(name: str) -> Prompt:
    """Load `<name>.md` from this package. Cached: prompts are read once per process."""
    file = resources.files(__package__).joinpath(f"{name}.md")
    if not file.is_file():
        raise PromptError(f"No prompt named {name!r}")
    return parse_prompt(file.read_text(encoding="utf-8"), expected_name=name)


def available_prompts() -> list[str]:
    return sorted(
        f.name.removesuffix(".md")
        for f in resources.files(__package__).iterdir()
        if f.name.endswith(".md") and f.name != "CHANGELOG.md"
    )
