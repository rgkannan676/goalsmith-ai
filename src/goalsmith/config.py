"""Settings read from environment variables (and `.env` for local dev).

Naming:
- Provider keys use their standard names, e.g. `ANTHROPIC_API_KEY`.
- Everything else is prefixed `GOALSMITH_`. Nested fields use `__`, e.g.
  `GOALSMITH_WORKER__MODEL=claude-haiku-5-5` changes only the worker's model.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, get_args

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from goalsmith.harness.tools import Risk

Role = Literal["planner", "worker", "router", "judge"]
"""What a model is used for. Code asks for a role, never for a model name."""

ROLES: tuple[Role, ...] = get_args(Role)


class ModelSpec(BaseModel):
    """Which model serves one role, and how it is called."""

    model_config = ConfigDict(frozen=True)

    provider: str = "anthropic"
    model: str
    temperature: float | None = None  # None = provider default
    max_tokens: int = 4096


class ModelPrice(BaseModel):
    """USD per million tokens. Cache writes are for the 5-minute cache."""

    model_config = ConfigDict(frozen=True)

    input: float
    output: float
    cache_read: float
    cache_write: float


# Anthropic list prices (checked 2026-11-10; see https://www.anthropic.com/pricing).
# Override with GOALSMITH_PRICES='{"model-id": {"input": ..., ...}}'.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(input=4.0, output=20.0, cache_read=0.20, cache_write=5.0),
    "claude-sonnet-5-5": ModelPrice(input=2.0, output=10.0, cache_read=0.20, cache_write=2.5),
    "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
}


class LangfuseSettings(BaseModel):
    """Where traces go (M5). Tracing is on only when both keys are set.

    The defaults match the self-hosted Langfuse in docker-compose.yml
    (`make obs-up`), which creates this project and these keys on first start.
    """

    model_config = ConfigDict(frozen=True)

    host: str = "http://localhost:3000"
    project_id: str = "goalsmith"  # only used to print links to traces
    public_key: str | None = None
    secret_key: SecretStr | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.public_key and self.secret_key and self.secret_key.get_secret_value())


class RagSettings(BaseModel):
    """Document search (M3). Env vars look like `GOALSMITH_RAG__QDRANT_URL`."""

    model_config = ConfigDict(frozen=True)

    qdrant_url: str = "http://localhost:6333"
    # Set a folder to run Qdrant inside this process instead of in Docker.
    qdrant_path: Path | None = None
    default_collection: str = "docs"

    # Local models run by fastembed. Downloaded once into model_cache_dir.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str | None = "Xenova/ms-marco-MiniLM-L-6-v2"  # None turns reranking off
    model_cache_dir: Path = Path(".cache/fastembed")

    chunk_chars: int = Field(default=1200, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)
    candidates: int = Field(default=20, ge=1)  # per retriever, before fusion and reranking


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GOALSMITH_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        populate_by_name=True,
        extra="ignore",
    )

    # Provider keys: optional here, checked only when a model is built.
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "GOALSMITH_ANTHROPIC_API_KEY"),
    )

    tavily_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("TAVILY_API_KEY", "GOALSMITH_TAVILY_API_KEY"),
    )

    # Runs
    workspace_root: Path = Path("workspaces")  # each run gets workspaces/<run_id>/
    max_steps: int = Field(default=15, ge=1)  # default model-call budget per run

    # Applied to every model call. The provider SDK retries rate limits,
    # timeouts and 5xx errors with exponential backoff.
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3

    # Model per role: strong for planning and judging, mid for work, small for routing.
    planner: ModelSpec = ModelSpec(model="claude-opus-5-5")
    worker: ModelSpec = ModelSpec(model="claude-sonnet-5-5")
    router: ModelSpec = ModelSpec(model="claude-haiku-5-5", max_tokens=1024)
    judge: ModelSpec = ModelSpec(model="claude-opus-5-5")

    rag: RagSettings = RagSettings()

    # Checkpoints and long-term memory (M4). Matches docker-compose.yml for local dev;
    # set it to an empty string to keep everything in memory.
    postgres_url: SecretStr | None = SecretStr(
        "postgresql://goalsmith:goalsmith@localhost:5432/goalsmith"
    )
    user_id: str = "local"  # whose memories `remember` and `recall` use
    # Tool risk levels that pause for a human. e.g. GOALSMITH_APPROVAL_RISKS='["external"]'
    approval_risks: frozenset[Risk] = frozenset({"write", "external"})

    # Observability (M5)
    langfuse: LangfuseSettings = LangfuseSettings()
    # Send prompts, tool arguments and outputs with traces. Turn off when runs
    # handle data that must not leave this machine.
    trace_content: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "WARNING"
    log_format: Literal["console", "json"] = "console"
    prices: dict[str, ModelPrice] = Field(default_factory=lambda: dict(DEFAULT_PRICES))

    def postgres_dsn(self) -> str | None:
        value = self.postgres_url.get_secret_value() if self.postgres_url else ""
        return value or None

    def model_for(self, role: Role) -> ModelSpec:
        specs: dict[Role, ModelSpec] = {
            "planner": self.planner,
            "worker": self.worker,
            "router": self.router,
            "judge": self.judge,
        }
        return specs[role]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once. Tests build `Settings(...)` directly instead."""
    return Settings()
