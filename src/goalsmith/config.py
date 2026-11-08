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
