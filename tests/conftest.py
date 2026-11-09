import os
import socket
from typing import Any

import pytest

pytest_plugins = ["tests.fixtures.tracing"]  # the `spans` fixture

_ENV_PREFIXES = ("GOALSMITH_",)
_ENV_KEYS = {"ANTHROPIC_API_KEY", "TAVILY_API_KEY"}


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the developer's real keys and overrides so tests behave the same everywhere."""
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES) or key in _ENV_KEYS:
            monkeypatch.delenv(key)


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests must not touch the network. Integration tests are exempt."""
    if request.node.get_closest_marker("integration"):
        return

    def guard(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access attempted in a unit test")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
