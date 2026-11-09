import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from goalsmith.config import Settings

TEST_DB = "goalsmith_test"


@pytest.fixture(scope="session")
def test_dsn() -> str:
    """A separate database, so tests never touch your development data
    (or fix its vector size with test embeddings)."""
    base = Settings(_env_file=None).postgres_dsn()
    assert base, "GOALSMITH_POSTGRES_URL is empty"
    with psycopg.connect(base, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_DB)))
    return make_conninfo(base, dbname=TEST_DB)
