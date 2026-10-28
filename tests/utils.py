"""Get fixtures for the database connection."""

import os
from contextlib import asynccontextmanager, contextmanager

import psycopg
from typing_extensions import AsyncGenerator, Generator

# Default connection settings target a standard local PostgreSQL instance.
# Override via environment variables for CI or custom setups.
#
# To use the project's docker-compose (langchain-specific credentials):
#   POSTGRES_USER=langchain POSTGRES_PASSWORD=langchain POSTGRES_DB=langchain_test \
#     docker-compose up pgvector
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")

POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")

DSN = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}"
    f":{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Connection string used primarily by the vectorstores tests
# it's written to work with SQLAlchemy (takes a driver name)
# It is also running on a postgres instance that has the pgvector extension
VECTORSTORE_CONNECTION_STRING = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}"
    f":{POSTGRES_PORT}/{POSTGRES_DB}"
)


@asynccontextmanager
async def asyncpg_client() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    # Establish a connection to your test database
    conn = await psycopg.AsyncConnection.connect(conninfo=DSN)
    try:
        yield conn
    finally:
        # Cleanup: close the connection after the test is done
        await conn.close()


@contextmanager
def syncpg_client() -> Generator[psycopg.Connection, None, None]:
    # Establish a connection to your test database
    conn = psycopg.connect(conninfo=DSN)
    try:
        yield conn
    finally:
        # Cleanup: close the connection after the test is done
        conn.close()
