"""Get fixtures for the database connection."""
import os
from contextlib import asynccontextmanager, contextmanager

import psycopg
from typing_extensions import AsyncGenerator, Generator

POSTGRES_USER = os.environ.get("POSTGRES_USER", "langchain")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "langchain")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "langchain")


# Using a different port for testing than the default 5432
# to avoid conflicts with a running PostgreSQL instance
# This port matches the convention in langchain/docker/docker-compose.yml
# To spin up a PostgreSQL instance for testing, run:
# docker-compose -f docker/docker-compose.yml up -d postgres
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "6023")

DSN = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}"
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
