"""Get fixtures for the database connection."""
import os
from contextlib import asynccontextmanager, contextmanager

import psycopg
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession as SqlAlchemyAsyncSession,
)
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session as SqlAlchemySession
from sqlalchemy.orm import sessionmaker
from typing_extensions import AsyncGenerator, Generator

# Env variables match the default settings in the docker-compose file
# located in the root of the repository: [root]/docker-compose.yml
# Non-standard ports are used to avoid conflicts with other local postgres
# instances.
# To spint up the postgres service for testing, run:
# cd [root]/docker-compose.yml
# docker-compose up pgvector
POSTGRES_USER = os.environ.get("POSTGRES_USER", "langchain")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "langchain")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "langchain")

POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "6024")

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


@asynccontextmanager
async def async_session() -> AsyncGenerator[SqlAlchemyAsyncSession, None]:
    engine = create_async_engine(VECTORSTORE_CONNECTION_STRING)
    AsyncSession = async_sessionmaker(bind=engine)

    session = AsyncSession()
    try:
        yield session
    finally:
        await session.close()


@contextmanager
def sync_session() -> Generator[SqlAlchemySession, None, None]:
    engine = create_engine(VECTORSTORE_CONNECTION_STRING)
    Session = sessionmaker(bind=engine)

    session = Session()
    try:
        yield session
    finally:
        session.close()
