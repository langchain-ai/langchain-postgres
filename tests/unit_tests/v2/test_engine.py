import os
import uuid
from typing import AsyncIterator, Sequence

import asyncpg  # type: ignore
import pytest
import pytest_asyncio
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import VARCHAR, text
from sqlalchemy.engine.row import RowMapping
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from langchain_postgres import Column, PGEngine
from tests.utils import VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING

DEFAULT_TABLE = "default" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "custom" + str(uuid.uuid4()).replace("-", "_")
INT_ID_CUSTOM_TABLE = "custom_int_id" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_SYNC = "default_sync" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE_SYNC = "custom_sync" + str(uuid.uuid4()).replace("-", "_")
INT_ID_CUSTOM_TABLE_SYNC = "custom_int_id_sync" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def aexecute(
    engine: PGEngine,
    query: str,
) -> None:
    async def run(engine: PGEngine, query: str) -> None:
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await engine._run_as_async(run(engine, query))


async def afetch(engine: PGEngine, query: str) -> Sequence[RowMapping]:
    async def run(engine: PGEngine, query: str) -> Sequence[RowMapping]:
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()
        return result_fetch

    return await engine._run_as_async(run(engine, query))


@pytest.mark.enable_socket
@pytest.mark.asyncio
class TestEngineAsync:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self) -> AsyncIterator[PGEngine]:
        kwargs = {
            "pool_size": 3,
            "max_overflow": 2,
        }
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING, **kwargs)

        yield engine
        await aexecute(engine, f'DROP TABLE "{CUSTOM_TABLE}"')
        await aexecute(engine, f'DROP TABLE "{DEFAULT_TABLE}"')
        await aexecute(engine, f'DROP TABLE "{INT_ID_CUSTOM_TABLE}"')
        await engine.close()

    async def test_init_table(self, engine: PGEngine) -> None:
        await engine.ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        id = str(uuid.uuid4())
        content = "coffee"
        embedding = await embeddings_service.aembed_query(content)
        # Note: DeterministicFakeEmbedding generates a numpy array, converting to list a list of float values
        embedding_string = [float(dimension) for dimension in embedding]
        stmt = f"INSERT INTO {DEFAULT_TABLE} (langchain_id, content, embedding) VALUES ('{id}', '{content}','{embedding_string}');"
        await aexecute(engine, stmt)

    async def test_engine_args(self, engine: PGEngine) -> None:
        assert "Pool size: 3" in engine._pool.pool.status()

    async def test_init_table_custom(self, engine: PGEngine) -> None:
        await engine.ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TABLE}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

    async def test_init_table_with_int_id(self, engine: PGEngine) -> None:
        await engine.ainit_vectorstore_table(
            INT_ID_CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column=Column(name="integer_id", data_type="INTEGER", nullable=False),
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{INT_ID_CUSTOM_TABLE}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "integer_id", "data_type": "integer"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

    async def test_from_engine(self) -> None:
        engine = create_async_engine(
            CONNECTION_STRING,
        )

        pg_engine = PGEngine.from_engine(engine=engine)
        await aexecute(pg_engine, "SELECT 1")
        await pg_engine.close()

    async def test_from_connection_string(self) -> None:
        engine = PGEngine.from_connection_string(
            CONNECTION_STRING,
            echo=True,
            poolclass=NullPool,
        )
        await aexecute(engine, "SELECT 1")
        await engine.close()

    async def test_from_connection_string_url_error(
        self,
    ) -> None:
        with pytest.raises(ValueError):
            PGEngine.from_connection_string(
                f"postgresql+pg8000://user:password@host:port/db_name",
            )

    async def test_column(self, engine: PGEngine) -> None:
        with pytest.raises(ValueError):
            Column("test", VARCHAR)  # type: ignore
        with pytest.raises(ValueError):
            Column(1, "INTEGER")  # type: ignore


@pytest.mark.enable_socket
@pytest.mark.asyncio
class TestEngineSync:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield engine
        await aexecute(engine, f'DROP TABLE "{CUSTOM_TABLE_SYNC}"')
        await aexecute(engine, f'DROP TABLE "{DEFAULT_TABLE_SYNC}"')
        await aexecute(engine, f'DROP TABLE "{INT_ID_CUSTOM_TABLE_SYNC}"')
        await engine.close()

    async def test_init_table(self, engine: PGEngine) -> None:
        engine.init_vectorstore_table(DEFAULT_TABLE_SYNC, VECTOR_SIZE)
        id = str(uuid.uuid4())
        content = "coffee"
        embedding = await embeddings_service.aembed_query(content)
        # Note: DeterministicFakeEmbedding generates a numpy array, converting to list a list of float values
        embedding_string = [float(dimension) for dimension in embedding]
        stmt = f"INSERT INTO {DEFAULT_TABLE_SYNC} (langchain_id, content, embedding) VALUES ('{id}', '{content}','{embedding_string}');"
        await aexecute(engine, stmt)

    async def test_init_table_custom(self, engine: PGEngine) -> None:
        engine.init_vectorstore_table(
            CUSTOM_TABLE_SYNC,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TABLE_SYNC}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

    async def test_init_table_with_int_id(self, engine: PGEngine) -> None:
        engine.init_vectorstore_table(
            INT_ID_CUSTOM_TABLE_SYNC,
            VECTOR_SIZE,
            id_column=Column(name="integer_id", data_type="INTEGER", nullable=False),
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{INT_ID_CUSTOM_TABLE_SYNC}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "integer_id", "data_type": "integer"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

    async def test_engine_constructor_key(
        self,
        engine: PGEngine,
    ) -> None:
        key = object()
        with pytest.raises(Exception):
            PGEngine(key, engine._pool, None, None)
