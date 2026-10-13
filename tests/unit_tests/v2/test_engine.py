import os
import uuid
from typing import AsyncIterator, Sequence

import pytest
import pytest_asyncio
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import VARCHAR, text
from sqlalchemy.engine.row import RowMapping
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from langchain_postgres import Column, PGEngine
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig
from tests.utils import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)
from tests.utils import (
    VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING,
)

DEFAULT_TABLE = "default" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "custom" + str(uuid.uuid4()).replace("-", "_")
HYBRID_SEARCH_TABLE = "hybrid" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TYPEDDICT_TABLE = "custom_td" + str(uuid.uuid4()).replace("-", "_")
INT_ID_CUSTOM_TABLE = "custom_int_id" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_SYNC = "default_sync" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE_SYNC = "custom_sync" + str(uuid.uuid4()).replace("-", "_")
HYBRID_SEARCH_TABLE_SYNC = "hybrid_sync" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TYPEDDICT_TABLE_SYNC = "custom_td_sync" + str(uuid.uuid4()).replace("-", "_")
INT_ID_CUSTOM_TABLE_SYNC = "custom_int_id_sync" + str(uuid.uuid4()).replace("-", "_")
MULTI_HOST_TABLE = "multi_host" + str(uuid.uuid4()).replace("-", "_")
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
        await aexecute(engine, f'DROP TABLE IF EXISTS "{CUSTOM_TABLE}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{HYBRID_SEARCH_TABLE}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{CUSTOM_TYPEDDICT_TABLE}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{INT_ID_CUSTOM_TABLE}"')
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

    async def test_init_table_hybrid_search(self, engine: PGEngine) -> None:
        await engine.ainit_vectorstore_table(
            HYBRID_SEARCH_TABLE,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
            hybrid_search_config=HybridSearchConfig(),
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{HYBRID_SEARCH_TABLE}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "my-content_tsv", "data_type": "tsvector"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

    async def test_invalid_typed_dict(self, engine: PGEngine) -> None:
        with pytest.raises(TypeError):
            await engine.ainit_vectorstore_table(
                CUSTOM_TYPEDDICT_TABLE,
                VECTOR_SIZE,
                id_column={"name": "uuid", "data_type": "UUID"},  # type: ignore
                content_column="my-content",
                embedding_column="my_embedding",
                metadata_columns=[
                    {"name": "page", "data_type": "TEXT", "nullable": True},
                    {"name": "source", "data_type": "TEXT", "nullable": True},
                ],
                store_metadata=True,
                overwrite_existing=True,
            )
        with pytest.raises(TypeError):
            await engine.ainit_vectorstore_table(
                CUSTOM_TYPEDDICT_TABLE,
                VECTOR_SIZE,
                id_column={"name": "uuid", "data_type": "UUID", "nullable": False},
                content_column="my-content",
                embedding_column="my_embedding",
                metadata_columns=[
                    {"name": "page", "nullable": True},  # type: ignore
                    {"data_type": "TEXT", "nullable": True},  # type: ignore
                ],
                store_metadata=True,
                overwrite_existing=True,
            )

    async def test_init_table_custom_with_typed_dict(self, engine: PGEngine) -> None:
        await engine.ainit_vectorstore_table(
            CUSTOM_TYPEDDICT_TABLE,
            VECTOR_SIZE,
            id_column={"name": "uuid", "data_type": "UUID", "nullable": False},
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[
                {"name": "page", "data_type": "TEXT", "nullable": True},
                {"name": "source", "data_type": "TEXT", "nullable": True},
            ],
            store_metadata=True,
            overwrite_existing=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TYPEDDICT_TABLE}';"
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
                "postgresql+pg8000://user:password@host:port/db_name",
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
        await aexecute(engine, f'DROP TABLE IF EXISTS "{CUSTOM_TABLE_SYNC}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{HYBRID_SEARCH_TABLE_SYNC}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE_SYNC}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{INT_ID_CUSTOM_TABLE_SYNC}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{CUSTOM_TYPEDDICT_TABLE_SYNC}"')
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

    async def test_init_table_hybrid_search(self, engine: PGEngine) -> None:
        engine.init_vectorstore_table(
            HYBRID_SEARCH_TABLE_SYNC,
            VECTOR_SIZE,
            id_column="uuid",
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=True,
            hybrid_search_config=HybridSearchConfig(),
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{HYBRID_SEARCH_TABLE_SYNC}';"
        results = await afetch(engine, stmt)
        expected = [
            {"column_name": "uuid", "data_type": "uuid"},
            {"column_name": "my_embedding", "data_type": "USER-DEFINED"},
            {"column_name": "langchain_metadata", "data_type": "json"},
            {"column_name": "my-content", "data_type": "text"},
            {"column_name": "my-content_tsv", "data_type": "tsvector"},
            {"column_name": "page", "data_type": "text"},
            {"column_name": "source", "data_type": "text"},
        ]
        for row in results:
            assert row in expected

    async def test_invalid_typed_dict(self, engine: PGEngine) -> None:
        with pytest.raises(TypeError):
            engine.init_vectorstore_table(
                CUSTOM_TYPEDDICT_TABLE_SYNC,
                VECTOR_SIZE,
                id_column={"name": "uuid", "data_type": "UUID"},  # type: ignore
                content_column="my-content",
                embedding_column="my_embedding",
                metadata_columns=[
                    {"name": "page", "data_type": "TEXT", "nullable": True},
                    {"name": "source", "data_type": "TEXT", "nullable": True},
                ],
                store_metadata=True,
                overwrite_existing=True,
            )
        with pytest.raises(TypeError):
            engine.init_vectorstore_table(
                CUSTOM_TYPEDDICT_TABLE_SYNC,
                VECTOR_SIZE,
                id_column={"name": "uuid", "data_type": "UUID", "nullable": False},
                content_column="my-content",
                embedding_column="my_embedding",
                metadata_columns=[
                    {"name": "page", "nullable": True},  # type: ignore
                    {"data_type": "TEXT", "nullable": True},  # type: ignore
                ],
                store_metadata=True,
                overwrite_existing=True,
            )

    async def test_init_table_custom_with_typed_dict(self, engine: PGEngine) -> None:
        engine.init_vectorstore_table(
            CUSTOM_TYPEDDICT_TABLE_SYNC,
            VECTOR_SIZE,
            id_column={"name": "uuid", "data_type": "UUID", "nullable": False},
            content_column="my-content",
            embedding_column="my_embedding",
            metadata_columns=[
                {"name": "page", "data_type": "TEXT", "nullable": True},
                {"name": "source", "data_type": "TEXT", "nullable": True},
            ],
            store_metadata=True,
        )
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{CUSTOM_TYPEDDICT_TABLE_SYNC}';"
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


class TestBuildMultiHostConnectionString:
    """Unit tests for multi-host connection string building."""

    def test_basic_two_hosts(self) -> None:
        url = PGEngine._build_multi_host_connection_string(
            hosts=["host1", "host2"],
            user="myuser",
            password="mypass",
            database="mydb",
            ports=[5432, 5433],
            target_session_attrs="primary",
        )
        assert "postgresql+psycopg://" in url
        assert "host=host1,host2" in url
        assert "port=5432,5433" in url
        assert "target_session_attrs=primary" in url
        assert "myuser" in url
        assert "/mydb" in url

    def test_single_port_broadcast(self) -> None:
        url = PGEngine._build_multi_host_connection_string(
            hosts=["h1", "h2", "h3"],
            user="u",
            password="p",
            database="db",
            ports=[5432],
        )
        assert "port=5432,5432,5432" in url

    def test_default_ports(self) -> None:
        url = PGEngine._build_multi_host_connection_string(
            hosts=["h1", "h2"],
            user="u",
            password="p",
            database="db",
        )
        assert "port=5432,5432" in url

    def test_default_target_session_attrs(self) -> None:
        url = PGEngine._build_multi_host_connection_string(
            hosts=["h1"],
            user="u",
            password="p",
            database="db",
        )
        assert "target_session_attrs=any" in url

    def test_special_characters_in_password(self) -> None:
        url = PGEngine._build_multi_host_connection_string(
            hosts=["h1"],
            user="user",
            password="p@ss:word/special",
            database="db",
        )
        assert "p%40ss" in url

    def test_empty_hosts_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one host"):
            PGEngine._build_multi_host_connection_string(
                hosts=[],
                user="u",
                password="p",
                database="db",
            )

    def test_empty_host_string_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            PGEngine._build_multi_host_connection_string(
                hosts=["h1", ""],
                user="u",
                password="p",
                database="db",
            )

    def test_host_with_comma_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot contain comma"):
            PGEngine._build_multi_host_connection_string(
                hosts=["host1,host2"],
                user="u",
                password="p",
                database="db",
            )

    def test_whitespace_in_hosts_stripped(self) -> None:
        url = PGEngine._build_multi_host_connection_string(
            hosts=["  host1  ", " host2"],
            user="u",
            password="p",
            database="db",
        )
        assert "host=host1,host2" in url

    def test_ports_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="must match length of hosts"):
            PGEngine._build_multi_host_connection_string(
                hosts=["h1", "h2", "h3"],
                user="u",
                password="p",
                database="db",
                ports=[5432, 5433],
            )

    def test_invalid_port_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            PGEngine._build_multi_host_connection_string(
                hosts=["h1"],
                user="u",
                password="p",
                database="db",
                ports=[0],
            )

    def test_port_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            PGEngine._build_multi_host_connection_string(
                hosts=["h1"],
                user="u",
                password="p",
                database="db",
                ports=[99999],
            )

    def test_all_target_session_attrs_values(self) -> None:
        for attr in [
            "any",
            "read-write",
            "read-only",
            "primary",
            "standby",
            "prefer-standby",
        ]:
            url = PGEngine._build_multi_host_connection_string(
                hosts=["h1"],
                user="u",
                password="p",
                database="db",
                target_session_attrs=attr,  # type: ignore[arg-type]
            )
            assert f"target_session_attrs={attr}" in url


@pytest.mark.enable_socket
@pytest.mark.asyncio
class TestEngineMultiHost:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_hosts(
            hosts=[POSTGRES_HOST],
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB,
            ports=[int(POSTGRES_PORT)],
            target_session_attrs="any",
        )
        yield engine
        await aexecute(engine, f'DROP TABLE IF EXISTS "{MULTI_HOST_TABLE}"')
        await engine.close()

    async def test_from_hosts_single_host(self, engine: PGEngine) -> None:
        await aexecute(engine, "SELECT 1")

    async def test_from_hosts_creates_table(self, engine: PGEngine) -> None:
        await engine.ainit_vectorstore_table(MULTI_HOST_TABLE, VECTOR_SIZE)
        stmt = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{MULTI_HOST_TABLE}';"
        results = await afetch(engine, stmt)
        assert len(results) > 0

    async def test_from_hosts_with_engine_kwargs(self) -> None:
        engine = PGEngine.from_hosts(
            hosts=[POSTGRES_HOST],
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB,
            ports=[int(POSTGRES_PORT)],
            target_session_attrs="any",
            pool_size=3,
            max_overflow=2,
        )
        assert "Pool size: 3" in engine._pool.pool.status()
        await engine.close()
