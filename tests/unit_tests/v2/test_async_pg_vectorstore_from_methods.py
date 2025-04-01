import os
import uuid
from typing import AsyncIterator, Sequence

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from langchain_postgres import Column, PGEngine
from langchain_postgres.vectorstore.async_vectorstore import AsyncPGVectorStore
from tests.utils import VECTORSTORE_CONNECTION_STRING_ASYNCPG as CONNECTION_STRING

DEFAULT_TABLE = "default" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_SYNC = "default_sync" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "custom" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE_WITH_INT_ID = "custom_sync" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768


embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz"]
metadatas = [{"page": str(i), "source": "postgres"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]

embeddings = [embeddings_service.embed_query(texts[i]) for i in range(len(texts))]


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def aexecute(engine: PGEngine, query: str) -> None:
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()


async def afetch(engine: PGEngine, query: str) -> Sequence[RowMapping]:
    async with engine._pool.connect() as conn:
        result = await conn.execute(text(query))
        result_map = result.mappings()
        result_fetch = result_map.fetchall()
    return result_fetch


@pytest.mark.enable_socket
@pytest.mark.asyncio
class TestVectorStoreFromMethods:
    @pytest_asyncio.fixture
    async def engine(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        await engine._ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        await engine._ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=False,
        )
        await engine._ainit_vectorstore_table(
            CUSTOM_TABLE_WITH_INT_ID,
            VECTOR_SIZE,
            id_column=Column(name="integer_id", data_type="INTEGER", nullable=False),
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            store_metadata=False,
        )
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {CUSTOM_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {CUSTOM_TABLE_WITH_INT_ID}")
        await engine.close()

    async def test_afrom_texts(self, engine: PGEngine) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await AsyncPGVectorStore.afrom_texts(
            texts,
            embeddings_service,
            engine,
            DEFAULT_TABLE,
            metadatas=metadatas,
            ids=ids,
        )
        results = await afetch(engine, f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await aexecute(engine, f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_afrom_docs(self, engine: PGEngine) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await AsyncPGVectorStore.afrom_documents(
            docs,
            embeddings_service,
            engine,
            DEFAULT_TABLE,
            ids=ids,
        )
        results = await afetch(engine, f"SELECT * FROM {DEFAULT_TABLE}")
        assert len(results) == 3
        await aexecute(engine, f"TRUNCATE TABLE {DEFAULT_TABLE}")

    async def test_afrom_texts_custom(self, engine: PGEngine) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await AsyncPGVectorStore.afrom_texts(
            texts,
            embeddings_service,
            engine,
            CUSTOM_TABLE,
            ids=ids,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
        )
        results = await afetch(engine, f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] is None
        assert results[0]["source"] is None

    async def test_afrom_docs_custom(self, engine: PGEngine) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "postgres"},
            )
            for i in range(len(texts))
        ]
        await AsyncPGVectorStore.afrom_documents(
            docs,
            embeddings_service,
            engine,
            CUSTOM_TABLE,
            ids=ids,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
        )

        results = await afetch(engine, f"SELECT * FROM {CUSTOM_TABLE}")
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] == "0"
        assert results[0]["source"] == "postgres"
        await aexecute(engine, f"TRUNCATE TABLE {CUSTOM_TABLE}")

    async def test_afrom_docs_custom_with_int_id(self, engine: PGEngine) -> None:
        ids = [i for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "postgres"},
            )
            for i in range(len(texts))
        ]
        await AsyncPGVectorStore.afrom_documents(
            docs,
            embeddings_service,
            engine,
            CUSTOM_TABLE_WITH_INT_ID,
            ids=ids,
            id_column="integer_id",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
        )

        results = await afetch(engine, f"SELECT * FROM {CUSTOM_TABLE_WITH_INT_ID}")
        assert len(results) == 3
        for row in results:
            assert isinstance(row["integer_id"], int)
        await aexecute(engine, f"TRUNCATE TABLE {CUSTOM_TABLE_WITH_INT_ID}")
