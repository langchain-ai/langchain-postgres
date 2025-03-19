import os
import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.indexes import (
    DEFAULT_INDEX_NAME_SUFFIX,
    DistanceStrategy,
    HNSWIndex,
    IVFFlatIndex,
)
from tests.utils import VECTORSTORE_CONNECTION_STRING_ASYNCPG as CONNECTION_STRING

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_ASYNC = "test_table" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_OMNI = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test_table_custom" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_INDEX_NAME = DEFAULT_TABLE + DEFAULT_INDEX_NAME_SUFFIX
DEFAULT_INDEX_NAME_ASYNC = DEFAULT_TABLE_ASYNC + DEFAULT_INDEX_NAME_SUFFIX
DEFAULT_INDEX_NAME_OMNI = DEFAULT_TABLE_OMNI + DEFAULT_INDEX_NAME_SUFFIX
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz"]
ids = [str(uuid.uuid4()) for i in range(len(texts))]
metadatas = [{"page": str(i), "source": "postgres"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]

embeddings = [embeddings_service.embed_query("foo") for i in range(len(texts))]


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


@pytest.mark.enable_socket
@pytest.mark.asyncio(scope="class")
class TestIndex:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine: PGEngine) -> AsyncIterator[PGVectorStore]:
        engine.init_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        vs = PGVectorStore.create_sync(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )

        vs.add_texts(texts, ids=ids)
        vs.drop_vector_index()
        yield vs

    async def test_aapply_vector_index(self, vs: PGVectorStore) -> None:
        index = HNSWIndex()
        vs.apply_vector_index(index)
        assert vs.is_valid_index(DEFAULT_INDEX_NAME)
        vs.drop_vector_index()

    async def test_areindex(self, vs: PGVectorStore) -> None:
        if not vs.is_valid_index(DEFAULT_INDEX_NAME):
            index = HNSWIndex()
            vs.apply_vector_index(index)
        vs.reindex()
        vs.reindex(DEFAULT_INDEX_NAME)
        assert vs.is_valid_index(DEFAULT_INDEX_NAME)
        vs.drop_vector_index(DEFAULT_INDEX_NAME)

    async def test_dropindex(self, vs: PGVectorStore) -> None:
        vs.drop_vector_index()
        result = vs.is_valid_index(DEFAULT_INDEX_NAME)
        assert not result

    async def test_aapply_vector_index_ivfflat(self, vs: PGVectorStore) -> None:
        index = IVFFlatIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
        vs.apply_vector_index(index, concurrently=True)
        assert vs.is_valid_index(DEFAULT_INDEX_NAME)
        index = IVFFlatIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        vs.apply_vector_index(index)
        assert vs.is_valid_index("secondindex")
        vs.drop_vector_index("secondindex")
        vs.drop_vector_index()

    async def test_is_valid_index(self, vs: PGVectorStore) -> None:
        is_valid = vs.is_valid_index("invalid_index")
        assert is_valid == False


@pytest.mark.enable_socket
@pytest.mark.asyncio(scope="class")
class TestAsyncIndex:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE_ASYNC}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine: PGEngine) -> AsyncIterator[PGVectorStore]:
        await engine.ainit_vectorstore_table(DEFAULT_TABLE_ASYNC, VECTOR_SIZE)
        vs = await PGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE_ASYNC,
        )

        await vs.aadd_texts(texts, ids=ids)
        await vs.adrop_vector_index()
        yield vs

    async def test_aapply_vector_index(self, vs: PGVectorStore) -> None:
        index = HNSWIndex()
        await vs.aapply_vector_index(index)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)

    async def test_areindex(self, vs: PGVectorStore) -> None:
        if not await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC):
            index = HNSWIndex()
            await vs.aapply_vector_index(index)
        await vs.areindex()
        await vs.areindex(DEFAULT_INDEX_NAME_ASYNC)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME_ASYNC)

    async def test_dropindex(self, vs: PGVectorStore) -> None:
        await vs.adrop_vector_index()
        result = await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        assert not result

    async def test_aapply_vector_index_ivfflat(self, vs: PGVectorStore) -> None:
        index = IVFFlatIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
        await vs.aapply_vector_index(index, concurrently=True)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        index = IVFFlatIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        await vs.aapply_vector_index(index)
        assert await vs.ais_valid_index("secondindex")
        await vs.adrop_vector_index("secondindex")
        await vs.adrop_vector_index()

    async def test_is_valid_index(self, vs: PGVectorStore) -> None:
        is_valid = await vs.ais_valid_index("invalid_index")
        assert is_valid == False
