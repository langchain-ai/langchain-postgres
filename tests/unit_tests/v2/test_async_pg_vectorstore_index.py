import os
import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from langchain_postgres import PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig
from langchain_postgres.v2.indexes import (DistanceStrategy, HNSWIndex,
                                           IVFFlatIndex)
from tests.utils import VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING

uuid_str = str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE = "default" + uuid_str
DEFAULT_HYBRID_TABLE = "hybrid" + uuid_str
DEFAULT_INDEX_NAME = "index" + uuid_str
VECTOR_SIZE = 768
SIMPLE_TABLE = "default_table"

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


async def aexecute(engine: PGEngine, query: str) -> None:
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()


@pytest.mark.enable_socket
@pytest.mark.asyncio(scope="class")
class TestIndex:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield engine

        engine._adrop_table(DEFAULT_TABLE)
        engine._adrop_table(DEFAULT_HYBRID_TABLE)
        engine._adrop_table(SIMPLE_TABLE)
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine: PGEngine) -> AsyncIterator[AsyncPGVectorStore]:
        await engine._ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        vs = await AsyncPGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )

        await vs.aadd_texts(texts, ids=ids)
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME)
        yield vs

    async def test_apply_default_name_vector_index(self, engine: PGEngine) -> None:
        await engine._ainit_vectorstore_table(SIMPLE_TABLE, VECTOR_SIZE)
        vs = await AsyncPGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=SIMPLE_TABLE,
        )
        await vs.aadd_texts(texts, ids=ids)
        await vs.adrop_vector_index()
        index = HNSWIndex()
        await vs.aapply_vector_index(index)
        assert await vs.is_valid_index()
        await vs.adrop_vector_index()

    async def test_aapply_vector_index(self, vs: AsyncPGVectorStore) -> None:
        index = HNSWIndex(name=DEFAULT_INDEX_NAME)
        await vs.aapply_vector_index(index)
        assert await vs.is_valid_index(DEFAULT_INDEX_NAME)
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME)

    async def test_aapply_vector_index_non_hybrid_search_vs_without_tsv_column(
        self, vs
    ) -> None:
        index = HNSWIndex(name="test_index_hybrid" + uuid_str)

        tsv_index_name = DEFAULT_TABLE + "langchain_tsv_index"
        is_valid_index = await vs.is_valid_index(tsv_index_name)
        assert is_valid_index == False
        await vs.aapply_vector_index(index)
        is_valid_index = await vs.is_valid_index(tsv_index_name)
        assert is_valid_index == False
        await vs.adrop_vector_index(tsv_index_name)
        is_valid_index = await vs.is_valid_index(tsv_index_name)
        assert is_valid_index == False

    async def test_aapply_vector_index_hybrid_search_vs_without_tsv_column(
        self, engine, vs
    ) -> None:
        # overwriting vs to get a hybrid vs
        vs = await AsyncPGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
            hybrid_search_config=HybridSearchConfig(),
        )
        index = HNSWIndex(name="test_index_hybrid" + uuid_str)

        tsv_index_name = DEFAULT_TABLE + "langchain_tsv_index"
        is_valid_index = await vs.is_valid_index(tsv_index_name)
        assert is_valid_index == False
        await vs.adrop_vector_index(index.name)
        await vs.aapply_vector_index(index)
        assert await vs.is_valid_index(tsv_index_name)
        await vs.areindex(tsv_index_name)
        assert await vs.is_valid_index(tsv_index_name)
        await vs.adrop_vector_index(tsv_index_name)
        is_valid_index = await vs.is_valid_index(tsv_index_name)
        assert is_valid_index == False
        await vs.adrop_vector_index(index.name)

    async def test_aapply_vector_index_hybrid_search_with_tsv_column(
        self, engine
    ) -> None:
        await engine._ainit_vectorstore_table(
            DEFAULT_HYBRID_TABLE, VECTOR_SIZE, hybrid_search_config=HybridSearchConfig()
        )
        vs = await AsyncPGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_HYBRID_TABLE,
            hybrid_search_config=HybridSearchConfig(),
        )
        tsv_index_name = DEFAULT_HYBRID_TABLE + "langchain_tsv_index"
        is_valid_index = await vs.is_valid_index(tsv_index_name)
        assert is_valid_index == False
        index = HNSWIndex(name=DEFAULT_INDEX_NAME)
        await vs.aapply_vector_index(index)
        await vs.adrop_vector_index(tsv_index_name)
        await vs.adrop_vector_index(index.name)
        is_valid_index = await vs.is_valid_index(tsv_index_name)
        assert is_valid_index == False

    async def test_areindex(self, vs: AsyncPGVectorStore) -> None:
        if not await vs.is_valid_index(DEFAULT_INDEX_NAME):
            index = HNSWIndex(name=DEFAULT_INDEX_NAME)
            await vs.aapply_vector_index(index)
        await vs.areindex(DEFAULT_INDEX_NAME)
        await vs.areindex(DEFAULT_INDEX_NAME)
        assert await vs.is_valid_index(DEFAULT_INDEX_NAME)
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME)

    async def test_dropindex(self, vs: AsyncPGVectorStore) -> None:
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME)
        result = await vs.is_valid_index(DEFAULT_INDEX_NAME)
        assert not result

    async def test_aapply_vector_index_ivfflat(self, vs: AsyncPGVectorStore) -> None:
        index = IVFFlatIndex(
            name=DEFAULT_INDEX_NAME, distance_strategy=DistanceStrategy.EUCLIDEAN
        )
        await vs.aapply_vector_index(index, concurrently=True)
        assert await vs.is_valid_index(DEFAULT_INDEX_NAME)
        index = IVFFlatIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        await vs.aapply_vector_index(index)
        assert await vs.is_valid_index("secondindex")
        await vs.adrop_vector_index("secondindex")
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME)

    async def test_is_valid_index(self, vs: AsyncPGVectorStore) -> None:
        is_valid = await vs.is_valid_index("invalid_index")
        assert not is_valid
