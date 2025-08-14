import os
import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig
from langchain_postgres.v2.indexes import (
    DistanceStrategy,
    HNSWIndex,
    IVFFlatIndex,
)
from tests.utils import VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING

uuid_str = str(uuid.uuid4()).replace("-", "_")
uuid_str_async = str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE = "default" + uuid_str
DEFAULT_HYBRID_TABLE = "hybrid" + uuid_str
DEFAULT_TABLE_ASYNC = "default_async" + uuid_str_async
DEFAULT_HYBRID_TABLE_ASYNC = "hybrid_async" + uuid_str_async
CUSTOM_TABLE = "custom" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_INDEX_NAME = "index" + uuid_str
DEFAULT_INDEX_NAME_ASYNC = "index" + uuid_str_async
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
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_HYBRID_TABLE}")
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
        vs.drop_vector_index(DEFAULT_INDEX_NAME)
        yield vs

    async def test_aapply_vector_index(self, vs: PGVectorStore) -> None:
        index = HNSWIndex(name=DEFAULT_INDEX_NAME)
        vs.apply_vector_index(index)
        assert vs.is_valid_index(DEFAULT_INDEX_NAME)
        vs.drop_vector_index(DEFAULT_INDEX_NAME)

    async def test_areindex(self, vs: PGVectorStore) -> None:
        if not vs.is_valid_index(DEFAULT_INDEX_NAME):
            index = HNSWIndex(name=DEFAULT_INDEX_NAME)
            vs.apply_vector_index(index)
        vs.reindex(DEFAULT_INDEX_NAME)
        vs.reindex(DEFAULT_INDEX_NAME)
        assert vs.is_valid_index(DEFAULT_INDEX_NAME)
        vs.drop_vector_index(DEFAULT_INDEX_NAME)

    async def test_dropindex(self, vs: PGVectorStore) -> None:
        vs.drop_vector_index(DEFAULT_INDEX_NAME)
        result = vs.is_valid_index(DEFAULT_INDEX_NAME)
        assert not result

    async def test_aapply_vector_index_ivfflat(self, vs: PGVectorStore) -> None:
        index = IVFFlatIndex(
            name=DEFAULT_INDEX_NAME, distance_strategy=DistanceStrategy.EUCLIDEAN
        )
        vs.apply_vector_index(index, concurrently=True)
        assert vs.is_valid_index(DEFAULT_INDEX_NAME)
        index = IVFFlatIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        vs.apply_vector_index(index)
        assert vs.is_valid_index("secondindex")
        vs.drop_vector_index("secondindex")
        vs.drop_vector_index(DEFAULT_INDEX_NAME)

    async def test_is_valid_index(self, vs: PGVectorStore) -> None:
        is_valid = vs.is_valid_index("invalid_index")
        assert not is_valid

    async def test_apply_hybrid_search_index_non_hybrid_search_vs(
        self, vs: PGVectorStore
    ) -> None:
        with pytest.raises(ValueError):
            vs.apply_hybrid_search_index()

    async def test_apply_hybrid_search_index_table_without_tsv_column(
        self, engine: PGEngine, vs: PGVectorStore
    ) -> None:
        tsv_index_name = "tsv_index_on_table_without_tsv_column_" + uuid_str
        vs_hybrid = PGVectorStore.create_sync(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
            hybrid_search_config=HybridSearchConfig(index_name=tsv_index_name),
        )
        is_valid_index = vs_hybrid.is_valid_index(tsv_index_name)
        assert is_valid_index == False
        vs_hybrid.apply_hybrid_search_index()
        assert vs_hybrid.is_valid_index(tsv_index_name)
        vs_hybrid.drop_vector_index(tsv_index_name)
        is_valid_index = vs_hybrid.is_valid_index(tsv_index_name)
        assert is_valid_index == False

    async def test_apply_hybrid_search_index_table_with_tsv_column(
        self, engine: PGEngine
    ) -> None:
        tsv_index_name = "tsv_index_on_table_with_tsv_column_" + uuid_str
        config = HybridSearchConfig(
            tsv_column="tsv_column",
            tsv_lang="pg_catalog.english",
            index_name=tsv_index_name,
        )
        engine.init_vectorstore_table(
            DEFAULT_HYBRID_TABLE,
            VECTOR_SIZE,
            hybrid_search_config=config,
        )
        vs_hybrid = PGVectorStore.create_sync(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_HYBRID_TABLE,
            hybrid_search_config=config,
        )
        is_valid_index = vs_hybrid.is_valid_index(tsv_index_name)
        assert is_valid_index == False
        vs_hybrid.apply_hybrid_search_index()
        assert vs_hybrid.is_valid_index(tsv_index_name)
        vs_hybrid.reindex(tsv_index_name)
        assert vs_hybrid.is_valid_index(tsv_index_name)
        vs_hybrid.drop_vector_index(tsv_index_name)
        is_valid_index = vs_hybrid.is_valid_index(tsv_index_name)
        assert is_valid_index == False


@pytest.mark.enable_socket
@pytest.mark.asyncio(scope="class")
class TestAsyncIndex:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE_ASYNC}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_HYBRID_TABLE_ASYNC}")
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
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME_ASYNC)
        yield vs

    async def test_aapply_vector_index(self, vs: PGVectorStore) -> None:
        index = HNSWIndex(name=DEFAULT_INDEX_NAME_ASYNC)
        await vs.aapply_vector_index(index)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)

    async def test_areindex(self, vs: PGVectorStore) -> None:
        if not await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC):
            index = HNSWIndex(name=DEFAULT_INDEX_NAME_ASYNC)
            await vs.aapply_vector_index(index)
        await vs.areindex(DEFAULT_INDEX_NAME_ASYNC)
        await vs.areindex(DEFAULT_INDEX_NAME_ASYNC)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME_ASYNC)

    async def test_dropindex(self, vs: PGVectorStore) -> None:
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME_ASYNC)
        result = await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        assert not result

    async def test_aapply_vector_index_ivfflat(self, vs: PGVectorStore) -> None:
        index = IVFFlatIndex(
            name=DEFAULT_INDEX_NAME_ASYNC, distance_strategy=DistanceStrategy.EUCLIDEAN
        )
        await vs.aapply_vector_index(index, concurrently=True)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        index = IVFFlatIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        await vs.aapply_vector_index(index)
        assert await vs.ais_valid_index("secondindex")
        await vs.adrop_vector_index("secondindex")
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME_ASYNC)

    async def test_is_valid_index(self, vs: PGVectorStore) -> None:
        is_valid = await vs.ais_valid_index("invalid_index")
        assert not is_valid

    async def test_aapply_hybrid_search_index_non_hybrid_search_vs(
        self, vs: PGVectorStore
    ) -> None:
        with pytest.raises(ValueError):
            await vs.aapply_hybrid_search_index()

    async def test_aapply_hybrid_search_index_table_without_tsv_column(
        self, engine: PGEngine, vs: PGVectorStore
    ) -> None:
        tsv_index_name = "tsv_index_on_table_without_tsv_column_" + uuid_str_async
        vs_hybrid = await PGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE_ASYNC,
            hybrid_search_config=HybridSearchConfig(index_name=tsv_index_name),
        )
        is_valid_index = await vs_hybrid.ais_valid_index(tsv_index_name)
        assert is_valid_index == False
        await vs_hybrid.aapply_hybrid_search_index()
        assert await vs_hybrid.ais_valid_index(tsv_index_name)
        await vs_hybrid.adrop_vector_index(tsv_index_name)
        is_valid_index = await vs_hybrid.ais_valid_index(tsv_index_name)
        assert is_valid_index == False

    async def test_aapply_hybrid_search_index_table_with_tsv_column(
        self, engine: PGEngine
    ) -> None:
        tsv_index_name = "tsv_index_on_table_with_tsv_column_" + uuid_str_async
        config = HybridSearchConfig(
            tsv_column="tsv_column",
            tsv_lang="pg_catalog.english",
            index_name=tsv_index_name,
        )
        await engine.ainit_vectorstore_table(
            DEFAULT_HYBRID_TABLE_ASYNC,
            VECTOR_SIZE,
            hybrid_search_config=config,
        )
        vs_hybrid = await PGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_HYBRID_TABLE_ASYNC,
            hybrid_search_config=config,
        )
        is_valid_index = await vs_hybrid.ais_valid_index(tsv_index_name)
        assert is_valid_index == False
        await vs_hybrid.aapply_hybrid_search_index()
        assert await vs_hybrid.ais_valid_index(tsv_index_name)
        await vs_hybrid.areindex(tsv_index_name)
        assert await vs_hybrid.ais_valid_index(tsv_index_name)
        await vs_hybrid.adrop_vector_index(tsv_index_name)
        is_valid_index = await vs_hybrid.ais_valid_index(tsv_index_name)
        assert is_valid_index == False
