import os
import uuid
from typing import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from PIL import Image
from sqlalchemy import text

from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import DistanceStrategy, HNSWQueryOptions
from tests.unit_tests.fixtures.metadata_filtering_data import (
    FILTERING_TEST_CASES,
    METADATAS,
    NEGATIVE_TEST_CASES,
)
from tests.utils import VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING

DEFAULT_TABLE = "default" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_SYNC = "default_sync" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "custom" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_FILTER_TABLE = "custom_filter" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_FILTER_TABLE_SYNC = "custom_filter_sync" + str(uuid.uuid4()).replace("-", "_")
IMAGE_TABLE = "image_table" + str(uuid.uuid4()).replace("-", "_")
IMAGE_TABLE_SYNC = "image_table_sync" + str(uuid.uuid4()).replace("-", "_")
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

# Note: The following texts are chosen to produce diverse
# similarity scores when using the DeterministicFakeEmbedding service. This ensures
# that the test cases can effectively validate the filtering and scoring logic.
# The scoring might be different if using a different embedding service.
texts = ["foo", "bar", "baz", "boo"]
ids = [str(uuid.uuid4()) for i in range(len(texts))]
metadatas = [{"page": str(i), "source": "postgres"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]
filter_docs = [
    Document(page_content=texts[i], metadata=METADATAS[i]) for i in range(len(texts))
]

embeddings = [embeddings_service.embed_query("foo") for i in range(len(texts))]


class FakeImageEmbedding(DeterministicFakeEmbedding):
    def embed_image(self, image_paths: list[str]) -> list[list[float]]:
        return [self.embed_query(path) for path in image_paths]


image_embedding_service = FakeImageEmbedding(size=VECTOR_SIZE)


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
class TestVectorStoreSearch:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {CUSTOM_FILTER_TABLE}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {IMAGE_TABLE}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine: PGEngine) -> AsyncIterator[PGVectorStore]:
        await engine.ainit_vectorstore_table(
            DEFAULT_TABLE, VECTOR_SIZE, store_metadata=False
        )
        vs = await PGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )

        await vs.aadd_documents(docs, ids=ids)
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def engine_sync(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {CUSTOM_TABLE}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom(self, engine_sync: PGEngine) -> AsyncIterator[PGVectorStore]:
        engine_sync.init_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[
                Column("page", "TEXT"),
                Column("source", "TEXT"),
            ],
            store_metadata=False,
        )

        vs_custom = PGVectorStore.create_sync(
            engine_sync,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            index_query_options=HNSWQueryOptions(ef_search=1),
        )
        vs_custom.add_documents(docs, ids=ids)
        yield vs_custom

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom_filter(self, engine: PGEngine) -> AsyncIterator[PGVectorStore]:
        await engine.ainit_vectorstore_table(
            CUSTOM_FILTER_TABLE,
            VECTOR_SIZE,
            metadata_columns=[
                Column("name", "TEXT"),
                Column("code", "TEXT"),
                Column("price", "FLOAT"),
                Column("is_available", "BOOLEAN"),
                Column("tags", "TEXT[]"),
                Column("inventory_location", "INTEGER[]"),
                Column("available_quantity", "INTEGER", nullable=True),
            ],
            id_column="langchain_id",
            store_metadata=False,
            overwrite_existing=True,
        )

        vs_custom_filter = await PGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=CUSTOM_FILTER_TABLE,
            metadata_columns=[
                "name",
                "code",
                "price",
                "is_available",
                "tags",
                "inventory_location",
                "available_quantity",
            ],
            id_column="langchain_id",
        )
        await vs_custom_filter.aadd_documents(filter_docs, ids=ids)
        yield vs_custom_filter

    @pytest_asyncio.fixture(scope="class")
    async def image_uris(self) -> AsyncIterator[list[str]]:
        red_uri = str(uuid.uuid4()).replace("-", "_") + "test_image_red.jpg"
        green_uri = str(uuid.uuid4()).replace("-", "_") + "test_image_green.jpg"
        blue_uri = str(uuid.uuid4()).replace("-", "_") + "test_image_blue.jpg"
        image = Image.new("RGB", (100, 100), color="red")
        image.save(red_uri)
        image = Image.new("RGB", (100, 100), color="green")
        image.save(green_uri)
        image = Image.new("RGB", (100, 100), color="blue")
        image.save(blue_uri)
        image_uris = [red_uri, green_uri, blue_uri]
        yield image_uris
        for uri in image_uris:
            try:
                os.remove(uri)
            except FileNotFoundError:
                pass

    @pytest_asyncio.fixture(scope="class")
    async def image_vs(
        self, engine: PGEngine, image_uris: list[str]
    ) -> AsyncIterator[PGVectorStore]:
        await engine.ainit_vectorstore_table(IMAGE_TABLE, VECTOR_SIZE)
        vs = await PGVectorStore.create(
            engine,
            embedding_service=image_embedding_service,
            table_name=IMAGE_TABLE,
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        )
        ids = [str(uuid.uuid4()) for i in range(len(image_uris))]
        await vs.aadd_images(image_uris, ids=ids)
        yield vs

    async def test_asimilarity_search_score(self, vs: PGVectorStore) -> None:
        results = await vs.asimilarity_search_with_score("foo")
        assert len(results) == 4
        assert results[0][0] == Document(page_content="foo", id=ids[0])
        assert results[0][1] == 0

    async def test_asimilarity_search_by_vector(self, vs: PGVectorStore) -> None:
        embedding = embeddings_service.embed_query("foo")
        results = await vs.asimilarity_search_by_vector(embedding)
        assert len(results) == 4
        assert results[0] == Document(page_content="foo", id=ids[0])
        result = await vs.asimilarity_search_with_score_by_vector(embedding=embedding)
        assert result[0][0] == Document(page_content="foo", id=ids[0])
        assert result[0][1] == 0

    async def test_asimilarity_search_image(
        self, image_vs: PGVectorStore, image_uris: list[str]
    ) -> None:
        results = await image_vs.asimilarity_search_image(image_uris[0], k=1)
        assert len(results) == 1
        assert results[0].metadata["image_uri"] == image_uris[0]
        results = await image_vs.asimilarity_search_image(image_uris[3], k=1)
        assert len(results) == 1
        assert results[0].metadata["image_uri"] == image_uris[3]

    async def test_similarity_search_with_relevance_scores_threshold_cosine(
        self, vs: PGVectorStore
    ) -> None:
        score_threshold = {"score_threshold": 0}
        results = await vs.asimilarity_search_with_relevance_scores(
            "foo", **score_threshold
        )
        # Note: Since tests use FakeEmbeddings which are non-normalized vectors, results might have scores beyond the range [0,1].
        # For a normalized embedding service, a threshold of zero will yield all matched documents.
        assert len(results) == 2

        score_threshold = {"score_threshold": 0.02}  # type: ignore
        results = await vs.asimilarity_search_with_relevance_scores(
            "foo", **score_threshold
        )
        assert len(results) == 2

        score_threshold = {"score_threshold": 0.9}  # type: ignore
        results = await vs.asimilarity_search_with_relevance_scores(
            "foo", **score_threshold
        )
        assert len(results) == 1
        assert results[0][0] == Document(page_content="foo", id=ids[0])

    async def test_similarity_search_with_relevance_scores_threshold_euclidean(
        self, engine: PGEngine
    ) -> None:
        vs = await PGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
            distance_strategy=DistanceStrategy.EUCLIDEAN,
        )

        score_threshold = {"score_threshold": 0.9}
        results = await vs.asimilarity_search_with_relevance_scores(
            "foo",
            **score_threshold,  # type: ignore
        )
        assert len(results) == 1
        assert results[0][0] == Document(page_content="foo", id=ids[0])

    async def test_amax_marginal_relevance_search_vector(
        self, vs: PGVectorStore
    ) -> None:
        embedding = embeddings_service.embed_query("bar")
        results = await vs.amax_marginal_relevance_search_by_vector(embedding)
        assert results[0] == Document(page_content="bar", id=ids[1])

    async def test_amax_marginal_relevance_search_vector_score(
        self, vs: PGVectorStore
    ) -> None:
        embedding = embeddings_service.embed_query("bar")
        results = await vs.amax_marginal_relevance_search_with_score_by_vector(
            embedding
        )
        assert results[0][0] == Document(page_content="bar", id=ids[1])

        results = await vs.amax_marginal_relevance_search_with_score_by_vector(
            embedding, lambda_mult=0.75, fetch_k=10
        )
        assert results[0][0] == Document(page_content="bar", id=ids[1])

    async def test_aget_by_ids(self, vs: PGVectorStore) -> None:
        test_ids = [ids[0]]
        results = await vs.aget_by_ids(ids=test_ids)

        assert results[0] == Document(page_content="foo", id=ids[0])

    async def test_aget_by_ids_custom_vs(self, vs_custom: PGVectorStore) -> None:
        test_ids = [ids[0]]
        results = await vs_custom.aget_by_ids(ids=test_ids)

        assert results[0] == Document(page_content="foo", id=ids[0])

    @pytest.mark.parametrize("test_filter, expected_ids", FILTERING_TEST_CASES)
    async def test_vectorstore_with_metadata_filters(
        self,
        vs_custom_filter: PGVectorStore,
        test_filter: dict,
        expected_ids: list[str],
    ) -> None:
        """Test end to end construction and search."""
        docs = await vs_custom_filter.asimilarity_search(
            "meow", k=5, filter=test_filter
        )
        assert [doc.metadata["code"] for doc in docs] == expected_ids, test_filter


@pytest.mark.enable_socket
class TestVectorStoreSearchSync:
    @pytest_asyncio.fixture(scope="class")
    async def engine_sync(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield engine
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE_SYNC}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {CUSTOM_FILTER_TABLE_SYNC}")
        await aexecute(engine, f"DROP TABLE IF EXISTS {IMAGE_TABLE_SYNC}")
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom(self, engine_sync: PGEngine) -> AsyncIterator[PGVectorStore]:
        engine_sync.init_vectorstore_table(
            DEFAULT_TABLE_SYNC,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[
                Column("page", "TEXT"),
                Column("source", "TEXT"),
            ],
            store_metadata=False,
        )

        vs_custom = await PGVectorStore.create(
            engine_sync,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE_SYNC,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            index_query_options=HNSWQueryOptions(ef_search=1),
        )
        vs_custom.add_documents(docs, ids=ids)
        yield vs_custom

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom_filter_sync(
        self, engine_sync: PGEngine
    ) -> AsyncIterator[PGVectorStore]:
        engine_sync.init_vectorstore_table(
            CUSTOM_FILTER_TABLE_SYNC,
            VECTOR_SIZE,
            metadata_columns=[
                Column("name", "TEXT"),
                Column("code", "TEXT"),
                Column("price", "FLOAT"),
                Column("is_available", "BOOLEAN"),
                Column("tags", "TEXT[]"),
                Column("inventory_location", "INTEGER[]"),
                Column("available_quantity", "INTEGER", nullable=True),
            ],
            id_column="langchain_id",
            store_metadata=False,
            overwrite_existing=True,
        )

        vs_custom_filter_sync = await PGVectorStore.create(
            engine_sync,
            embedding_service=embeddings_service,
            table_name=CUSTOM_FILTER_TABLE_SYNC,
            metadata_columns=[
                "name",
                "code",
                "price",
                "is_available",
                "tags",
                "inventory_location",
                "available_quantity",
            ],
            id_column="langchain_id",
        )

        vs_custom_filter_sync.add_documents(filter_docs, ids=ids)
        yield vs_custom_filter_sync

    @pytest_asyncio.fixture(scope="class")
    async def image_uris(self) -> AsyncIterator[list[str]]:
        red_uri = str(uuid.uuid4()).replace("-", "_") + "test_image_red.jpg"
        green_uri = str(uuid.uuid4()).replace("-", "_") + "test_image_green.jpg"
        blue_uri = str(uuid.uuid4()).replace("-", "_") + "test_image_blue.jpg"
        image = Image.new("RGB", (100, 100), color="red")
        image.save(red_uri)
        image = Image.new("RGB", (100, 100), color="green")
        image.save(green_uri)
        image = Image.new("RGB", (100, 100), color="blue")
        image.save(blue_uri)
        image_uris = [red_uri, green_uri, blue_uri]
        yield image_uris
        for uri in image_uris:
            os.remove(uri)

    @pytest_asyncio.fixture(scope="class")
    def image_vs(
        self, engine_sync: PGEngine, image_uris: list[str]
    ) -> Iterator[PGVectorStore]:
        engine_sync.init_vectorstore_table(IMAGE_TABLE_SYNC, VECTOR_SIZE)
        vs = PGVectorStore.create_sync(
            engine_sync,
            embedding_service=image_embedding_service,
            table_name=IMAGE_TABLE_SYNC,
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        )
        ids = [str(uuid.uuid4()) for i in range(len(image_uris))]
        vs.add_images(image_uris, ids=ids)
        yield vs

    def test_similarity_search_score(self, vs_custom: PGVectorStore) -> None:
        results = vs_custom.similarity_search_with_score("foo")
        assert len(results) == 4
        assert results[0][0] == Document(page_content="foo", id=ids[0])
        assert results[0][1] == 0

    def test_similarity_search_by_vector(self, vs_custom: PGVectorStore) -> None:
        embedding = embeddings_service.embed_query("foo")
        results = vs_custom.similarity_search_by_vector(embedding)
        assert len(results) == 4
        assert results[0] == Document(page_content="foo", id=ids[0])
        result = vs_custom.similarity_search_with_score_by_vector(embedding=embedding)
        assert result[0][0] == Document(page_content="foo", id=ids[0])
        assert result[0][1] == 0

    def test_similarity_search_image(
        self, image_vs: PGVectorStore, image_uris: list[str]
    ) -> None:
        results = image_vs.similarity_search_image(image_uris[0], k=1)
        assert len(results) == 1
        assert results[0].metadata["image_uri"] == image_uris[0]

    def test_max_marginal_relevance_search_vector(
        self, vs_custom: PGVectorStore
    ) -> None:
        embedding = embeddings_service.embed_query("bar")
        results = vs_custom.max_marginal_relevance_search_by_vector(embedding)
        assert results[0] == Document(page_content="bar", id=ids[1])

    def test_max_marginal_relevance_search_vector_score(
        self, vs_custom: PGVectorStore
    ) -> None:
        embedding = embeddings_service.embed_query("bar")
        results = vs_custom.max_marginal_relevance_search_with_score_by_vector(
            embedding
        )
        assert results[0][0] == Document(page_content="bar", id=ids[1])

        results = vs_custom.max_marginal_relevance_search_with_score_by_vector(
            embedding, lambda_mult=0.75, fetch_k=10
        )
        assert results[0][0] == Document(page_content="bar", id=ids[1])

    def test_get_by_ids_custom_vs(self, vs_custom: PGVectorStore) -> None:
        test_ids = [ids[0]]
        results = vs_custom.get_by_ids(ids=test_ids)

        assert results[0] == Document(page_content="foo", id=ids[0])

    @pytest.mark.parametrize("test_filter, expected_ids", FILTERING_TEST_CASES)
    def test_sync_vectorstore_with_metadata_filters(
        self,
        vs_custom_filter_sync: PGVectorStore,
        test_filter: dict,
        expected_ids: list[str],
    ) -> None:
        """Test end to end construction and search."""

        docs = vs_custom_filter_sync.similarity_search("meow", k=5, filter=test_filter)
        assert [doc.metadata["code"] for doc in docs] == expected_ids, test_filter

    @pytest.mark.parametrize("test_filter", NEGATIVE_TEST_CASES)
    def test_metadata_filter_negative_tests(
        self, vs_custom_filter_sync: PGVectorStore, test_filter: dict
    ) -> None:
        with pytest.raises((ValueError, NotImplementedError)):
            docs = vs_custom_filter_sync.similarity_search(
                "meow", k=5, filter=test_filter
            )
