import os
from unittest.mock import MagicMock, patch
import uuid
from typing import AsyncIterator, Sequence

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from PIL import Image
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from langchain_postgres import Column, PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from tests.utils import VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING

DEFAULT_TABLE = "default" + str(uuid.uuid4())
DEFAULT_TABLE_SYNC = "default_sync" + str(uuid.uuid4())
CUSTOM_TABLE = "custom" + str(uuid.uuid4())
IMAGE_TABLE = "image_table" + str(uuid.uuid4())
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz"]
metadatas = [{"page": str(i), "source": "postgres"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]

embeddings = [embeddings_service.embed_query(texts[i]) for i in range(len(texts))]


class FakeImageEmbedding(DeterministicFakeEmbedding):
    def embed_image(self, image_paths: list[str]) -> list[list[float]]:
        return [self.embed_query(path) for path in image_paths]


image_embedding_service = FakeImageEmbedding(size=VECTOR_SIZE)


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
@pytest.mark.asyncio(scope="class")
class TestVectorStore:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

        yield engine
        await engine.adrop_table(DEFAULT_TABLE)
        await engine.adrop_table(CUSTOM_TABLE)
        await engine.adrop_table(IMAGE_TABLE)
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine: PGEngine) -> AsyncIterator[AsyncPGVectorStore]:
        await engine._ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        vs = await AsyncPGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom(self, engine: PGEngine) -> AsyncIterator[AsyncPGVectorStore]:
        await engine._ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            metadata_json_column="mymeta",
        )
        vs = await AsyncPGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
            metadata_json_column="mymeta",
        )
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def image_vs(self, engine: PGEngine) -> AsyncIterator[AsyncPGVectorStore]:
        await engine._ainit_vectorstore_table(
            IMAGE_TABLE,
            VECTOR_SIZE,
            metadata_columns=[
                Column("image_id", "TEXT"),
                Column("source", "TEXT"),
            ],
        )
        vs = await AsyncPGVectorStore.create(
            engine,
            embedding_service=image_embedding_service,
            table_name=IMAGE_TABLE,
            metadata_columns=["image_id", "source"],
            metadata_json_column="mymeta",
        )
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def image_uris(self) -> AsyncIterator[list[str]]:
        red_uri = str(uuid.uuid4()).replace("-", "_") + "test_image_red.jpg"
        green_uri = str(uuid.uuid4()).replace("-", "_") + "test_image_green.jpg"
        blue_uri = str(uuid.uuid4()).replace("-", "_") + "test_image_blue.jpg"
        gcs_uri = "gs://github-repo/img/vision/google-cloud-next.jpeg"
        image = Image.new("RGB", (100, 100), color="red")
        image.save(red_uri)
        image = Image.new("RGB", (100, 100), color="green")
        image.save(green_uri)
        image = Image.new("RGB", (100, 100), color="blue")
        image.save(blue_uri)
        image_uris = [red_uri, green_uri, blue_uri, gcs_uri]
        yield image_uris
        for uri in image_uris:
            try:
                os.remove(uri)
            except FileNotFoundError:
                pass

    async def test_init_with_constructor(self, engine: PGEngine) -> None:
        with pytest.raises(Exception):
            AsyncPGVectorStore(
                key={},
                engine=engine._pool,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="noname",
                embedding_column="myembedding",
                metadata_columns=["page", "source"],
                metadata_json_column="mymeta",
            )

    async def test_post_init(self, engine: PGEngine) -> None:
        with pytest.raises(ValueError):
            await AsyncPGVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="noname",
                embedding_column="myembedding",
                metadata_columns=["page", "source"],
                metadata_json_column="mymeta",
            )

    async def test_aadd_texts(self, engine: PGEngine, vs: AsyncPGVectorStore) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, metadatas, ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 6
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_texts_edge_cases(
        self, engine: PGEngine, vs: AsyncPGVectorStore
    ) -> None:
        texts = ["Taylor's", '"Swift"', "best-friend"]
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_docs(self, engine: PGEngine, vs: AsyncPGVectorStore) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_documents(docs, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_docs_no_ids(
        self, engine: PGEngine, vs: AsyncPGVectorStore
    ) -> None:
        await vs.aadd_documents(docs)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_adelete(self, engine: PGEngine, vs: AsyncPGVectorStore) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        # delete an ID
        await vs.adelete([ids[0]])
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 2
        # delete with no ids
        result = await vs.adelete()
        assert result == False
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    @patch('langchain_postgres.v2.async_vectorstore.storage.Client')
    async def test_aadd_images(
        self, MockStorageClient: MagicMock, engine: PGEngine, image_vs: AsyncPGVectorStore, image_uris: list[str]
    ) -> None:
        mock_blob_instance = MagicMock()
        fake_image_bytes = b"fake_gcs_image_data" # Differentiated fake data
        mock_blob_instance.download_as_bytes.return_value = fake_image_bytes

        mock_bucket_instance = MagicMock()
        mock_bucket_instance.blob.return_value = mock_blob_instance

        mock_storage_client_instance = MagicMock()
        mock_storage_client_instance.bucket.return_value = mock_bucket_instance

        MockStorageClient.return_value = mock_storage_client_instance
        ids = [str(uuid.uuid4()) for i in range(len(image_uris))]
        metadatas = [
            {"image_id": str(i), "source": "postgres"} for i in range(len(image_uris))
        ]
        await image_vs.aadd_images(image_uris, metadatas, ids)
        results = await afetch(engine, (f'SELECT * FROM "{IMAGE_TABLE}"'))
        assert len(results) == len(image_uris)
        assert results[0]["image_id"] == "0"
        assert results[0]["source"] == "postgres"
        await aexecute(engine, (f'TRUNCATE TABLE "{IMAGE_TABLE}"'))

    ##### Custom Vector Store  #####
    async def test_aadd_embeddings(
        self, engine: PGEngine, vs_custom: AsyncPGVectorStore
    ) -> None:
        await vs_custom.aadd_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas
        )
        results = await afetch(engine, f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] == "0"
        assert results[0]["source"] == "postgres"
        await aexecute(engine, f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_aadd_texts_custom(
        self, engine: PGEngine, vs_custom: AsyncPGVectorStore
    ) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] is None
        assert results[0]["source"] is None

        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, metadatas, ids)
        results = await afetch(engine, f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 6
        await aexecute(engine, f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_aadd_docs_custom(
        self, engine: PGEngine, vs_custom: AsyncPGVectorStore
    ) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "postgres"},
            )
            for i in range(len(texts))
        ]
        await vs_custom.aadd_documents(docs, ids=ids)

        results = await afetch(engine, f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] == "0"
        assert results[0]["source"] == "postgres"
        await aexecute(engine, f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_adelete_custom(
        self, engine: PGEngine, vs_custom: AsyncPGVectorStore
    ) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_custom.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{CUSTOM_TABLE}"')
        content = [result["mycontent"] for result in results]
        assert len(results) == 3
        assert "foo" in content
        # delete an ID
        await vs_custom.adelete([ids[0]])
        results = await afetch(engine, f'SELECT * FROM "{CUSTOM_TABLE}"')
        content = [result["mycontent"] for result in results]
        assert len(results) == 2
        assert "foo" not in content
        await aexecute(engine, f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_ignore_metadata_columns(self, engine: PGEngine) -> None:
        column_to_ignore = "source"
        vs = await AsyncPGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            ignore_metadata_columns=[column_to_ignore],
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_json_column="mymeta",
        )
        assert column_to_ignore not in vs.metadata_columns

    async def test_create_vectorstore_with_invalid_parameters_1(
        self, engine: PGEngine
    ) -> None:
        with pytest.raises(ValueError):
            await AsyncPGVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="myembedding",
                metadata_columns=["random_column"],  # invalid metadata column
            )

    async def test_create_vectorstore_with_invalid_parameters_2(
        self, engine: PGEngine
    ) -> None:
        with pytest.raises(ValueError):
            await AsyncPGVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="langchain_id",  # invalid content column type
                embedding_column="myembedding",
                metadata_columns=["random_column"],
            )

    async def test_create_vectorstore_with_invalid_parameters_3(
        self, engine: PGEngine
    ) -> None:
        with pytest.raises(ValueError):
            await AsyncPGVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="random_column",  # invalid embedding column
                metadata_columns=["random_column"],
            )

    async def test_create_vectorstore_with_invalid_parameters_4(
        self, engine: PGEngine
    ) -> None:
        with pytest.raises(ValueError):
            await AsyncPGVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="langchain_id",  # invalid embedding column data type
                metadata_columns=["random_column"],
            )

    async def test_create_vectorstore_with_invalid_parameters_5(
        self, engine: PGEngine
    ) -> None:
        with pytest.raises(ValueError):
            await AsyncPGVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="langchain_id",
                metadata_columns=["random_column"],
                ignore_metadata_columns=[
                    "one",
                    "two",
                ],  # invalid use of metadata_columns and ignore columns
            )

    async def test_create_vectorstore_with_init(self, engine: PGEngine) -> None:
        with pytest.raises(Exception):
            AsyncPGVectorStore(
                key={},
                engine=engine._pool,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="myembedding",
                metadata_columns=["random_column"],  # invalid metadata column
            )
