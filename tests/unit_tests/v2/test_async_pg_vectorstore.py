import uuid
from typing import AsyncIterator, Sequence

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from langchain_postgres import Column, PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from tests.utils import VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING

DEFAULT_TABLE = "default" + str(uuid.uuid4())
DEFAULT_TABLE_SYNC = "default_sync" + str(uuid.uuid4())
CUSTOM_TABLE = "custom" + str(uuid.uuid4())
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz"]
metadatas = [{"page": str(i), "source": "postgres"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]

embeddings = [embeddings_service.embed_query(texts[i]) for i in range(len(texts))]


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
        assert not result
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_adelete_with_filter(
        self, engine: PGEngine, vs: AsyncPGVectorStore
    ) -> None:
        """Test deletion by metadata filter."""
        # Add texts with different metadata
        test_metadatas = [
            {"page": "0", "source": "postgres", "category": "docs"},
            {"page": "1", "source": "web", "category": "docs"},
            {"page": "2", "source": "postgres", "category": "blog"},
        ]
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, metadatas=test_metadatas, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

        # Delete all documents with source="postgres"
        await vs.adelete(filter={"source": "postgres"})
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 1
        # The remaining document should have source="web"
        assert results[0]["langchain_metadata"]["source"] == "web"
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_adelete_with_filter_and_operator(
        self, engine: PGEngine, vs: AsyncPGVectorStore
    ) -> None:
        """Test deletion with filter using operators."""
        # Add texts with different metadata including numeric values
        test_metadatas = [
            {"page": "0", "source": "postgres", "year": 2020},
            {"page": "1", "source": "web", "year": 2021},
            {"page": "2", "source": "postgres", "year": 2022},
        ]
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, metadatas=test_metadatas, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

        # Delete all documents with year < 2022
        await vs.adelete(filter={"year": {"$lt": 2022}})
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 1
        # The remaining document should have year=2022
        assert results[0]["langchain_metadata"]["year"] == 2022
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_adelete_with_complex_filter(
        self, engine: PGEngine, vs: AsyncPGVectorStore
    ) -> None:
        """Test deletion with complex filter using $and."""
        # Add texts with different metadata
        test_metadatas = [
            {"page": "0", "source": "postgres", "category": "obsolete"},
            {"page": "1", "source": "web", "category": "obsolete"},
            {"page": "2", "source": "postgres", "category": "current"},
        ]
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, metadatas=test_metadatas, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

        # Delete documents with source="postgres" AND category="obsolete"
        await vs.adelete(
            filter={"$and": [{"source": "postgres"}, {"category": "obsolete"}]}
        )
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 2
        # Should have removed only the first document
        remaining_categories = [
            result["langchain_metadata"]["category"] for result in results
        ]
        assert "obsolete" in remaining_categories  # web/obsolete still exists
        assert "current" in remaining_categories  # postgres/current still exists
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_adelete_with_filter_and_ids(
        self, engine: PGEngine, vs: AsyncPGVectorStore
    ) -> None:
        """Test deletion with both IDs and filter (must match both)."""
        # Add texts with different metadata
        test_metadatas = [
            {"page": "0", "source": "postgres"},
            {"page": "1", "source": "web"},
            {"page": "2", "source": "postgres"},
        ]
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, metadatas=test_metadatas, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

        # Try to delete ids[0] and ids[2] but only where source="web"
        # This should delete nothing since ids[0] and ids[2] have source="postgres"
        # (well, actually it should delete if ANY match since we're using AND logic)
        # Let me reconsider: with AND logic, it means id IN (ids) AND source="web"
        # So this should only delete if the id is in the list AND source is web
        # Since ids[0] and ids[2] are postgres, and ids[1] is web but not in the list,
        # nothing should be deleted
        await vs.adelete(ids=[ids[0], ids[2]], filter={"source": "web"})
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3  # Nothing deleted

        # Now delete ids[0] and ids[1] where source="web"
        # This should delete only ids[1] (which has source="web")
        await vs.adelete(ids=[ids[0], ids[1]], filter={"source": "web"})
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 2
        remaining_ids = [result["langchain_id"] for result in results]
        assert ids[1] not in remaining_ids  # ids[1] was deleted
        assert ids[0] in remaining_ids  # ids[0] not deleted (wrong source)
        assert ids[2] in remaining_ids  # ids[2] not deleted (not in id list)
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_adelete_with_filter_no_matches(
        self, engine: PGEngine, vs: AsyncPGVectorStore
    ) -> None:
        """Test deletion with filter that matches no documents."""
        # Add texts
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, metadatas=metadatas, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

        # Try to delete with a filter that matches nothing
        await vs.adelete(filter={"source": "nonexistent"})
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3  # Nothing deleted
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

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
