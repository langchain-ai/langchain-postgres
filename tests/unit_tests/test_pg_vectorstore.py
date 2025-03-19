import asyncio
import os
import uuid
from threading import Thread
from typing import AsyncIterator, Iterator, Sequence

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from langchain_postgres import Column, PGEngine, PGVectorStore
from tests.utils import VECTORSTORE_CONNECTION_STRING_ASYNCPG as CONNECTION_STRING

DEFAULT_TABLE = "test_table" + str(uuid.uuid4())
DEFAULT_TABLE_SYNC = "test_table_sync" + str(uuid.uuid4())
CUSTOM_TABLE = "test-table-custom" + str(uuid.uuid4())
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
@pytest.mark.asyncio(scope="class")
class TestVectorStore:
    @pytest_asyncio.fixture(scope="class")
    async def engine(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

        yield engine
        await aexecute(engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine: PGEngine) -> AsyncIterator[PGVectorStore]:
        await engine.ainit_vectorstore_table(DEFAULT_TABLE, VECTOR_SIZE)
        vs = await PGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def engine_sync(self) -> AsyncIterator[PGEngine]:
        engine_sync = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield engine_sync

        await aexecute(engine_sync, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE_SYNC}"')
        await engine_sync.close()

    @pytest_asyncio.fixture(scope="class")
    def vs_sync(self, engine_sync: PGEngine) -> Iterator[PGVectorStore]:
        engine_sync.init_vectorstore_table(DEFAULT_TABLE_SYNC, VECTOR_SIZE)

        vs = PGVectorStore.create_sync(
            engine_sync,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE_SYNC,
        )
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def vs_custom(self, engine: PGEngine) -> AsyncIterator[PGVectorStore]:
        await engine.ainit_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            metadata_json_column="mymeta",
        )
        vs = await PGVectorStore.create(
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
        await aexecute(engine, f'DROP TABLE IF EXISTS "{CUSTOM_TABLE}"')

    async def test_init_with_constructor(self, engine: PGEngine) -> None:
        with pytest.raises(Exception):
            PGVectorStore(  # type: ignore
                engine=engine,
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
            await PGVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="noname",
                embedding_column="myembedding",
                metadata_columns=["page", "source"],
                metadata_json_column="mymeta",
            )

    async def test_aadd_texts(self, engine: PGEngine, vs: PGVectorStore) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, metadatas, ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 6
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_cross_env_add_texts(
        self, engine: PGEngine, vs: PGVectorStore
    ) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs.add_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        vs.delete(ids)
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_texts_edge_cases(
        self, engine: PGEngine, vs: PGVectorStore
    ) -> None:
        texts = ["Taylor's", '"Swift"', "best-friend"]
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_docs(self, engine: PGEngine, vs: PGVectorStore) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_documents(docs, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_embeddings(
        self, engine: PGEngine, vs_custom: PGVectorStore
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

    async def test_adelete(self, engine: PGEngine, vs: PGVectorStore) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs.aadd_texts(texts, ids=ids)
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3
        # delete an ID
        await vs.adelete([ids[0]])
        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 2
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')

    async def test_aadd_texts_custom(
        self, engine: PGEngine, vs_custom: PGVectorStore
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
        self, engine: PGEngine, vs_custom: PGVectorStore
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
        self, engine: PGEngine, vs_custom: PGVectorStore
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

    async def test_add_docs(
        self, engine_sync: PGEngine, vs_sync: PGVectorStore
    ) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs_sync.add_documents(docs, ids=ids)
        results = await afetch(engine_sync, f'SELECT * FROM "{DEFAULT_TABLE_SYNC}"')
        assert len(results) == 3
        vs_sync.delete(ids)
        await aexecute(engine_sync, f'TRUNCATE TABLE "{DEFAULT_TABLE_SYNC}"')

    async def test_add_texts(
        self, engine_sync: PGEngine, vs_sync: PGVectorStore
    ) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs_sync.add_texts(texts, ids=ids)
        results = await afetch(engine_sync, f'SELECT * FROM "{DEFAULT_TABLE_SYNC}"')
        assert len(results) == 3
        await vs_sync.adelete(ids)
        await aexecute(engine_sync, f'TRUNCATE TABLE "{DEFAULT_TABLE_SYNC}"')

    async def test_cross_env(
        self, engine_sync: PGEngine, vs_sync: PGVectorStore
    ) -> None:
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        await vs_sync.aadd_texts(texts, ids=ids)
        results = await afetch(engine_sync, f'SELECT * FROM "{DEFAULT_TABLE_SYNC}"')
        assert len(results) == 3
        await vs_sync.adelete(ids)
        await aexecute(engine_sync, f'TRUNCATE TABLE "{DEFAULT_TABLE_SYNC}"')

    async def test_add_embeddings(
        self, engine_sync: PGEngine, vs_custom: PGVectorStore
    ) -> None:
        vs_custom.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=[
                {"page": str(i), "source": "postgres"} for i in range(len(texts))
            ],
        )
        results = await afetch(engine_sync, f'SELECT * FROM "{CUSTOM_TABLE}"')
        assert len(results) == 3
        assert results[0]["mycontent"] == "foo"
        assert results[0]["myembedding"]
        assert results[0]["page"] == "0"
        assert results[0]["source"] == "postgres"
        await aexecute(engine_sync, f'TRUNCATE TABLE "{CUSTOM_TABLE}"')

    async def test_create_vectorstore_with_invalid_parameters(
        self, engine: PGEngine
    ) -> None:
        with pytest.raises(ValueError):
            await PGVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="myembedding",
                metadata_columns=["random_column"],  # invalid metadata column
            )
        with pytest.raises(ValueError):
            await PGVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="langchain_id",  # invalid content column type
                embedding_column="myembedding",
                metadata_columns=["random_column"],
            )
        with pytest.raises(ValueError):
            await PGVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="random_column",  # invalid embedding column
                metadata_columns=["random_column"],
            )
        with pytest.raises(ValueError):
            await PGVectorStore.create(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="mycontent",
                embedding_column="langchain_id",  # invalid embedding column data type
                metadata_columns=["random_column"],
            )

    async def test_from_engine(self) -> None:
        async_engine = create_async_engine(url=CONNECTION_STRING)

        engine = PGEngine.from_engine(async_engine)
        table_name = "test_table" + str(uuid.uuid4()).replace("-", "_")
        await engine.ainit_vectorstore_table(table_name, VECTOR_SIZE)
        vs = await PGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=table_name,
        )
        await vs.aadd_texts(["foo"])
        results = await afetch(engine, f"SELECT * FROM {table_name}")
        assert len(results) == 1

        await aexecute(engine, f"DROP TABLE {table_name}")
        await engine.close()

    async def test_from_engine_loop_connector(
        self,
    ) -> None:
        async def init_connection_pool() -> AsyncEngine:
            pool = create_async_engine(url=CONNECTION_STRING)
            return pool

        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever, daemon=True)
        thread.start()

        coro = init_connection_pool()
        pool = asyncio.run_coroutine_threadsafe(coro, loop).result()
        engine = PGEngine.from_engine(pool, loop)
        table_name = "test_table" + str(uuid.uuid4()).replace("-", "_")
        await engine.ainit_vectorstore_table(table_name, VECTOR_SIZE)
        vs = await PGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=table_name,
        )
        await vs.aadd_texts(["foo"])
        vs.add_texts(["foo"])
        results = await afetch(engine, f"SELECT * FROM {table_name}")
        assert len(results) == 2

        await aexecute(engine, f"TRUNCATE TABLE {table_name}")
        await engine.close()

        vs = PGVectorStore.create_sync(
            engine,
            embedding_service=embeddings_service,
            table_name=table_name,
        )
        await vs.aadd_texts(["foo"])
        vs.add_texts(["foo"])
        results = await afetch(engine, f"SELECT * FROM {table_name}")
        assert len(results) == 2

        await aexecute(engine, f"DROP TABLE {table_name}")

    async def test_from_connection_string(self) -> None:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        table_name = "test_table" + str(uuid.uuid4()).replace("-", "_")
        await engine.ainit_vectorstore_table(table_name, VECTOR_SIZE)
        vs = await PGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=table_name,
        )
        await vs.aadd_texts(["foo"])
        vs.add_texts(["foo"])
        results = await afetch(engine, f"SELECT * FROM {table_name}")
        assert len(results) == 2

        await aexecute(engine, f"TRUNCATE TABLE {table_name}")
        vs = PGVectorStore.create_sync(
            engine,
            embedding_service=embeddings_service,
            table_name=table_name,
        )
        await vs.aadd_texts(["foo"])
        vs.add_texts(["bar"])
        results = await afetch(engine, f"SELECT * FROM {table_name}")
        assert len(results) == 2
        await aexecute(engine, f"DROP TABLE {table_name}")
        await engine.close()

    async def test_from_engine_loop(self) -> None:
        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever, daemon=True)
        thread.start()
        pool = create_async_engine(url=CONNECTION_STRING)
        engine = PGEngine.from_engine(pool, loop)

        table_name = "test_table" + str(uuid.uuid4()).replace("-", "_")
        await engine.ainit_vectorstore_table(table_name, VECTOR_SIZE)
        vs = await PGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=table_name,
        )
        await vs.aadd_texts(["foo"])
        vs.add_texts(["foo"])
        results = await afetch(engine, f"SELECT * FROM {table_name}")
        assert len(results) == 2

        await aexecute(engine, f"TRUNCATE TABLE {table_name}")
        vs = PGVectorStore.create_sync(
            engine,
            embedding_service=embeddings_service,
            table_name=table_name,
        )
        await vs.aadd_texts(["foo"])
        vs.add_texts(["bar"])
        results = await afetch(engine, f"SELECT * FROM {table_name}")
        assert len(results) == 2
        await aexecute(engine, f"DROP TABLE {table_name}")
        await engine.close()

    @pytest.mark.filterwarnings("ignore")
    def test_get_table_name(self, vs: PGVectorStore) -> None:
        assert vs.get_table_name() == DEFAULT_TABLE
