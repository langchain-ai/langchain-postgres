"""Tests for binding the hybrid search text language as a SQL parameter."""

import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy.exc import ProgrammingError

from langchain_postgres import Column, PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
)
from tests.unit_tests.fixtures.metadata_filtering_data import METADATAS
from tests.utils import VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING

# Payload from the external report; must reach the database as data.
LANGUAGE_PAYLOAD = (
    "x'), plainto_tsquery('x')) AS distance, (SELECT 1) AS canary"
    ' FROM "public"."t" --'
)

pytestmark = pytest.mark.enable_socket

texts = ["foo", "bar", "baz", "boo"]
ids = [str(uuid.uuid4()) for _ in texts]
docs = [
    Document(page_content=texts[i], metadata=METADATAS[i]) for i in range(len(texts))
]

VECTOR_SIZE = 768
embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)


class TestTSVLangParameterBinding:
    @pytest_asyncio.fixture
    async def engine(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield engine
        await engine.close()

    @pytest_asyncio.fixture
    async def tsv_lang_store(
        self, engine: PGEngine
    ) -> AsyncIterator[AsyncPGVectorStore]:
        table_name = "tsv_lang_" + str(uuid.uuid4()).replace("-", "_")
        await engine._ainit_vectorstore_table(
            table_name,
            VECTOR_SIZE,
            id_column=Column("langchain_id", "TEXT"),
            store_metadata=False,
        )
        store = await AsyncPGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=table_name,
            hybrid_search_config=HybridSearchConfig(
                tsv_column="",  # no TSV column; language reaches on-the-fly tsvectors
                tsv_lang="pg_catalog.english",
                fusion_function=reciprocal_rank_fusion,
                fusion_function_parameters={"rrf_k": 60, "fetch_top_k": 10},
            ),
        )
        await store.aadd_documents(docs, ids=ids)
        yield store
        await engine.adrop_table(table_name)

    async def test_valid_language_hybrid_search(
        self, tsv_lang_store: AsyncPGVectorStore
    ) -> None:
        results = await tsv_lang_store.asimilarity_search(
            "foo",
            k=2,
            hybrid_search_config=HybridSearchConfig(tsv_lang="pg_catalog.english"),
        )
        assert len(results) == 2
        assert results[0] == Document(page_content="foo", id=ids[0])

    @pytest.mark.parametrize("tsv_lang", [None, ""])
    async def test_unset_language_hybrid_search(
        self, tsv_lang_store: AsyncPGVectorStore, tsv_lang: str | None
    ) -> None:
        results = await tsv_lang_store.asimilarity_search(
            "foo",
            k=2,
            hybrid_search_config=HybridSearchConfig(tsv_lang=tsv_lang),
        )
        assert len(results) == 2
        assert results[0] == Document(page_content="foo", id=ids[0])

    async def test_report_payload_raises_database_error(
        self, tsv_lang_store: AsyncPGVectorStore
    ) -> None:
        with pytest.raises(ProgrammingError) as exc_info:
            await tsv_lang_store.asimilarity_search(
                "foo",
                k=2,
                hybrid_search_config=HybridSearchConfig(tsv_lang=LANGUAGE_PAYLOAD),
            )
        cause = exc_info.value.__cause__
        # The payload must fail as an invalid regconfig literal inside the
        # bound parameter (SQLSTATE 42602, "invalid name syntax"), never as
        # a syntax error from spliced SQL (SQLSTATE 42601).
        assert cause is not None
        assert getattr(cause, "sqlstate", None) == "42602", cause
