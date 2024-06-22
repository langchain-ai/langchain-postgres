"""Test PGVector functionality."""
import contextlib
from abc import ABC, abstractmethod
from typing import Generator, List

import pytest
from langchain_core.documents import Document
from langchain_core.indexes import Index

# from langchain_postgres.indexer.document_indexer import PostgresDocumentIndex
from langchain_postgres.vectorstores2 import PGVector
from tests.unit_tests.fake_embeddings import FakeEmbeddings
from tests.utils import VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING

ADA_TOKEN_COUNT = 12


class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(0.0)]


@contextlib.contextmanager
def get_document_indexer() -> Generator[Index, None, None]:
    """Get a pre-populated-vectorstore"""
    embeddings = FakeEmbeddingsWithAdaDimension()
    indexer = PGVector(
        embeddings=embeddings,
        collection_name="test_collection2",
        connection=CONNECTION_STRING,
    )
    try:
        yield indexer
    finally:
        indexer.drop_tables()  # Postgres specific method


class VectorIndexTest(ABC):
    @abstractmethod
    @pytest.fixture
    def index(self) -> Index:
        """Get the vectorstore class to test.

        The returned vectorstore should be EMPTY.
        """

    def test_upsert_is_idempotent(self, index: Index) -> None:
        """Verify that upsert works as an upsert."""
        documents = [
            Document(page_content="id 1", metadata={"id": 1}),
            Document(page_content="id 2", metadata={"id": 2}),
        ]
        index.upsert(documents=documents, ids=["1", "2"])
        assert index.get_by_ids(ids=["1", "2"]) == documents
        index.upsert(documents=documents, ids=["1", "2"])
        assert index.get_by_ids(ids=["1", "2"]) == documents

    def test_upsert_mutation(self, index: Index) -> None:
        """Test that the underlying content can be changed."""
        documents = [
            Document(page_content="id 1", metadata={"id": 1}),
            Document(page_content="id 2", metadata={"id": 2}),
        ]

        new_documents = [
            Document(page_content="new id 1", metadata={"new_id": 1}),
            Document(page_content="new id 2", metadata={"new_id": 2}),
            Document(page_content="new id 3", metadata={"new_id": 3}),
        ]

        index.upsert(documents=documents, ids=["1", "2"])
        assert index.get_by_ids(ids=["1", "2"]) == documents
        index.upsert(documents=new_documents, ids=["1", "2", "3"])
        assert index.get_by_ids(ids=["1", "2"]) == new_documents[:2]

    def test_deletion_does_not_error(self, index: Index) -> None:
        """Test that deletion does not error."""
        assert index.get_by_ids(ids=["1", "2"]) == []
        index.delete_by_ids(ids=["1", "2"])
        assert index.get_by_ids(ids=["1", "2"]) == []

    def test_lazy_get_no_filters(self, index: Index) -> None:
        """Test lazy get without filters."""
        documents = [
            Document(page_content="id 1", metadata={"id": 1}),
            Document(page_content="id 2", metadata={"id": 2}),
        ]
        index.upsert(documents=documents, ids=["1", "2"])
        assert isinstance(index.lazy_get(), Generator)
        results = list(index.lazy_get())
        assert results == documents

    def test_lazy_get(self, index: Index) -> None:
        """Test lazy get with filters."""
        documents = [
            Document(page_content="id 1", metadata={"id": 1, "is_active": True}),
            Document(page_content="id 2", metadata={"id": 2, "is_active": False}),
        ]
        index.upsert(documents=documents, ids=["1", "2"])

        results = list(index.lazy_get(filters={"is_active": True}))
        assert results == [documents[0]]
        # Invert the filter
        results = list(index.lazy_get(filters={"is_active": False}))
        assert results == [documents[1]]

    def test_upsert_by_vector(self, index: Index) -> None:
        """Test upsert by vector."""
        documents = [
            Document(page_content="id 1", metadata={"id": 1}),
            Document(page_content="id 2", metadata={"id": 2}),
        ]
        index.upsert(documents=documents, vectors=[[1.0] * 3] * 3, ids=["1", "2"])
        assert index.get_by_ids(ids=["1", "2"]) == documents
        assert index.get_by_ids(ids=["1", "2"]) == documents


class TestPostgresDocumentIndex(VectorIndexTest):
    @pytest.fixture()
    def index(self) -> Index:
        """Get the vectorstore class to test."""
        with get_document_indexer() as indexer:
            yield indexer
