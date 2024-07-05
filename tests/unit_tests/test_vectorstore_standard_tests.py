import pytest
from langchain_core.vectorstores import VectorStore
from langchain_standard_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)

from tests.unit_tests.test_vectorstore import aget_vectorstore, get_vectorstore


class TestSync(ReadWriteTestSuite):
    @pytest.fixture()
    def vectorstore(self) -> VectorStore:
        """Get an empty vectorstore for unit tests."""
        with get_vectorstore(embedding=self.get_embeddings()) as vectorstore:
            vectorstore.drop_tables()
            vectorstore.create_tables_if_not_exists()
            vectorstore.create_collection()
            yield vectorstore


class TestAsync(AsyncReadWriteTestSuite):
    @pytest.fixture()
    async def vectorstore(self) -> VectorStore:
        """Get an empty vectorstore for unit tests."""
        async with aget_vectorstore(embedding=self.get_embeddings()) as vectorstore:
            await vectorstore.adrop_tables()
            await vectorstore.acreate_tables_if_not_exists()
            await vectorstore.acreate_collection()
            yield vectorstore
