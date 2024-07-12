import pytest
from langchain_core.vectorstores import VectorStore
from langchain_standard_tests.integration_tests.vectorstores import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)

from tests.unit_tests.test_vectorstore import get_vectorstore, aget_vectorstore


class TestSync(ReadWriteTestSuite):
    @pytest.fixture()
    def vectorstore(self) -> VectorStore:
        """Get an empty vectorstore for unit tests."""
        with get_vectorstore(embedding=self.get_embeddings()) as vstore:
            vstore.drop_tables()
            vstore.create_tables_if_not_exists()
            vstore.create_collection()
            yield vstore


class TestAsync(AsyncReadWriteTestSuite):
    @pytest.fixture()
    async def vectorstore(self) -> VectorStore:
        """Get an empty vectorstore for unit tests."""
        async with aget_vectorstore(embedding=self.get_embeddings()) as vstore:
            await vstore.adrop_tables()
            await vstore.acreate_tables_if_not_exists()
            await vstore.acreate_collection()
            yield vstore
