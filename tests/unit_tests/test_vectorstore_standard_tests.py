from typing import AsyncGenerator, Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from tests.unit_tests.test_vectorstore import aget_vectorstore, get_vectorstore


class TestSync(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        with get_vectorstore(embedding=self.get_embeddings()) as vstore:
            vstore.drop_tables()
            vstore.create_tables_if_not_exists()
            vstore.create_collection()
            yield vstore

    @property
    def has_async(self) -> bool:
        return False  # Skip async tests for sync vector store


class TestAsync(VectorStoreIntegrationTests):
    @pytest.fixture()
    async def vectorstore(self) -> AsyncGenerator[VectorStore, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        async with aget_vectorstore(embedding=self.get_embeddings()) as vstore:
            await vstore.adrop_tables()
            await vstore.acreate_tables_if_not_exists()
            await vstore.acreate_collection()
            yield vstore

    @property
    def has_sync(self) -> bool:
        return False  # Skip sync tests for async vector store
