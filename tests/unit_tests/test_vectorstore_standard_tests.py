from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from tests.unit_tests.test_vectorstore import get_vectorstore


class TestStandard(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        with get_vectorstore(embedding=self.get_embeddings()) as vstore:
            vstore.drop_tables()
            vstore.create_tables_if_not_exists()
            vstore.create_collection()
            yield vstore
