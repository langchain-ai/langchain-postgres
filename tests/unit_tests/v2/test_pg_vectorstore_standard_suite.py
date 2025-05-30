import os
import uuid
from typing import AsyncGenerator, AsyncIterator

import pytest
import pytest_asyncio
from langchain_tests.integration_tests import VectorStoreIntegrationTests
from langchain_tests.integration_tests.vectorstores import EMBEDDING_SIZE
from sqlalchemy import text

from langchain_postgres import Column, PGEngine, PGVectorStore
from tests.utils import VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING

DEFAULT_TABLE = "standard" + str(uuid.uuid4())
DEFAULT_TABLE_SYNC = "sync_standard" + str(uuid.uuid4())


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
# @pytest.mark.filterwarnings("ignore")
@pytest.mark.asyncio
class TestStandardSuiteSync(VectorStoreIntegrationTests):
    @pytest_asyncio.fixture(scope="function")
    async def sync_engine(self) -> AsyncGenerator[PGEngine, None]:
        sync_engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield sync_engine
        await aexecute(sync_engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE_SYNC}"')
        await sync_engine.close()

    @pytest.fixture(scope="function")
    def vectorstore(self, sync_engine: PGEngine) -> PGVectorStore:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        sync_engine.init_vectorstore_table(
            DEFAULT_TABLE_SYNC,
            EMBEDDING_SIZE,
            id_column=Column(name="langchain_id", data_type="VARCHAR", nullable=False),
        )

        vs = PGVectorStore.create_sync(
            sync_engine,
            embedding_service=self.get_embeddings(),
            table_name=DEFAULT_TABLE_SYNC,
        )
        yield vs


@pytest.mark.enable_socket
@pytest.mark.asyncio
class TestStandardSuiteAsync(VectorStoreIntegrationTests):
    @pytest_asyncio.fixture(scope="function")
    async def async_engine(self) -> AsyncIterator[PGEngine]:
        async_engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield async_engine
        await aexecute(async_engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await async_engine.close()

    @pytest_asyncio.fixture(scope="function")
    async def vectorstore(  # type: ignore[override]
        self, async_engine: PGEngine
    ) -> AsyncGenerator[PGVectorStore, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        await async_engine.ainit_vectorstore_table(
            DEFAULT_TABLE,
            EMBEDDING_SIZE,
            id_column=Column(name="langchain_id", data_type="VARCHAR", nullable=False),
        )

        vs = await PGVectorStore.create(
            async_engine,
            embedding_service=self.get_embeddings(),
            table_name=DEFAULT_TABLE,
        )

        yield vs
