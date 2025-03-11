"""
Integration Test for PGVector Metadata Deserialization Using a Temporary Test Database

This integration test verifies that PGVector correctly deserializes stored metadata into a Python dictionary.
It follows these steps:
  1. Dynamically create a temporary test database (named "langchain_test_<random_suffix>") using a session-scoped
     pytest fixture. This ensures an isolated environment for testing.
  2. Initialize PGVector with FakeEmbeddings while disabling automatic extension creation.
  3. Manually create the pgvector extension (using AUTOCOMMIT) so that the custom "vector" type becomes available.
  4. Create the collection and tables in the temporary database.
  5. Insert test documents with predefined metadata.
  6. Execute a similarity search.
  7. Assert that each returned document’s metadata is deserialized as a Python dict.
  8. Finally, clean up by terminating connections and dropping the temporary database.

Usage:
  Ensure your PostgreSQL instance (with pgvector enabled) is running, then execute:
    poetry run pytest tests/integration/test_pgvector_metadata_integration.py
"""

import asyncio
import uuid
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings


@pytest.fixture(scope="session")
async def test_database_url():
    """
    Create a temporary test database for integration testing and drop it afterwards.

    The temporary database name is generated as "langchain_test_<random_suffix>".
    This fixture connects to the default "postgres" database (with AUTOCOMMIT enabled) to run
    CREATE/DROP DATABASE commands outside of a transaction block. It yields the connection string
    for the temporary test database.

    Yields:
        str: Connection string for the temporary test database.
    """
    # Generate a unique database name.
    db_suffix = uuid.uuid4().hex
    test_db = f"langchain_test_{db_suffix}"

    # Connection string to the default 'postgres' database.
    default_db_url = "postgresql+asyncpg://langchain:langchain@localhost:6024/postgres"
    engine = create_async_engine(default_db_url, isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        await conn.execute(text(f"CREATE DATABASE {test_db}"))
    await engine.dispose()

    # Build the test database connection string.
    test_db_url = f"postgresql+asyncpg://langchain:langchain@localhost:6024/{test_db}"
    yield test_db_url

    # Teardown: Drop the temporary test database.
    engine = create_async_engine(default_db_url, isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        # Terminate active connections to the test database.
        await conn.execute(text(f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{test_db}'
              AND pid <> pg_backend_pid();
        """))
        await conn.execute(text(f"DROP DATABASE IF EXISTS {test_db}"))
    await engine.dispose()


@pytest.mark.asyncio
async def test_pgvector_metadata_integration(test_database_url: str) -> None:
    """
    Integration test for PGVector metadata deserialization using a temporary test database.

    Steps:
      1. Use the temporary database connection string provided by the fixture.
      2. Initialize PGVector with FakeEmbeddings and disable automatic extension creation.
      3. Manually create the pgvector extension using AUTOCOMMIT so that the "vector" type is available.
      4. Create the collection (and tables) in the temporary database.
      5. Insert test documents with predefined metadata.
      6. Execute a similarity search.
      7. Assert that each returned document's metadata is deserialized as a Python dict.
    """
    # Use the temporary test database connection string.
    connection = test_database_url

    # Initialize FakeEmbeddings (with a vector size of 1352 for testing purposes).
    embeddings = FakeEmbeddings(size=1352)

    # Initialize PGVector in asynchronous mode.
    vectorstore = PGVector(
        embeddings=embeddings,
        connection=connection,
        collection_name="integration_test_metadata",
        use_jsonb=True,
        async_mode=True,
        create_extension=False,  # Disable auto extension creation; we'll do it manually.
        pre_delete_collection=True,  # Ensure a fresh collection.
    )

    # Manually create the pgvector extension on the test database.
    async with vectorstore._async_engine.connect() as conn:
        # Await the execution_options method to obtain a connection with AUTOCOMMIT.
        conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    # Create the collection (and tables) in the database.
    await vectorstore.acreate_collection()

    # Define sample documents with predefined metadata.
    documents = [
        Document(page_content="Document 1", metadata={"user": "foo"}),
        Document(page_content="Document 2", metadata={"user": "bar"}),
    ]

    # Add the documents to the vectorstore.
    await vectorstore.aadd_texts(
        texts=[doc.page_content for doc in documents],
        metadatas=[doc.metadata for doc in documents]
    )

    # Perform a similarity search.
    results = await vectorstore.asimilarity_search_with_score("Document", k=2)

    # Verify that each returned document's metadata is deserialized as a Python dict.
    for doc, score in results:
        assert isinstance(doc.metadata, dict), f"Metadata is not a dict: {doc.metadata}"
        print(f"Document: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

    # Clean up: Dispose the async engine to close all connections.
    await vectorstore._async_engine.dispose()
