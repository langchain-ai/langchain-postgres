"""
Integration Test for PGVector Metadata Deserialization

This test verifies that PGVector correctly deserializes stored metadata into a Python dictionary.
It connects to a live PostgreSQL instance (with the pgvector extension enabled) and uses
FakeEmbeddings from langchain_core for testing purposes.

Workflow:
  1. Connect to the PostgreSQL instance via asyncpg.
  2. Initialize PGVector with create_extension=False (to avoid multi-command issues).
  3. Manually create the pgvector extension in the database.
  4. Create the collection (and tables) in the database.
  5. Insert test documents with known metadata.
  6. Perform a similarity search.
  7. Assert that each returned document's metadata is a dict.

Usage:
  Make sure your Docker Compose container for pgvector (exposed on port 6024) is running.
  Then run:
    poetry run pytest tests/integration/test_pgvector_metadata_integration.py
"""

import pytest
from sqlalchemy import text
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings


@pytest.fixture(scope="module")
async def pgvector_instance():
    """
    Fixture to create and tear down a PGVector instance for integration testing.
    """
    # Connection string for the PostgreSQL instance (pgvector container on port 6024)
    connection: str = "postgresql+asyncpg://langchain:langchain@localhost:6024/langchain"

    # Instantiate FakeEmbeddings for testing (with a vector size of 1352)
    embeddings = FakeEmbeddings(size=1352)

    # Initialize PGVector in async mode.
    # Set create_extension=False to avoid multi-command issues.
    vectorstore = PGVector(
        embeddings=embeddings,
        connection=connection,
        collection_name="integration_test_metadata",
        use_jsonb=True,
        async_mode=True,
        create_extension=False,  # Disable automatic extension creation
        pre_delete_collection=True  # Ensure a fresh collection for testing
    )

    # Manually create the pgvector extension so that the "vector" type is available.
    async with vectorstore._async_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    # Create the collection (and corresponding tables) in the database.
    await vectorstore.acreate_collection()

    yield vectorstore

    # Teardown: Drop the collection (and tables) after tests.
    await vectorstore.adelete_collection()


@pytest.mark.asyncio
async def test_pgvector_metadata_integration(pgvector_instance: PGVector) -> None:
    """
    Test that PGVector deserializes metadata correctly.

    Steps:
      - Insert documents with predefined metadata.
      - Perform a similarity search.
      - Verify that each returned document's metadata is a Python dict.
    """
    vectorstore = pgvector_instance

    # Prepare sample documents with predefined metadata.
    documents = [
        Document(page_content="Document 1", metadata={"user": "foo"}),
        Document(page_content="Document 2", metadata={"user": "bar"})
    ]

    # Add the documents to the vectorstore.
    await vectorstore.aadd_texts(
        texts=[doc.page_content for doc in documents],
        metadatas=[doc.metadata for doc in documents]
    )

    # Perform a similarity search that should match the inserted documents.
    results = await vectorstore.asimilarity_search_with_score("Document", k=2)

    # Verify that each returned document has its metadata deserialized as a dict.
    for doc, score in results:
        assert isinstance(doc.metadata, dict), f"Metadata is not a dict: {doc.metadata}"
        print(f"Document: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
