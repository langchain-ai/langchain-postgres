"""
Integration Test for PGVector Metadata Deserialization

This test verifies that PGVector correctly deserializes stored metadata into a Python dictionary.
It connects to a live PostgreSQL instance (with the pgvector extension enabled) and uses
FakeEmbeddings (from langchain_core) for testing purposes.

Workflow:
  1. Connect to the PostgreSQL instance (pgvector container on port 6024).
  2. Initialize PGVector with FakeEmbeddings, disabling automatic extension creation.
  3. Manually create the vector extension.
  4. Create the collection and tables.
  5. Insert test documents with known metadata.
  6. Perform a similarity search.
  7. Assert that each returned document's metadata is a Python dict.
  8. Dispose the async engine to clean up connections.

Usage:
  Ensure your Docker Compose container for pgvector is running, then run:
    poetry run pytest tests/integration/test_pgvector_metadata_integration.py
"""

import asyncio
import pytest
from sqlalchemy import text
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings


@pytest.mark.asyncio
async def test_pgvector_metadata_integration() -> None:
    """
    Test that PGVector deserializes metadata correctly.

    Steps:
      - Connect to a PostgreSQL instance with the pgvector extension.
      - Initialize PGVector with FakeEmbeddings and disable automatic extension creation.
      - Manually create the pgvector extension so that the "vector" type is available.
      - Create the collection and tables.
      - Insert documents with predefined metadata.
      - Perform a similarity search.
      - Verify that each returned document's metadata is a dict.
    """
    # Connection string for the PostgreSQL instance (pgvector container on port 6024)
    connection: str = "postgresql+asyncpg://langchain:langchain@localhost:6024/langchain"

    # Instantiate FakeEmbeddings for testing (vector size set to 1352)
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
        pre_delete_collection=True,  # Ensure a fresh collection for testing
    )

    # Manually create the pgvector extension so that the "vector" type exists.
    async with vectorstore._async_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    # Create the collection (and tables) in the database.
    await vectorstore.acreate_collection()

    # Prepare sample documents with predefined metadata.
    documents = [
        Document(page_content="Document 1", metadata={"user": "foo"}),
        Document(page_content="Document 2", metadata={"user": "bar"}),
    ]

    # Add the documents to the vectorstore.
    await vectorstore.aadd_texts(
        texts=[doc.page_content for doc in documents],
        metadatas=[doc.metadata for doc in documents]
    )

    # Perform a similarity search; the query should match the inserted documents.
    results = await vectorstore.asimilarity_search_with_score("Document", k=2)

    # Verify that each returned document's metadata is deserialized as a dict.
    for doc, score in results:
        assert isinstance(doc.metadata, dict), f"Metadata is not a dict: {doc.metadata}"
        print(f"Document: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

    # Clean up: Dispose the async engine to close connections.
    await vectorstore._async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_pgvector_metadata_integration())
