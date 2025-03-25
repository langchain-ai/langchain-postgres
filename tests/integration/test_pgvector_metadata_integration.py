"""
Integration Test for PGVector Retrieval Methods Using a Temporary Test Database

This integration test verifies that PGVector retrieval methods work as expected, including:
  - Retriever interface (ainvoke)
  - asimilarity_search_with_score (retrieval with score by query string)
  - asimilarity_search_by_vector (retrieval by query embedding)
  - asimilarity_search_with_score_by_vector (retrieval with score by query embedding)

Steps:
  1. Dynamically create a temporary test database (named "langchain_test_<random_suffix>") using a session-scoped
     pytest fixture. This ensures an isolated environment for testing.
  2. Initialize PGVector with FakeEmbeddings while disabling automatic extension creation.
  3. Manually create the pgvector extension (using AUTOCOMMIT) so that the custom "vector" type becomes available.
  4. Create the collection and tables in the temporary database.
  5. Insert test documents with predefined metadata.
  6. Perform retrieval using:
      a) the retriever interface,
      b) asimilarity_search_with_score (by query string),
      c) asimilarity_search_by_vector (by query embedding),
      d) asimilarity_search_with_score_by_vector (by query embedding).
  7. Assert that each returned document's metadata is deserialized as a Python dict and that scores (if applicable) are floats.
  8. Finally, clean up by disposing the async engine and dropping the temporary database.

Usage:
  Ensure your PostgreSQL instance (with pgvector enabled) is running, then execute:
    poetry run pytest -s tests/integration/test_pgvector_retrieval_integration.py
"""

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
    """
    db_suffix = uuid.uuid4().hex
    test_db = f"langchain_test_{db_suffix}"
    default_db_url = "postgresql+psycopg://langchain:langchain@localhost:6024/postgres"
    engine = create_async_engine(default_db_url, isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        await conn.execute(text(f"CREATE DATABASE {test_db}"))
    await engine.dispose()

    test_db_url = f"postgresql+psycopg://langchain:langchain@localhost:6024/{test_db}"
    yield test_db_url

    engine = create_async_engine(default_db_url, isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        await conn.execute(text(f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{test_db}'
              AND pid <> pg_backend_pid();
        """))
        await conn.execute(text(f"DROP DATABASE IF EXISTS {test_db}"))
    await engine.dispose()


@pytest.mark.asyncio
async def test_all_retrieval_methods(test_database_url: str) -> None:
    """
    Integration test for all PGVector retrieval methods.

    This test verifies:
      a) The retriever interface via ainvoke().
      b) asimilarity_search_with_score() using a query string.
      c) asimilarity_search_by_vector() using a query embedding.
      d) asimilarity_search_with_score_by_vector() using a query embedding.

    In all cases, the metadata of returned documents should be deserialized as a Python dict,
    and where scores are provided, they must be floats.
    """
    connection = test_database_url
    embeddings = FakeEmbeddings(size=1536)
    vectorstore = PGVector(
        embeddings=embeddings,
        connection=connection,
        collection_name="integration_test_retrieval",
        use_jsonb=True,
        async_mode=True,
        create_extension=False,  # We'll manually create the extension.
        pre_delete_collection=True,  # Ensure a fresh collection.
    )

    # Manually create the pgvector extension on the test database.
    async with vectorstore._async_engine.connect() as conn:
        conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    # Create the collection (and underlying tables).
    await vectorstore.acreate_collection()

    # Insert sample documents with metadata.
    documents = [
        Document(page_content="Document 1", metadata={"user": "foo"}),
        Document(page_content="Document 2", metadata={"user": "bar"}),
        Document(page_content="Another Document", metadata={"user": "baz"}),
    ]
    await vectorstore.aadd_texts(
        texts=[doc.page_content for doc in documents],
        metadatas=[doc.metadata for doc in documents]
    )

    # a) Test the retriever interface.
    retriever = vectorstore.as_retriever()
    retrieved_docs = await retriever.ainvoke("Document")
    for doc in retrieved_docs:
        assert isinstance(doc.metadata, dict), f"[Retriever] Metadata is not a dict: {doc.metadata}"
        print(f"[Retriever] Document: {doc.page_content}, Metadata: {doc.metadata}")

    # b) Test asimilarity_search_with_score (by query string).
    scored_results = await vectorstore.asimilarity_search_with_score("Document", k=2)
    for doc, score in scored_results:
        assert isinstance(doc.metadata, dict), f"[With Score] Metadata is not a dict: {doc.metadata}"
        assert isinstance(score, float), f"[With Score] Score is not a float: {score}"
        print(f"[With Score] Document: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

    # Obtain a query embedding.
    query_embedding = await vectorstore.embeddings.aembed_query("Document")

    # c) Test asimilarity_search_by_vector (by query embedding).
    docs_by_vector = await vectorstore.asimilarity_search_by_vector(query_embedding, k=2)
    for doc in docs_by_vector:
        assert isinstance(doc.metadata, dict), f"[By Vector] Metadata is not a dict: {doc.metadata}"
        print(f"[By Vector] Document: {doc.page_content}, Metadata: {doc.metadata}")

    # d) Test asimilarity_search_with_score_by_vector (by query embedding).
    scored_docs_by_vector = await vectorstore.asimilarity_search_with_score_by_vector(query_embedding, k=2)
    for doc, score in scored_docs_by_vector:
        assert isinstance(doc.metadata, dict), f"[With Score By Vector] Metadata is not a dict: {doc.metadata}"
        assert isinstance(score, float), f"[With Score By Vector] Score is not a float: {score}"
        print(f"[With Score By Vector] Document: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

    # Clean up: Dispose the async engine to close all connections.
    await vectorstore._async_engine.dispose()
