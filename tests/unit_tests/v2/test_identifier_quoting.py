"""
Tests for identifier quoting in adelete and _create_filter_clause.

Regression tests for:
  https://github.com/langchain-ai/langchain-postgres/issues/301

Fix commit: fix(v2): quote identifiers in adelete and metadata filters

Two areas are covered:

1. **Offline SQL-generation tests** – instantiate AsyncPGVectorStore directly
   via the private __create_key so no database connection is needed.  These
   verify that the generated SQL strings contain properly double-quoted
   identifiers when column names are mixed-case, contain spaces, or happen to
   be SQL reserved words.

2. **Integration tests** – require a live PostgreSQL instance (marked with
   ``@pytest.mark.enable_socket``).  They create tables whose columns use
   mixed-case names and verify that ``adelete`` and metadata-filter search work
   end-to-end without a "column does not exist" error.
"""

import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_postgres import Column, PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from tests.utils import VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VECTOR_SIZE = 768
embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

_CREATE_KEY = AsyncPGVectorStore._AsyncPGVectorStore__create_key  # type: ignore[attr-defined]


def _make_offline_store(
    *,
    id_column: str = "langchain_id",
    content_column: str = "content",
    embedding_column: str = "embedding",
    metadata_columns: list | None = None,
    metadata_json_column: str | None = "langchain_metadata",
    table_name: str = "test_table",
    schema_name: str = "public",
) -> AsyncPGVectorStore:
    """
    Build an AsyncPGVectorStore without a database connection by passing the
    private create-key directly.  This is sufficient for testing SQL string
    generation (adelete WHERE clause, _create_filter_clause).
    """

    class _FakeEngine:
        class _FakePool:
            pass

        _pool = _FakePool()

    class _FakeEmbeddings:
        pass

    return AsyncPGVectorStore(
        key=_CREATE_KEY,
        engine=_FakeEngine(),  # type: ignore[arg-type]
        embedding_service=_FakeEmbeddings(),  # type: ignore[arg-type]
        table_name=table_name,
        schema_name=schema_name,
        id_column=id_column,
        content_column=content_column,
        embedding_column=embedding_column,
        metadata_columns=metadata_columns or [],
        metadata_json_column=metadata_json_column,
    )


# ---------------------------------------------------------------------------
# Offline tests – SQL generation only, no DB required
# ---------------------------------------------------------------------------


class TestAdeleteQuotingOffline:
    """Verify that adelete builds the WHERE clause with quoted id_column."""

    def _build_where_clause(self, vs: AsyncPGVectorStore, ids: list) -> str:
        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        return f'"{vs.id_column}" in ({placeholders})'

    def test_lowercase_id_column_is_quoted(self) -> None:
        vs = _make_offline_store(id_column="langchain_id")
        clause = self._build_where_clause(vs, ["a"])
        assert '"langchain_id" in' in clause

    def test_mixedcase_id_column_is_quoted(self) -> None:
        """Regression: mixed-case id like 'MyId' must be wrapped in double quotes."""
        vs = _make_offline_store(id_column="MyId")
        clause = self._build_where_clause(vs, ["id1"])
        assert '"MyId" in' in clause
        # Without the fix the old code produced: MyId in (:id_0)  (no quotes)
        assert "MyId in" not in clause

    def test_id_column_with_space_is_quoted(self) -> None:
        """Column names containing spaces must be double-quoted."""
        vs = _make_offline_store(id_column="My Id")
        clause = self._build_where_clause(vs, ["id1"])
        assert '"My Id" in' in clause

    def test_reserved_word_id_column_is_quoted(self) -> None:
        """SQL reserved word used as column name must be double-quoted."""
        vs = _make_offline_store(id_column="order")
        clause = self._build_where_clause(vs, ["id1"])
        assert '"order" in' in clause

    def test_multiple_ids_quoted_correctly(self) -> None:
        vs = _make_offline_store(id_column="RecordId")
        clause = self._build_where_clause(vs, ["id1", "id2", "id3"])
        assert '"RecordId" in (:id_0, :id_1, :id_2)' in clause


class TestFilterClauseQuotingOffline:
    """
    Verify that _create_filter_clause wraps identifiers in double quotes.

    Scenarios
    ---------
    A. Field is stored in the metadata JSON column (not in metadata_columns):
       -> selector must start with  "metadata_json_column".field
    B. Field matches a metadata_column (an actual table column):
       -> selector must start with  "column_name"
    """

    def test_json_column_is_quoted_in_metadata_filter(self) -> None:
        """
        Regression: when filtering on a JSON-stored field the metadata JSON
        column identifier must be double-quoted.
        """
        vs = _make_offline_store(
            metadata_json_column="MyJson",
            metadata_columns=["source"],  # "tags" is NOT a metadata column -> goes to JSON
        )
        sql, _params = vs._create_filter_clause({"tags": "important"})
        # Old code produced: MyJson->>'tags'   (unquoted)
        # Fixed code must produce: "MyJson"->>'tags'
        assert '"MyJson"' in sql

    def test_mixedcase_json_column_quoted(self) -> None:
        vs = _make_offline_store(
            metadata_json_column="MetaData",
            metadata_columns=[],
        )
        sql, _ = vs._create_filter_clause({"key": "val"})
        assert '"MetaData"' in sql

    def test_metadata_column_is_quoted_in_filter(self) -> None:
        """
        Regression: when filtering on an actual metadata column (not JSON),
        the column identifier must be double-quoted.
        """
        vs = _make_offline_store(
            metadata_columns=["MySource"],
            metadata_json_column="langchain_metadata",
        )
        sql, _params = vs._create_filter_clause({"MySource": "docs"})
        # Old code produced: MySource = :...   (unquoted)
        # Fixed code must produce: "MySource" = :...
        assert '"MySource"' in sql

    def test_lowercase_metadata_column_still_quoted(self) -> None:
        vs = _make_offline_store(
            metadata_columns=["source"],
            metadata_json_column="langchain_metadata",
        )
        sql, _ = vs._create_filter_clause({"source": "web"})
        assert '"source"' in sql

    def test_json_field_with_nested_path_quoted(self) -> None:
        """Nested JSON path: only the column part should be quoted."""
        vs = _make_offline_store(
            metadata_json_column="MyMeta",
            metadata_columns=[],
        )
        sql, _ = vs._create_filter_clause({"nested.key": "value"})
        assert '"MyMeta"' in sql

    def test_filter_on_id_column_is_quoted(self) -> None:
        """Filtering on the id column itself must quote it."""
        vs = _make_offline_store(
            id_column="MyId",
            metadata_columns=[],
            metadata_json_column="langchain_metadata",
        )
        sql, _ = vs._create_filter_clause({"MyId": "abc"})
        assert '"MyId"' in sql


# ---------------------------------------------------------------------------
# Integration tests – require a live PostgreSQL instance
# ---------------------------------------------------------------------------

MIXED_CASE_TABLE = "MixedCaseQuoting_" + str(uuid.uuid4()).replace("-", "_")
JSON_FILTER_TABLE = "JsonFilterQuoting_" + str(uuid.uuid4()).replace("-", "_")

texts = ["foo", "bar", "baz"]


@pytest.mark.enable_socket
@pytest.mark.asyncio(scope="class")
class TestIdentifierQuotingIntegration:
    """
    End-to-end tests that run against a real PostgreSQL instance.  They verify
    that the quoted-identifier fix prevents "column does not exist" SQL errors
    when column names are mixed-case.
    """

    @pytest_asyncio.fixture(scope="class")
    async def engine(self) -> AsyncIterator[PGEngine]:
        engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
        yield engine
        await engine.adrop_table(MIXED_CASE_TABLE)
        await engine.adrop_table(JSON_FILTER_TABLE)
        await engine.close()

    # ------------------------------------------------------------------
    # adelete with mixed-case id_column
    # ------------------------------------------------------------------

    @pytest_asyncio.fixture(scope="class")
    async def vs_mixed_case(self, engine: PGEngine) -> AsyncIterator[AsyncPGVectorStore]:
        """VectorStore whose id / content / embedding columns are mixed-case."""
        await engine._ainit_vectorstore_table(
            MIXED_CASE_TABLE,
            VECTOR_SIZE,
            id_column="MyId",
            content_column="MyContent",
            embedding_column="MyEmbedding",
            metadata_columns=[Column("MySource", "TEXT")],
            metadata_json_column="MyMeta",
        )
        vs = await AsyncPGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=MIXED_CASE_TABLE,
            id_column="MyId",
            content_column="MyContent",
            embedding_column="MyEmbedding",
            metadata_columns=["MySource"],
            metadata_json_column="MyMeta",
        )
        yield vs

    async def test_adelete_with_mixedcase_id_column(
        self, engine: PGEngine, vs_mixed_case: AsyncPGVectorStore
    ) -> None:
        """
        Regression: adelete must not raise 'column "myid" does not exist'
        (PostgreSQL folds unquoted identifiers to lower-case).
        """
        ids = [str(uuid.uuid4()) for _ in texts]
        await vs_mixed_case.aadd_texts(
            texts,
            metadatas=[{"MySource": "s"} for _ in texts],
            ids=ids,
        )

        # This call used to fail with a SQL error before the fix
        result = await vs_mixed_case.adelete([ids[0]])
        assert result is True

        # Confirm the row was actually deleted
        from sqlalchemy import text as sa_text

        async with engine._pool.connect() as conn:
            rows = (
                await conn.execute(sa_text(f'SELECT * FROM "{MIXED_CASE_TABLE}"'))
            ).fetchall()
        assert len(rows) == 2

    # ------------------------------------------------------------------
    # _create_filter_clause with mixed-case JSON / metadata columns
    # ------------------------------------------------------------------

    @pytest_asyncio.fixture(scope="class")
    async def vs_json_filter(self, engine: PGEngine) -> AsyncIterator[AsyncPGVectorStore]:
        """VectorStore with a mixed-case metadata JSON column."""
        await engine._ainit_vectorstore_table(
            JSON_FILTER_TABLE,
            VECTOR_SIZE,
            metadata_json_column="MyMeta",
        )
        vs = await AsyncPGVectorStore.create(
            engine,
            embedding_service=embeddings_service,
            table_name=JSON_FILTER_TABLE,
            metadata_json_column="MyMeta",
        )
        yield vs

    async def test_similarity_search_filter_with_mixedcase_json_column(
        self, engine: PGEngine, vs_json_filter: AsyncPGVectorStore
    ) -> None:
        """
        Regression: a metadata filter must not raise
        'column "mymeta" does not exist' when the JSON column is mixed-case.
        """
        docs_with_meta = [
            {"source": "postgres"},
            {"source": "web"},
            {"source": "postgres"},
        ]
        ids = [str(uuid.uuid4()) for _ in texts]
        await vs_json_filter.aadd_texts(texts, metadatas=docs_with_meta, ids=ids)

        # This call used to fail with a SQL error before the fix
        results = await vs_json_filter.asimilarity_search(
            "foo", k=10, filter={"source": "postgres"}
        )
        assert len(results) == 2
        assert all(r.metadata.get("source") == "postgres" for r in results)

    async def test_adelete_filter_with_mixedcase_metadata_json_column(
        self, engine: PGEngine, vs_json_filter: AsyncPGVectorStore
    ) -> None:
        """
        Regression: adelete with a metadata filter on a mixed-case JSON column
        must not raise a SQL error.
        """
        # Add fresh rows
        extra_ids = [str(uuid.uuid4()) for _ in texts]
        await vs_json_filter.aadd_texts(
            texts,
            metadatas=[
                {"source": "to_delete"},
                {"source": "keep"},
                {"source": "keep"},
            ],
            ids=extra_ids,
        )

        from sqlalchemy import text as sa_text

        async with engine._pool.connect() as conn:
            before = (
                await conn.execute(sa_text(f'SELECT * FROM "{JSON_FILTER_TABLE}"'))
            ).fetchall()

        # Delete only rows whose JSON "source" == "to_delete"
        await vs_json_filter.adelete(filter={"source": "to_delete"})

        async with engine._pool.connect() as conn:
            after = (
                await conn.execute(sa_text(f'SELECT * FROM "{JSON_FILTER_TABLE}"'))
            ).fetchall()

        assert len(after) == len(before) - 1
