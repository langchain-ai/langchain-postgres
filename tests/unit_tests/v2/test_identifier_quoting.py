"""Column identifiers in hand-built SQL must be quoted.

Postgres folds an unquoted identifier to lowercase, so a mixed-case
``id_column`` or ``metadata_json_column`` produced `column does not exist`
even though `aadd_embeddings` had created it with the configured casing.
"""

from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore


def _store() -> AsyncPGVectorStore:
    """A store with mixed-case identifiers, no engine needed."""
    store = AsyncPGVectorStore.__new__(AsyncPGVectorStore)
    store.metadata_json_column = "My Json"
    store.metadata_columns = ["MyCol"]
    store.id_column = "MyId"
    store.content_column = "content"
    store.embedding_column = "embedding"
    return store


def test_json_path_quotes_the_metadata_column() -> None:
    clause, _ = _store()._create_filter_clause({"some_key": "v"})

    assert clause.startswith('"My Json"->>')


def test_nested_json_path_quotes_only_the_column() -> None:
    clause, _ = _store()._create_filter_clause({"a.b": "v"})

    assert clause.startswith("\"My Json\"->'a'->>'b'")


def test_typed_json_path_quotes_the_column() -> None:
    clause, _ = _store()._create_filter_clause({"n": 5})

    assert clause.startswith("(\"My Json\"->>'n')::INTEGER")


def test_bare_metadata_column_is_quoted() -> None:
    clause, _ = _store()._create_filter_clause({"MyCol": "v"})

    assert clause.startswith('"MyCol" ')


def test_id_column_is_quoted() -> None:
    clause, _ = _store()._create_filter_clause({"MyId": "x"})

    assert clause.startswith('"MyId" ')
