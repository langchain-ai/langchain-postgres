from langchain_postgres import __all__

EXPECTED_ALL = [
    "__version__",
    "Column",
    "ColumnDict",
    "PGEngine",
    "PGVector",
    "PGVectorStore",
    "PGVectorTranslator",
    "PostgresChatMessageHistory",
    "PGChatMessageHistory",
]


def test_all_imports() -> None:
    """Test that __all__ is correctly defined."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
