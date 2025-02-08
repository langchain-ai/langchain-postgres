import sqlalchemy
from langchain_postgres.vectorstores import PGVector
from tests.unit_tests.fake_embeddings import FakeEmbeddings

# --- Monkey-patch PGVector to avoid DB operations during instantiation ---
# This prevents __post_init__ (table creation, etc.) from running.
PGVector.__post_init__ = lambda self: None


# --- Fake classes to simulate database result rows ---

class FakeFragment:
    """Simulate a non-dict JSON fragment (e.g. an asyncpg Fragment) with a 'buf' attribute."""

    def __init__(self, data: bytes):
        self.buf = data


class FakeBytes:
    """Simulate an object with a decode() method that returns a JSON string."""

    def __init__(self, data: bytes):
        self.data = data

    def decode(self, encoding: str) -> str:
        return self.data.decode(encoding)


class FakeEmbeddingStore:
    """Simulate the EmbeddingStore object attached to a result row."""

    def __init__(self, id, document, cmetadata):
        self.id = id
        self.document = document
        self.cmetadata = cmetadata


class FakeResult:
    """Simulate a database result row with an EmbeddingStore attribute and a 'distance' attribute."""

    def __init__(self, embedding_store, distance=0.0):
        self.EmbeddingStore = embedding_store
        self.distance = distance


def fake_results():
    # 1. Metadata already a dict.
    result1 = FakeResult(FakeEmbeddingStore("1", "doc1", {"user": "foo"}))
    # 2. Metadata as a JSON string.
    result2 = FakeResult(FakeEmbeddingStore("2", "doc2", '{"user": "bar"}'))
    # 3. Metadata as a fake Fragment (simulate asyncpg) with a .buf attribute.
    fragment = FakeFragment(b'{"user": "baz"}')
    result3 = FakeResult(FakeEmbeddingStore("3", "doc3", fragment))
    # 4. Metadata as an object with a decode() method.
    fake_bytes = FakeBytes(b'{"user": "qux"}')
    result4 = FakeResult(FakeEmbeddingStore("4", "doc4", fake_bytes))
    # 5. Metadata of an invalid type (e.g. an integer) should result in empty dict.
    result5 = FakeResult(FakeEmbeddingStore("5", "doc5", 12345))
    return [result1, result2, result3, result4, result5]


def test_metadata_deserialization():
    """Test that PGVector correctly converts non-dict metadata to dictionaries."""
    # Monkey-patch SQLAlchemy's MetaData.create_all so that table creation is skipped.
    orig_create_all = sqlalchemy.MetaData.create_all
    sqlalchemy.MetaData.create_all = lambda self, bind=None, checkfirst=True: None

    try:
        # Create a PGVector instance.
        # Using SQLite for testing, so we disable extension creation.
        pg_vector = PGVector(
            embeddings=FakeEmbeddings(),
            connection="sqlite://",  # using SQLite for testing
            collection_name="fake_collection",
            pre_delete_collection=True,
            use_jsonb=True,
            async_mode=False,
            create_extension=False  # Prevent executing PG-specific extension creation
        )
    finally:
        # Restore the original create_all method so that other tests aren’t affected.
        sqlalchemy.MetaData.create_all = orig_create_all

    results = fake_results()
    docs_and_scores = pg_vector._results_to_docs_and_scores(results)
    docs = [doc for doc, score in docs_and_scores]

    # Check that metadata is correctly deserialized.
    assert docs[0].metadata == {"user": "foo"}
    assert docs[1].metadata == {"user": "bar"}
    assert docs[2].metadata == {"user": "baz"}
    assert docs[3].metadata == {"user": "qux"}
    # For the invalid type, we expect an empty dict.
    assert docs[4].metadata == {}


if __name__ == "__main__":
    test_metadata_deserialization()
    print("All metadata deserialization tests passed.")
