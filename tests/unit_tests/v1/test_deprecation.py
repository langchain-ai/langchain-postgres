
import warnings
from langchain_postgres.vectorstores import PGVector
from tests.unit_tests.fake_embeddings import FakeEmbeddings

def test_pgvector_deprecation_warning() -> None:
    """Test that PGVector raises a DeprecationWarning on initialization."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # We expect initialization to fail due to missing connection/DB, 
        # but the warning should be emitted *before* that.
        # Or we can mock enough to make it succeed, but a try/except block 
        # is safer if we want to avoid dependency on a running DB.
        try:
            PGVector(
                embeddings=FakeEmbeddings(),
                connection="postgresql://user:pass@localhost:5432/db",
                collection_name="test_deprecation",
                # Disable side effects that try to connect immediately
                create_extension=False, 
                # pre_delete_collection=False is default
            )
        except Exception:
            # We don't care if it fails to connect, as long as it warned first
            pass
            
        # Check for the specific deprecation warning
        assert len(w) > 0
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "PGVector is deprecated" in str(w[-1].message)
