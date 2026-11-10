from importlib import metadata
from typing import TYPE_CHECKING, Any

from langchain_postgres.translator import PGVectorTranslator
from langchain_postgres.v2.engine import Column, ColumnDict, PGEngine
from langchain_postgres.v2.vectorstores import PGVectorStore
from langchain_postgres.vectorstores import PGVector

if TYPE_CHECKING:
    from langchain_postgres.chat_message_histories import (
        PostgresChatMessageHistory,
    )

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""


def __getattr__(name: str) -> Any:
    """Lazily import optional symbols that pull in heavyweight or
    LGPL-licensed transitive dependencies (e.g. `psycopg`).

    Importing `PostgresChatMessageHistory` from this package triggers
    `import psycopg`, which is LGPL-3.0 and may be undesirable for
    downstreams that only use the driver-agnostic v2 API (`PGEngine`,
    `PGVectorStore`, ...). Deferring the import until the symbol is
    actually accessed keeps `from langchain_postgres import PGEngine`
    (and similar) free of any psycopg load.
    """
    if name == "PostgresChatMessageHistory":
        from langchain_postgres.chat_message_histories import (
            PostgresChatMessageHistory,
        )

        return PostgresChatMessageHistory
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "__version__",
    "Column",
    "ColumnDict",
    "PGEngine",
    "PostgresChatMessageHistory",
    "PGVector",
    "PGVectorStore",
    "PGVectorTranslator",
]
