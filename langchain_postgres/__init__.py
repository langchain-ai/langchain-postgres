from importlib import metadata

from langchain_postgres.chat_message_histories import PostgresChatMessageHistory
from langchain_postgres.translator import PGVectorTranslator
from langchain_postgres.v2.engine import Column, ColumnDict, PGEngine
from langchain_postgres.v2.vectorstores import PGVectorStore
from langchain_postgres.vectorstores import PGVector

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

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
