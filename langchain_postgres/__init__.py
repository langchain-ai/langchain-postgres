from importlib import metadata

from langchain_postgres.chat_message_histories import \
    PostgresChatMessageHistory
from langchain_postgres.engine import Column, PGEngine
from langchain_postgres.translator import PGVectorTranslator
from langchain_postgres.vectorstore import PGVectorStore
from langchain_postgres.vectorstores import PGVector

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

__all__ = [
    "__version__",
    "Column",
    "PGEngine",
    "PostgresChatMessageHistory",
    "PGVector",
    "PGVectorStore",
    "PGVectorTranslator",
]
