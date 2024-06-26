import abc
from abc import ABC
from typing import TypeVar, Iterable, TypedDict


class Content(ABC):
    id: str


class Document(Content):
    content: str
    metadata: dict


class BinaryContent(Content):
    data: Union[bytes, str]
    encoding: str  # base64, utf-8, URI, etc.
    mimetype: str  # image/png, text/plain, etc.
    metadata: dict


class MultiModalContent(Content):
    contents: List[Content]
    metadata: dict


T = TypeVar("T", bound=Content)


class IndexingResult(TypedDict):
    """An indexing result."""

    failed: Sequence[str]
    indexed: Sequence[str]


class DeleteResponse(TypedDict):
    """A response to a delete request."""

    num_deleted: NotRequired[int]
    num_failed: NotRequired[int]
    failed: NotRequired[Sequence[str]]
    deleted: NotRequired[Sequence[str]]


class Query(TypedDict):
    """Query for an item.

    This enables querying for an item using similarity search +
    standard operations on the relational data associated with the item.
    """

    query: Optional[str]  # A query string
    filters: Optional[dict]
    limit: Optional[int]
    offset: Optional[int]
    sort: Optional[dict]
    # include: Optional[dict]
    # exclude: Optional[dict]


class Indexer(Generic[T]):
    @abc.abstractmethod
    def upsert(self, data: Iterable[T], /, **kwargs: Any) -> Iterable[IndexingResult]:
        """Upsert a stream of data by id."""

    @abc.abstractmethod
    def delete_by_ids(
        self, ids: Sequence[str], /, **kwargs: Any
    ) -> DeleteResponse:  # Sequence?
        """Delete an item by id."""

    @abc.abstractmethod
    def get_by_ids(self, ids: Iterable[str]) -> Iterable[T]:
        """Get items by id."""

    # Delete and get are still part of the READ/WRITE interface
    @abc.abstractmethod
    def delete_by_query(
        self, query: Query, /, **kwargs: Any
    ) -> Iterable[DeleteResponse]:
        """Delete items by query."""
        # Careful with halloween problem

    @abc.abstractmethod
    def get_by_query(self, query: Query, /, **kwargs: Any) -> Iterable[T]:
        """Get items by query."""


class DocumentIndexer(Indexer[Document]):
    """An indexer for documents."""


class VectorStoreQuery(TypedDict):
    vector: Optional[Vector]  # A query vector
    # Add some search methods
    kind: Optional[str]  # The kind of query


Vector = List[float]


class RetrievalResult(TypedDict):
    """A retrieval result."""

    id: str
    score: float
    data: dict


class Retriever(Generic[T]):
    @abc.abstractmethod
    def query(self, query: Query) -> Iterable[T]:
        """Query for items."""


class VectorStoreRetriever(Generic[T], Retriever[T], Indexer[T]):
    @abc.abstractmethod
    def filter(self, query: dict) -> Iterable[T]:
        """Filter for items."""


class VectorStore(Generic[T]):
    @abc.abstractmethod
    def insert(self, data_stream: Iterable[T]) -> Iterable[IndexingResult]:
        """Insert a stream of data."""

    @abc.abstractmethod
    def delete(self, ids: Iterable[str]) -> DeleteResponse:
        """Delete an item by id."""

    @abc.abstractmethod
    def query(self, query: dict) -> Iterable[T]:
        """Query for items."""
