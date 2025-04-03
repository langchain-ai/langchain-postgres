"""Index class to add vector indexes on the PGVectorStore.

Learn more about vector indexes at https://github.com/pgvector/pgvector?tab=readme-ov-file#indexing
"""

import enum
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyMixin:
    operator: str
    search_function: str
    index_function: str


class DistanceStrategy(StrategyMixin, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "<->", "l2_distance", "vector_l2_ops"
    COSINE_DISTANCE = "<=>", "cosine_distance", "vector_cosine_ops"
    INNER_PRODUCT = "<#>", "inner_product", "vector_ip_ops"


DEFAULT_DISTANCE_STRATEGY: DistanceStrategy = DistanceStrategy.COSINE_DISTANCE
DEFAULT_INDEX_NAME_SUFFIX: str = "langchainvectorindex"


def validate_identifier(identifier: str) -> None:
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier) is None:
        raise ValueError(
            f"Invalid identifier: {identifier}. Identifiers must start with a letter or underscore, and subsequent characters can be letters, digits, or underscores."
        )


@dataclass
class BaseIndex(ABC):
    """
    Abstract base class for defining vector indexes.

    Attributes:
        name (Optional[str]): A human-readable name for the index. Defaults to None.
        index_type (str): A string identifying the type of index. Defaults to "base".
        distance_strategy (DistanceStrategy): The strategy used to calculate distances
            between vectors in the index. Defaults to DistanceStrategy.COSINE_DISTANCE.
        partial_indexes (Optional[list[str]]): A list of names of partial indexes. Defaults to None.
        extension_name (Optional[str]): The name of the extension to be created for the index, if any. Defaults to None.
    """

    name: Optional[str] = None
    index_type: str = "base"
    distance_strategy: DistanceStrategy = field(
        default_factory=lambda: DistanceStrategy.COSINE_DISTANCE
    )
    partial_indexes: Optional[list[str]] = None
    extension_name: Optional[str] = None

    @abstractmethod
    def index_options(self) -> str:
        """Set index query options for vector store initialization."""
        raise NotImplementedError(
            "index_options method must be implemented by subclass"
        )

    def get_index_function(self) -> str:
        return self.distance_strategy.index_function

    def __post_init__(self) -> None:
        """Check if initialization parameters are valid.

        Raises:
            ValueError: extension_name is a valid postgreSQL identifier
        """

        if self.extension_name:
            validate_identifier(self.extension_name)
        if self.index_type:
            validate_identifier(self.index_type)


@dataclass
class ExactNearestNeighbor(BaseIndex):
    index_type: str = "exactnearestneighbor"


@dataclass
class QueryOptions(ABC):
    @abstractmethod
    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        raise NotImplementedError("to_parameter method must be implemented by subclass")

    @abstractmethod
    def to_string(self) -> str:
        """Convert index attributes to string."""
        raise NotImplementedError("to_string method must be implemented by subclass")


@dataclass
class HNSWIndex(BaseIndex):
    index_type: str = "hnsw"
    m: int = 16
    ef_construction: int = 64

    def index_options(self) -> str:
        """Set index query options for vector store initialization."""
        return f"(m = {self.m}, ef_construction = {self.ef_construction})"


@dataclass
class HNSWQueryOptions(QueryOptions):
    ef_search: int = 40

    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        return [f"hnsw.ef_search = {self.ef_search}"]

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return f"hnsw.ef_search = {self.ef_search}"


@dataclass
class IVFFlatIndex(BaseIndex):
    index_type: str = "ivfflat"
    lists: int = 100

    def index_options(self) -> str:
        """Set index query options for vector store initialization."""
        return f"(lists = {self.lists})"


@dataclass
class IVFFlatQueryOptions(QueryOptions):
    probes: int = 1

    def to_parameter(self) -> list[str]:
        """Convert index attributes to list of configurations."""
        return [f"ivfflat.probes = {self.probes}"]

    def to_string(self) -> str:
        """Convert index attributes to string."""
        warnings.warn(
            "to_string is deprecated, use to_parameter instead.",
            DeprecationWarning,
        )
        return f"ivfflat.probes = {self.probes}"
