# TODO: Remove below import when minimum supported Python version is 3.10
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .async_vectorstore import AsyncPGVectorStore
from .engine import PGEngine
from .hybrid_search_config import HybridSearchConfig
from .indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    BaseIndex,
    DistanceStrategy,
    QueryOptions,
)


class PGVectorStore(VectorStore):
    """Postgres Vector Store class"""

    __create_key = object()

    def __init__(self, key: object, engine: PGEngine, vs: AsyncPGVectorStore):
        """PGVectorStore constructor.
        Args:
            key (object): Prevent direct constructor usage.
            engine (PGEngine): Connection pool engine for managing connections to Postgres database.
            vs (AsyncPGVectorStore): The async only VectorStore implementation


        Raises:
            Exception: If called directly by user.
        """
        if key != PGVectorStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )

        self._engine = engine
        self.__vs = vs

    @classmethod
    async def create(
        cls: type[PGVectorStore],
        engine: PGEngine,
        embedding_service: Embeddings,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        ignore_metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ) -> PGVectorStore:
        """Create an PGVectorStore instance.

        Args:
            engine (PGEngine): Connection pool engine for managing connections to postgres database.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of an existing table.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            content_column (str): Column that represent a Document's page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (list[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (list[str]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.
            hybrid_search_config (HybridSearchConfig): Hybrid search configuration. Defaults to None.

        Returns:
            PGVectorStore
        """
        coro = AsyncPGVectorStore.create(
            engine,
            embedding_service,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
        )
        vs = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, vs)

    @classmethod
    def create_sync(
        cls,
        engine: PGEngine,
        embedding_service: Embeddings,
        table_name: str,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        ignore_metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ) -> PGVectorStore:
        """Create an PGVectorStore instance.

        Args:
            key (object): Prevent direct constructor usage.
            engine (PGEngine): Connection pool engine for managing connections to postgres database.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of an existing table.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            content_column (str, optional): Column that represent a Document's page_content. Defaults to "content".
            embedding_column (str, optional): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (list[str], optional): Column(s) that represent a document's metadata. Defaults to None.
            ignore_metadata_columns (Optional[list[str]]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str, optional): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str, optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy, optional): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int, optional): Number of Documents to return from search. Defaults to 4.
            fetch_k (int, optional): Number of Documents to fetch to pass to MMR algorithm. Defaults to 20.
            lambda_mult (float, optional): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (Optional[QueryOptions], optional): Index query option. Defaults to None.
            hybrid_search_config (HybridSearchConfig): Hybrid search configuration. Defaults to None.

        Returns:
            PGVectorStore
        """
        coro = AsyncPGVectorStore.create(
            engine,
            embedding_service,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
        )
        vs = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, vs)

    @property
    def embeddings(self) -> Embeddings:
        return self.__vs.embedding_service

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add data along with embeddings to the table."""
        return await self._engine._run_as_async(
            self.__vs.aadd_embeddings(texts, embeddings, metadatas, ids, **kwargs)
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed texts and add to the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        return await self._engine._run_as_async(
            self.__vs.aadd_texts(texts, metadatas, ids, **kwargs)
        )

    async def aadd_documents(
        self,
        documents: list[Document],
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed documents and add to the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        return await self._engine._run_as_async(
            self.__vs.aadd_documents(documents, ids, **kwargs)
        )

    def add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add data along with embeddings to the table."""
        return self._engine._run_as_sync(
            self.__vs.aadd_embeddings(texts, embeddings, metadatas, ids, **kwargs)
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed texts and add to the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        return self._engine._run_as_sync(
            self.__vs.aadd_texts(texts, metadatas, ids, **kwargs)
        )

    def add_documents(
        self,
        documents: list[Document],
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed documents and add to the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        return self._engine._run_as_sync(
            self.__vs.aadd_documents(documents, ids, **kwargs)
        )

    async def adelete(
        self,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete records from the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        return await self._engine._run_as_async(self.__vs.adelete(ids, **kwargs))

    def delete(
        self,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete records from the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        return self._engine._run_as_sync(self.__vs.adelete(ids, **kwargs))

    @classmethod
    async def afrom_texts(  # type: ignore[override]
        cls: type[PGVectorStore],
        texts: list[str],
        embedding: Embeddings,
        engine: PGEngine,
        table_name: str,
        schema_name: str = "public",
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        ignore_metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
        **kwargs: Any,
    ) -> PGVectorStore:
        """Create an PGVectorStore instance from texts.

        Args:
            texts (list[str]): Texts to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (PGEngine): Connection pool engine for managing connections to postgres database.
            table_name (str): Name of an existing table.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            metadatas (Optional[list[dict]], optional): List of metadatas to add to table records. Defaults to None.
            ids: (Optional[list]): List of IDs to add to table records. Defaults to None.
            content_column (str, optional): Column that represent a Document's page_content. Defaults to "content".
            embedding_column (str, optional): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (list[str], optional): Column(s) that represent a document's metadata. Defaults to an empty list.
            ignore_metadata_columns (Optional[list[str]], optional): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str, optional): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str, optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.
            hybrid_search_config (HybridSearchConfig): Hybrid search configuration. Defaults to None.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.

        Returns:
            PGVectorStore
        """
        vs = await cls.create(
            engine,
            embedding,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
        )
        await vs.aadd_texts(texts, metadatas=metadatas, ids=ids)
        return vs

    @classmethod
    async def afrom_documents(  # type: ignore[override]
        cls: type[PGVectorStore],
        documents: list[Document],
        embedding: Embeddings,
        engine: PGEngine,
        table_name: str,
        schema_name: str = "public",
        ids: Optional[list] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        ignore_metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
        **kwargs: Any,
    ) -> PGVectorStore:
        """Create an PGVectorStore instance from documents.

        Args:
            documents (list[Document]): Documents to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (PGEngine): Connection pool engine for managing connections to postgres database.
            table_name (str): Name of an existing table.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            ids: (Optional[list]): List of IDs to add to table records. Defaults to None.
            content_column (str, optional): Column that represent a Document's page_content. Defaults to "content".
            embedding_column (str, optional): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (list[str], optional): Column(s) that represent a document's metadata. Defaults to an empty list.
            ignore_metadata_columns (Optional[list[str]], optional): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str, optional): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str, optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.
            hybrid_search_config (HybridSearchConfig): Hybrid search configuration. Defaults to None.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.

        Returns:
            PGVectorStore
        """

        vs = await cls.create(
            engine,
            embedding,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
        )
        await vs.aadd_documents(documents, ids=ids)
        return vs

    @classmethod
    def from_texts(  # type: ignore[override]
        cls: type[PGVectorStore],
        texts: list[str],
        embedding: Embeddings,
        engine: PGEngine,
        table_name: str,
        schema_name: str = "public",
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        ignore_metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
        **kwargs: Any,
    ) -> PGVectorStore:
        """Create an PGVectorStore instance from texts.

        Args:
            texts (list[str]): Texts to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (PGEngine): Connection pool engine for managing connections to postgres database.
            table_name (str): Name of an existing table.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            metadatas (Optional[list[dict]], optional): List of metadatas to add to table records. Defaults to None.
            ids: (Optional[list]): List of IDs to add to table records. Defaults to None.
            content_column (str, optional): Column that represent a Document's page_content. Defaults to "content".
            embedding_column (str, optional): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (list[str], optional): Column(s) that represent a document's metadata. Defaults to empty list.
            ignore_metadata_columns (Optional[list[str]], optional): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str, optional): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str, optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.
            hybrid_search_config (HybridSearchConfig): Hybrid search configuration. Defaults to None.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.

        Returns:
            PGVectorStore
        """
        vs = cls.create_sync(
            engine,
            embedding,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
            **kwargs,
        )
        vs.add_texts(texts, metadatas=metadatas, ids=ids)
        return vs

    @classmethod
    def from_documents(  # type: ignore[override]
        cls: type[PGVectorStore],
        documents: list[Document],
        embedding: Embeddings,
        engine: PGEngine,
        table_name: str,
        schema_name: str = "public",
        ids: Optional[list] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        ignore_metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
        **kwargs: Any,
    ) -> PGVectorStore:
        """Create an PGVectorStore instance from documents.

        Args:
            documents (list[Document]): Documents to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (PGEngine): Connection pool engine for managing connections to postgres database.
            table_name (str): Name of an existing table.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            ids: (Optional[list]): List of IDs to add to table records. Defaults to None.
            content_column (str, optional): Column that represent a Document's page_content. Defaults to "content".
            embedding_column (str, optional): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (list[str], optional): Column(s) that represent a document's metadata. Defaults to an empty list.
            ignore_metadata_columns (Optional[list[str]], optional): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str, optional): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str, optional): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.
            hybrid_search_config (HybridSearchConfig): Hybrid search configuration. Defaults to None.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.

        Returns:
            PGVectorStore
        """
        vs = cls.create_sync(
            engine,
            embedding,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            metadata_json_column=metadata_json_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
            **kwargs,
        )
        vs.add_documents(documents, ids=ids)
        return vs

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by similarity search on query."""
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search(query, k, filter, **kwargs)
        )

    async def asimilarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by similarity search on query."""
        return await self._engine._run_as_async(
            self.__vs.asimilarity_search(query, k, filter, **kwargs)
        )

    # Required for (a)similarity_search_with_relevance_scores
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select a relevance function based on distance strategy."""
        # Calculate distance strategy provided in vectorstore constructor
        if self.__vs.distance_strategy == DistanceStrategy.COSINE_DISTANCE:
            return self._cosine_relevance_score_fn
        if self.__vs.distance_strategy == DistanceStrategy.INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        elif self.__vs.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and distance scores selected by similarity search on query."""
        return await self._engine._run_as_async(
            self.__vs.asimilarity_search_with_score(query, k, filter, **kwargs)
        )

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by vector similarity search."""
        return await self._engine._run_as_async(
            self.__vs.asimilarity_search_by_vector(embedding, k, filter, **kwargs)
        )

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and distance scores selected by vector similarity search."""
        return await self._engine._run_as_async(
            self.__vs.asimilarity_search_with_score_by_vector(
                embedding, k, filter, **kwargs
            )
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return await self._engine._run_as_async(
            self.__vs.amax_marginal_relevance_search(
                query, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return await self._engine._run_as_async(
            self.__vs.amax_marginal_relevance_search_by_vector(
                embedding, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and distance scores selected using the maximal marginal relevance."""
        return await self._engine._run_as_async(
            self.__vs.amax_marginal_relevance_search_with_score_by_vector(
                embedding, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and distance scores selected by similarity search on query."""
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search_with_score(query, k, filter, **kwargs)
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by vector similarity search."""
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search_by_vector(embedding, k, filter, **kwargs)
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and distance scores selected by similarity search on vector."""
        return self._engine._run_as_sync(
            self.__vs.asimilarity_search_with_score_by_vector(
                embedding, k, filter, **kwargs
            )
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return self._engine._run_as_sync(
            self.__vs.amax_marginal_relevance_search(
                query, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return self._engine._run_as_sync(
            self.__vs.amax_marginal_relevance_search_by_vector(
                embedding, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and distance scores selected using the maximal marginal relevance."""
        return self._engine._run_as_sync(
            self.__vs.amax_marginal_relevance_search_with_score_by_vector(
                embedding, k, fetch_k, lambda_mult, filter, **kwargs
            )
        )

    async def aapply_vector_index(
        self,
        index: BaseIndex,
        name: Optional[str] = None,
        concurrently: bool = False,
    ) -> None:
        """Create an index on the vector store table."""
        return await self._engine._run_as_async(
            self.__vs.aapply_vector_index(index, name, concurrently=concurrently)
        )

    def apply_vector_index(
        self,
        index: BaseIndex,
        name: Optional[str] = None,
        concurrently: bool = False,
    ) -> None:
        """Create an index on the vector store table."""
        return self._engine._run_as_sync(
            self.__vs.aapply_vector_index(index, name, concurrently=concurrently)
        )

    async def areindex(self, index_name: Optional[str] = None) -> None:
        """Re-index the vector store table."""
        return await self._engine._run_as_async(self.__vs.areindex(index_name))

    def reindex(self, index_name: Optional[str] = None) -> None:
        """Re-index the vector store table."""
        return self._engine._run_as_sync(self.__vs.areindex(index_name))

    async def adrop_vector_index(
        self,
        index_name: Optional[str] = None,
    ) -> None:
        """Drop the vector index."""
        return await self._engine._run_as_async(
            self.__vs.adrop_vector_index(index_name)
        )

    def drop_vector_index(
        self,
        index_name: Optional[str] = None,
    ) -> None:
        """Drop the vector index."""
        return self._engine._run_as_sync(self.__vs.adrop_vector_index(index_name))

    async def ais_valid_index(
        self,
        index_name: Optional[str] = None,
    ) -> bool:
        """Check if index exists in the table."""
        return await self._engine._run_as_async(self.__vs.is_valid_index(index_name))

    def is_valid_index(
        self,
        index_name: Optional[str] = None,
    ) -> bool:
        """Check if index exists in the table."""
        return self._engine._run_as_sync(self.__vs.is_valid_index(index_name))

    async def aget_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Get documents by ids."""
        return await self._engine._run_as_async(self.__vs.aget_by_ids(ids=ids))

    def get_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Get documents by ids."""
        return self._engine._run_as_sync(self.__vs.aget_by_ids(ids=ids))

    def get_table_name(self) -> str:
        return self.__vs.table_name
