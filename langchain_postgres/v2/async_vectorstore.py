# TODO: Remove below import when minimum supported Python version is 3.10
from __future__ import annotations

import copy
import json
import uuid
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, utils
from sqlalchemy import RowMapping, text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import PGEngine
from .hybrid_search_config import HybridSearchConfig
from .indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    DEFAULT_INDEX_NAME_SUFFIX,
    BaseIndex,
    DistanceStrategy,
    ExactNearestNeighbor,
    QueryOptions,
)

COMPARISONS_TO_NATIVE = {
    "$eq": "=",
    "$ne": "!=",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
}

SPECIAL_CASED_OPERATORS = {
    "$in",
    "$nin",
    "$between",
    "$exists",
}

TEXT_OPERATORS = {
    "$like",
    "$ilike",
}

LOGICAL_OPERATORS = {"$and", "$or", "$not"}

SUPPORTED_OPERATORS = (
    set(COMPARISONS_TO_NATIVE)
    .union(TEXT_OPERATORS)
    .union(LOGICAL_OPERATORS)
    .union(SPECIAL_CASED_OPERATORS)
)


class AsyncPGVectorStore(VectorStore):
    """Postgres Vector Store class"""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AsyncEngine,
        embedding_service: Embeddings,
        table_name: str,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        index_query_options: Optional[QueryOptions] = None,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ):
        """AsyncPGVectorStore constructor.
        Args:
            key (object): Prevent direct constructor usage.
            engine (PGEngine): Connection pool engine for managing connections to postgres database.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of the existing table or the table to be created.
            schema_name (str, optional): Name of the database schema. Defaults to "public".
            content_column (str): Column that represent a Document's page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (list[str]): Column(s) that represent a document's metadata.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            k (int): Number of Documents to return from search. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult (float): Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.
            index_query_options (QueryOptions): Index query option.
            hybrid_search_config (HybridSearchConfig): Hybrid search configuration. Defaults to None.


        Raises:
            Exception: If called directly by user.
        """
        if key != AsyncPGVectorStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )

        self.engine = engine
        self.embedding_service = embedding_service
        self.table_name = table_name
        self.schema_name = schema_name
        self.content_column = content_column
        self.embedding_column = embedding_column
        self.metadata_columns = metadata_columns if metadata_columns is not None else []
        self.id_column = id_column
        self.metadata_json_column = metadata_json_column
        self.distance_strategy = distance_strategy
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.index_query_options = index_query_options
        self.hybrid_search_config = hybrid_search_config

    @classmethod
    async def create(
        cls: type[AsyncPGVectorStore],
        engine: PGEngine,
        embedding_service: Embeddings,
        table_name: str,
        *,
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
    ) -> AsyncPGVectorStore:
        """Create an AsyncPGVectorStore instance.

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
            AsyncPGVectorStore
        """

        if metadata_columns is None:
            metadata_columns = []

        if metadata_columns and ignore_metadata_columns:
            raise ValueError(
                "Can not use both metadata_columns and ignore_metadata_columns."
            )
        # Get field type information
        stmt = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = :table_name AND table_schema = :schema_name"
        async with engine._pool.connect() as conn:
            result = await conn.execute(
                text(stmt),
                {"table_name": table_name, "schema_name": schema_name},
            )
            result_map = result.mappings()
            results = result_map.fetchall()
        columns = {}
        for field in results:
            columns[field["column_name"]] = field["data_type"]

        # Check columns
        if id_column not in columns:
            raise ValueError(f"Id column, {id_column}, does not exist.")
        if content_column not in columns:
            raise ValueError(f"Content column, {content_column}, does not exist.")
        content_type = columns[content_column]
        if content_type != "text" and "char" not in content_type:
            raise ValueError(
                f"Content column, {content_column}, is type, {content_type}. It must be a type of character string."
            )
        if hybrid_search_config:
            tsv_column_name = (
                hybrid_search_config.tsv_column
                if hybrid_search_config.tsv_column
                else content_column + "_tsv"
            )
            if tsv_column_name not in columns or columns[tsv_column_name] != "tsvector":
                # mark tsv_column as empty because there is no TSV column in table
                hybrid_search_config.tsv_column = ""
        if embedding_column not in columns:
            raise ValueError(f"Embedding column, {embedding_column}, does not exist.")
        if columns[embedding_column] != "USER-DEFINED":
            raise ValueError(
                f"Embedding column, {embedding_column}, is not type Vector."
            )

        metadata_json_column = (
            None if metadata_json_column not in columns else metadata_json_column
        )

        # If using metadata_columns check to make sure column exists
        for column in metadata_columns:
            if column not in columns:
                raise ValueError(f"Metadata column, {column}, does not exist.")

        # If using ignore_metadata_columns, filter out known columns and set known metadata columns
        all_columns = columns
        if ignore_metadata_columns:
            for column in ignore_metadata_columns:
                del all_columns[column]

            del all_columns[id_column]
            del all_columns[content_column]
            del all_columns[embedding_column]
            metadata_columns = [k for k in all_columns.keys()]

        return cls(
            cls.__create_key,
            engine._pool,
            embedding_service,
            table_name,
            schema_name=schema_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            id_column=id_column,
            metadata_json_column=metadata_json_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
        )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_service

    async def aadd_embeddings(
        self,
        texts: Iterable[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add data along with embeddings to the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            # This is done to fill in any missing ids
            ids = [id if id is not None else str(uuid.uuid4()) for id in ids]
        if not metadatas:
            metadatas = [{} for _ in texts]

        # Check for inline embedding capability
        inline_embed_func = getattr(self.embedding_service, "embed_query_inline", None)
        can_inline_embed = callable(inline_embed_func)
        # Insert embeddings
        for id, content, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            metadata_col_names = (
                ", " + ", ".join(f'"{col}"' for col in self.metadata_columns)
                if len(self.metadata_columns) > 0
                else ""
            )
            hybrid_search_column = (
                f', "{self.hybrid_search_config.tsv_column}"'
                if self.hybrid_search_config and self.hybrid_search_config.tsv_column
                else ""
            )
            insert_stmt = f'INSERT INTO "{self.schema_name}"."{self.table_name}"("{self.id_column}", "{self.content_column}", "{self.embedding_column}"{hybrid_search_column}{metadata_col_names}'
            values = {
                "langchain_id": id,
                "content": content,
                "embedding": str([float(dimension) for dimension in embedding]),
            }
            values_stmt = "VALUES (:langchain_id, :content, :embedding"

            if not embedding and can_inline_embed:
                values_stmt = f"VALUES (:langchain_id, :content, {self.embedding_service.embed_query_inline(content)}"  # type: ignore

            if self.hybrid_search_config and self.hybrid_search_config.tsv_column:
                lang = (
                    f"'{self.hybrid_search_config.tsv_lang}',"
                    if self.hybrid_search_config.tsv_lang
                    else ""
                )
                values_stmt += f", to_tsvector({lang} :tsv_content)"
                values["tsv_content"] = content
            # Add metadata
            extra = copy.deepcopy(metadata)
            for metadata_column in self.metadata_columns:
                if metadata_column in metadata:
                    values_stmt += f", :{metadata_column}"
                    values[metadata_column] = metadata[metadata_column]
                    del extra[metadata_column]
                else:
                    values_stmt += ",null"

            # Add JSON column and/or close statement
            insert_stmt += (
                f""", "{self.metadata_json_column}")"""
                if self.metadata_json_column
                else ")"
            )
            if self.metadata_json_column:
                values_stmt += ", :extra)"
                values["extra"] = json.dumps(extra)
            else:
                values_stmt += ")"

            upsert_stmt = f' ON CONFLICT ("{self.id_column}") DO UPDATE SET "{self.content_column}" = EXCLUDED."{self.content_column}", "{self.embedding_column}" = EXCLUDED."{self.embedding_column}"'

            if self.hybrid_search_config and self.hybrid_search_config.tsv_column:
                upsert_stmt += f', "{self.hybrid_search_config.tsv_column}" = EXCLUDED."{self.hybrid_search_config.tsv_column}"'

            if self.metadata_json_column:
                upsert_stmt += f', "{self.metadata_json_column}" = EXCLUDED."{self.metadata_json_column}"'

            for column in self.metadata_columns:
                upsert_stmt += f', "{column}" = EXCLUDED."{column}"'

            upsert_stmt += ";"

            query = insert_stmt + values_stmt + upsert_stmt
            async with self.engine.connect() as conn:
                await conn.execute(text(query), values)
                await conn.commit()

        return ids

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
        # Check for inline embedding query
        inline_embed_func = getattr(self.embedding_service, "embed_query_inline", None)
        if callable(inline_embed_func):
            embeddings: list[list[float]] = [[] for _ in list(texts)]
        else:
            embeddings = await self.embedding_service.aembed_documents(list(texts))

        ids = await self.aadd_embeddings(
            texts, embeddings, metadatas=metadatas, ids=ids, **kwargs
        )
        return ids

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
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        if not ids:
            ids = [doc.id for doc in documents]
        ids = await self.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return ids

    async def adelete(
        self,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete records from the table.

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.
        """
        if not ids:
            return False

        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        param_dict = {f"id_{i}": id for i, id in enumerate(ids)}
        query = f'DELETE FROM "{self.schema_name}"."{self.table_name}" WHERE {self.id_column} in ({placeholders})'
        async with self.engine.connect() as conn:
            await conn.execute(text(query), param_dict)
            await conn.commit()
        return True

    @classmethod
    async def afrom_texts(  # type: ignore[override]
        cls: type[AsyncPGVectorStore],
        texts: list[str],
        embedding: Embeddings,
        engine: PGEngine,
        table_name: str,
        *,
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
    ) -> AsyncPGVectorStore:
        """Create an AsyncPGVectorStore instance from texts.

        Args:
            texts (list[str]): Texts to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (PGEngine): Connection pool engine for managing connections to postgres database.
            table_name (str): Name of an existing table.
            metadatas (Optional[list[dict]]): List of metadatas to add to table records.
            ids: (Optional[list[str]]): List of IDs to add to table records.
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

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.

        Returns:
            AsyncPGVectorStore
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
            id_column=id_column,
            metadata_json_column=metadata_json_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
        )
        await vs.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return vs

    @classmethod
    async def afrom_documents(  # type: ignore[override]
        cls: type[AsyncPGVectorStore],
        documents: list[Document],
        embedding: Embeddings,
        engine: PGEngine,
        table_name: str,
        *,
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
    ) -> AsyncPGVectorStore:
        """Create an AsyncPGVectorStore instance from documents.

        Args:
            documents (list[Document]): Documents to add to the vector store.
            embedding (Embeddings): Text embedding model to use.
            engine (PGEngine): Connection pool engine for managing connections to postgres database.
            table_name (str): Name of an existing table.
            metadatas (Optional[list[dict]]): List of metadatas to add to table records.
            ids: (Optional[list[str]]): List of IDs to add to table records.
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

        Raises:
            :class:`InvalidTextRepresentationError <asyncpg.exceptions.InvalidTextRepresentationError>`: if the `ids` data type does not match that of the `id_column`.

        Returns:
            AsyncPGVectorStore
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
            id_column=id_column,
            metadata_json_column=metadata_json_column,
            distance_strategy=distance_strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            index_query_options=index_query_options,
            hybrid_search_config=hybrid_search_config,
        )
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        await vs.aadd_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return vs

    async def __query_collection(
        self,
        embedding: list[float],
        *,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> Sequence[RowMapping]:
        """
        Perform similarity search (or hybrid search) query on database.
        Queries might be slow if the hybrid search column does not exist.
        For best hybrid search performance, consider creating a TSV column
        and adding GIN index.
        """
        if not k:
            k = (
                max(
                    self.k,
                    self.hybrid_search_config.primary_top_k,
                    self.hybrid_search_config.secondary_top_k,
                )
                if self.hybrid_search_config
                else self.k
            )
        operator = self.distance_strategy.operator
        search_function = self.distance_strategy.search_function

        columns = [
            self.id_column,
            self.content_column,
            self.embedding_column,
        ] + self.metadata_columns
        if self.metadata_json_column:
            columns.append(self.metadata_json_column)

        column_names = ", ".join(f'"{col}"' for col in columns)

        safe_filter = None
        filter_dict = None
        if filter and isinstance(filter, dict):
            safe_filter, filter_dict = self._create_filter_clause(filter)

        inline_embed_func = getattr(self.embedding_service, "embed_query_inline", None)
        if not embedding and callable(inline_embed_func) and "query" in kwargs:
            query_embedding = self.embedding_service.embed_query_inline(kwargs["query"])  # type: ignore
            embedding_data_string = f"{query_embedding}"
        else:
            query_embedding = f"{[float(dimension) for dimension in embedding]}"
            embedding_data_string = ":query_embedding"
        where_filters = f"WHERE {safe_filter}" if safe_filter else ""
        dense_query_stmt = f"""SELECT {column_names}, {search_function}("{self.embedding_column}", {embedding_data_string}) as distance
        FROM "{self.schema_name}"."{self.table_name}" {where_filters} ORDER BY "{self.embedding_column}" {operator} {embedding_data_string} LIMIT :k;
        """
        param_dict = {"query_embedding": query_embedding, "k": k}
        if filter_dict:
            param_dict.update(filter_dict)
        if self.index_query_options:
            async with self.engine.connect() as conn:
                # Set each query option individually
                for query_option in self.index_query_options.to_parameter():
                    query_options_stmt = f"SET LOCAL {query_option};"
                    await conn.execute(text(query_options_stmt))
                result = await conn.execute(text(dense_query_stmt), param_dict)
                result_map = result.mappings()
                dense_results = result_map.fetchall()
        else:
            async with self.engine.connect() as conn:
                result = await conn.execute(text(dense_query_stmt), param_dict)
                result_map = result.mappings()
                dense_results = result_map.fetchall()

        hybrid_search_config = kwargs.get(
            "hybrid_search_config", self.hybrid_search_config
        )
        fts_query = (
            hybrid_search_config.fts_query
            if hybrid_search_config and hybrid_search_config.fts_query
            else kwargs.get("fts_query", "")
        )
        if hybrid_search_config and fts_query:
            hybrid_search_config.fusion_function_parameters["fetch_top_k"] = k
            # do the sparse query
            lang = (
                f"'{hybrid_search_config.tsv_lang}',"
                if hybrid_search_config.tsv_lang
                else ""
            )
            query_tsv = f"plainto_tsquery({lang} :fts_query)"
            param_dict["fts_query"] = fts_query
            if hybrid_search_config.tsv_column:
                content_tsv = f'"{hybrid_search_config.tsv_column}"'
            else:
                content_tsv = f'to_tsvector({lang} "{self.content_column}")'
            and_filters = f"AND ({safe_filter})" if safe_filter else ""
            sparse_query_stmt = f'SELECT {column_names}, ts_rank_cd({content_tsv}, {query_tsv}) as distance FROM "{self.schema_name}"."{self.table_name}" WHERE {content_tsv} @@ {query_tsv} {and_filters}  ORDER BY distance desc LIMIT {hybrid_search_config.secondary_top_k};'
            async with self.engine.connect() as conn:
                result = await conn.execute(text(sparse_query_stmt), param_dict)
                result_map = result.mappings()
                sparse_results = result_map.fetchall()

            combined_results = hybrid_search_config.fusion_function(
                dense_results,
                sparse_results,
                **hybrid_search_config.fusion_function_parameters,
            )
            return combined_results
        return dense_results

    async def asimilarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by similarity search on query."""
        inline_embed_func = getattr(self.embedding_service, "embed_query_inline", None)
        embedding = (
            []
            if callable(inline_embed_func)
            else await self.embedding_service.aembed_query(text=query)
        )
        kwargs["query"] = query

        # add fts_query to hybrid_search_config
        hybrid_search_config = kwargs.get(
            "hybrid_search_config", self.hybrid_search_config
        )
        if hybrid_search_config and not hybrid_search_config.fts_query:
            hybrid_search_config.fts_query = query
            kwargs["hybrid_search_config"] = hybrid_search_config

        return await self.asimilarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select a relevance function based on distance strategy."""
        # Calculate distance strategy provided in
        # vectorstore constructor
        if self.distance_strategy == DistanceStrategy.COSINE_DISTANCE:
            return self._cosine_relevance_score_fn
        if self.distance_strategy == DistanceStrategy.INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and distance scores selected by similarity search on query."""
        inline_embed_func = getattr(self.embedding_service, "embed_query_inline", None)
        embedding = (
            []
            if callable(inline_embed_func)
            else await self.embedding_service.aembed_query(text=query)
        )
        kwargs["query"] = query

        # add fts_query to hybrid_search_config
        hybrid_search_config = kwargs.get(
            "hybrid_search_config", self.hybrid_search_config
        )
        if hybrid_search_config and not hybrid_search_config.fts_query:
            hybrid_search_config.fts_query = query
            kwargs["hybrid_search_config"] = hybrid_search_config

        docs = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return docs

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected by vector similarity search."""
        docs_and_scores = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )

        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and distance scores selected by vector similarity search."""
        results = await self.__query_collection(
            embedding=embedding, k=k, filter=filter, **kwargs
        )

        documents_with_scores = []
        for row in results:
            metadata = (
                row[self.metadata_json_column]
                if self.metadata_json_column and row[self.metadata_json_column]
                else {}
            )
            for col in self.metadata_columns:
                metadata[col] = row[col]
            documents_with_scores.append(
                (
                    Document(
                        page_content=row[self.content_column],
                        metadata=metadata,
                        id=str(row[self.id_column]),
                    ),
                    row["distance"],
                )
            )

        return documents_with_scores

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
        embedding = await self.embedding_service.aembed_query(text=query)

        return await self.amax_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
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
        docs_and_scores = (
            await self.amax_marginal_relevance_search_with_score_by_vector(
                embedding,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
                **kwargs,
            )
        )

        return [result[0] for result in docs_and_scores]

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
        results = await self.__query_collection(
            embedding=embedding, k=fetch_k, filter=filter, **kwargs
        )

        k = k if k else self.k
        fetch_k = fetch_k if fetch_k else self.fetch_k
        lambda_mult = lambda_mult if lambda_mult else self.lambda_mult
        embedding_list = [json.loads(row[self.embedding_column]) for row in results]
        mmr_selected = utils.maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        documents_with_scores = []
        for row in results:
            metadata = (
                row[self.metadata_json_column]
                if self.metadata_json_column and row[self.metadata_json_column]
                else {}
            )
            for col in self.metadata_columns:
                metadata[col] = row[col]
            documents_with_scores.append(
                (
                    Document(
                        page_content=row[self.content_column],
                        metadata=metadata,
                        id=str(row[self.id_column]),
                    ),
                    row["distance"],
                )
            )

        return [r for i, r in enumerate(documents_with_scores) if i in mmr_selected]

    async def aapply_hybrid_search_index(
        self,
        concurrently: bool = False,
    ) -> None:
        """Creates a TSV index in the vector store table if possible."""
        if (
            not self.hybrid_search_config
            or not self.hybrid_search_config.index_type
            or not self.hybrid_search_config.index_name
        ):
            # no index needs to be created
            raise ValueError("Hybrid Search Config cannot create index.")

        lang = (
            f"'{self.hybrid_search_config.tsv_lang}',"
            if self.hybrid_search_config.tsv_lang
            else ""
        )
        tsv_column_name = (
            self.hybrid_search_config.tsv_column
            if self.hybrid_search_config.tsv_column
            else f"to_tsvector({lang} {self.content_column})"
        )
        tsv_index_query = f'CREATE INDEX {"CONCURRENTLY" if concurrently else ""} {self.hybrid_search_config.index_name} ON "{self.schema_name}"."{self.table_name}" USING {self.hybrid_search_config.index_type}({tsv_column_name});'
        if concurrently:
            async with self.engine.connect() as conn:
                autocommit_conn = await conn.execution_options(
                    isolation_level="AUTOCOMMIT"
                )
                await autocommit_conn.execute(text(tsv_index_query))
        else:
            async with self.engine.connect() as conn:
                await conn.execute(text(tsv_index_query))
                await conn.commit()

    async def aapply_vector_index(
        self,
        index: BaseIndex,
        name: Optional[str] = None,
        *,
        concurrently: bool = False,
    ) -> None:
        """Create index in the vector store table."""
        if isinstance(index, ExactNearestNeighbor):
            await self.adrop_vector_index()
            return

        # if extension name is mentioned, create the extension
        if index.extension_name:
            async with self.engine.connect() as conn:
                await conn.execute(
                    text(f"CREATE EXTENSION IF NOT EXISTS {index.extension_name}")
                )
                await conn.commit()
        function = index.get_index_function()

        filter = f"WHERE ({index.partial_indexes})" if index.partial_indexes else ""
        params = "WITH " + index.index_options()
        if name is None:
            if index.name is None:
                index.name = self.table_name + DEFAULT_INDEX_NAME_SUFFIX
            name = index.name
        stmt = f'CREATE INDEX {"CONCURRENTLY" if concurrently else ""} "{name}" ON "{self.schema_name}"."{self.table_name}" USING {index.index_type} ({self.embedding_column} {function}) {params} {filter};'

        if concurrently:
            async with self.engine.connect() as conn:
                autocommit_conn = await conn.execution_options(
                    isolation_level="AUTOCOMMIT"
                )
                await autocommit_conn.execute(text(stmt))
        else:
            async with self.engine.connect() as conn:
                await conn.execute(text(stmt))
                await conn.commit()

    async def areindex(self, index_name: Optional[str] = None) -> None:
        """Re-index the vector store table."""
        index_name = index_name or self.table_name + DEFAULT_INDEX_NAME_SUFFIX
        query = f'REINDEX INDEX "{index_name}";'
        async with self.engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def adrop_vector_index(
        self,
        index_name: Optional[str] = None,
    ) -> None:
        """Drop the vector index."""
        index_name = index_name or self.table_name + DEFAULT_INDEX_NAME_SUFFIX
        query = f'DROP INDEX IF EXISTS "{index_name}";'
        async with self.engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def is_valid_index(
        self,
        index_name: Optional[str] = None,
    ) -> bool:
        """Check if index exists in the table."""
        index_name = index_name or self.table_name + DEFAULT_INDEX_NAME_SUFFIX
        query = """
        SELECT tablename, indexname
        FROM pg_indexes
        WHERE tablename = :table_name AND schemaname = :schema_name AND indexname = :index_name;
        """
        param_dict = {
            "table_name": self.table_name,
            "schema_name": self.schema_name,
            "index_name": index_name,
        }
        async with self.engine.connect() as conn:
            result = await conn.execute(text(query), param_dict)
            result_map = result.mappings()
            results = result_map.fetchall()
        return bool(len(results) == 1)

    async def aget_by_ids(self, ids: Sequence[str]) -> list[Document]:
        """Get documents by ids."""

        columns = self.metadata_columns + [
            self.id_column,
            self.content_column,
        ]
        if self.metadata_json_column:
            columns.append(self.metadata_json_column)

        column_names = ", ".join(f'"{col}"' for col in columns)

        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        param_dict = {f"id_{i}": id for i, id in enumerate(ids)}

        query = f'SELECT {column_names} FROM "{self.schema_name}"."{self.table_name}" WHERE "{self.id_column}" IN ({placeholders});'

        async with self.engine.connect() as conn:
            result = await conn.execute(text(query), param_dict)
            result_map = result.mappings()
            results = result_map.fetchall()

        documents = []
        for row in results:
            metadata = (
                row[self.metadata_json_column]
                if self.metadata_json_column and row[self.metadata_json_column]
                else {}
            )
            for col in self.metadata_columns:
                metadata[col] = row[col]
            documents.append(
                (
                    Document(
                        page_content=row[self.content_column],
                        metadata=metadata,
                        id=str(row[self.id_column]),
                    )
                )
            )

        return documents

    def _handle_field_filter(
        self,
        *,
        field: str,
        value: Any,
    ) -> tuple[str, dict]:
        """Create a filter for a specific field.

        Args:
            field: name of field
            value: value to filter
                If provided as is then this will be an equality filter
                If provided as a dictionary then this will be a filter, the key
                will be the operator and the value will be the value to filter by

        Returns:
            sql where query as a string
        """
        if not isinstance(field, str):
            raise ValueError(
                f"field should be a string but got: {type(field)} with value: {field}"
            )

        if field.startswith("$"):
            raise ValueError(
                f"Invalid filter condition. Expected a field but got an operator: "
                f"{field}"
            )

        # Allow [a-zA-Z0-9_], disallow $ for now until we support escape characters
        if not field.isidentifier():
            raise ValueError(
                f"Invalid field name: {field}. Expected a valid identifier."
            )

        if isinstance(value, dict):
            # This is a filter specification
            if len(value) != 1:
                raise ValueError(
                    "Invalid filter condition. Expected a value which "
                    "is a dictionary with a single key that corresponds to an operator "
                    f"but got a dictionary with {len(value)} keys. The first few "
                    f"keys are: {list(value.keys())[:3]}"
                )
            operator, filter_value = list(value.items())[0]
            # Verify that that operator is an operator
            if operator not in SUPPORTED_OPERATORS:
                raise ValueError(
                    f"Invalid operator: {operator}. "
                    f"Expected one of {SUPPORTED_OPERATORS}"
                )
        else:  # Then we assume an equality operator
            operator = "$eq"
            filter_value = value

        suffix_id = str(uuid.uuid4()).split("-")[0]
        if operator in COMPARISONS_TO_NATIVE:
            # Then we implement an equality filter
            # native is trusted input
            native = COMPARISONS_TO_NATIVE[operator]
            param_name = f"{field}_{suffix_id}"
            return f"{field} {native} :{param_name}", {f"{param_name}": filter_value}
        elif operator == "$between":
            # Use AND with two comparisons
            low, high = filter_value
            low_param_name = f"{field}_low_{suffix_id}"
            high_param_name = f"{field}_high_{suffix_id}"
            return f"({field} BETWEEN :{low_param_name} AND :{high_param_name})", {
                f"{low_param_name}": low,
                f"{high_param_name}": high,
            }
        elif operator in {"$in", "$nin"}:
            # We'll do force coercion to text
            for val in filter_value:
                if not isinstance(val, (str, int, float)):
                    raise NotImplementedError(
                        f"Unsupported type: {type(val)} for value: {val}"
                    )

                if isinstance(val, bool):  # b/c bool is an instance of int
                    raise NotImplementedError(
                        f"Unsupported type: {type(val)} for value: {val}"
                    )
            param_name = f"{field}_{operator.replace('$', '')}_{suffix_id}"
            if operator == "$in":
                return f"{field} = ANY(:{param_name})", {f"{param_name}": filter_value}
            else:  # i.e. $nin
                return f"{field} <> ALL (:{param_name})", {
                    f"{param_name}": filter_value
                }

        elif operator in {"$like", "$ilike"}:
            param_name = f"{field}_{operator.replace('$', '')}_{suffix_id}"
            if operator == "$like":
                return f"({field} LIKE :{param_name})", {f"{param_name}": filter_value}
            else:  # i.e. $ilike
                return f"({field} ILIKE :{param_name})", {f"{param_name}": filter_value}
        elif operator == "$exists":
            if not isinstance(filter_value, bool):
                raise ValueError(
                    "Expected a boolean value for $exists "
                    f"operator, but got: {filter_value}"
                )
            else:
                if filter_value:
                    return f"({field} IS NOT NULL)", {}
                else:
                    return f"({field} IS NULL)", {}
        else:
            raise NotImplementedError()

    def _create_filter_clause(self, filters: Any) -> tuple[str, dict]:
        """Create LangChain filter representation to matching SQL where clauses

        Args:
            filters: Dictionary of filters to apply to the query.

        Returns:
            String containing the sql where query.
        """

        if not isinstance(filters, dict):
            raise ValueError(
                f"Invalid type: Expected a dictionary but got type: {type(filters)}"
            )
        if len(filters) == 1:
            # The only operators allowed at the top level are $AND, $OR, and $NOT
            # First check if an operator or a field
            key, value = list(filters.items())[0]
            if key.startswith("$"):
                # Then it's an operator
                if key.lower() not in ["$and", "$or", "$not"]:
                    raise ValueError(
                        f"Invalid filter condition. Expected $and, $or or $not "
                        f"but got: {key}"
                    )
            else:
                # Then it's a field
                return self._handle_field_filter(field=key, value=filters[key])

            if key.lower() == "$and" or key.lower() == "$or":
                if not isinstance(value, list):
                    raise ValueError(
                        f"Expected a list, but got {type(value)} for value: {value}"
                    )
                op = key[1:].upper()  # Extract the operator
                filter_clause = [self._create_filter_clause(el) for el in value]
                if len(filter_clause) > 1:
                    all_clauses = [clause[0] for clause in filter_clause]
                    params = {}
                    for clause in filter_clause:
                        params.update(clause[1])
                    return f"({f' {op} '.join(all_clauses)})", params
                elif len(filter_clause) == 1:
                    return filter_clause[0]
                else:
                    raise ValueError(
                        "Invalid filter condition. Expected a dictionary "
                        "but got an empty dictionary"
                    )
            elif key.lower() == "$not":
                if isinstance(value, list):
                    not_conditions = [
                        self._create_filter_clause(item) for item in value
                    ]
                    all_clauses = [clause[0] for clause in not_conditions]
                    params = {}
                    for clause in not_conditions:
                        params.update(clause[1])
                    not_stmts = [f"NOT {condition}" for condition in all_clauses]
                    return f"({' AND '.join(not_stmts)})", params
                elif isinstance(value, dict):
                    not_, params = self._create_filter_clause(value)
                    return f"(NOT {not_})", params
                else:
                    raise ValueError(
                        f"Invalid filter condition. Expected a dictionary "
                        f"or a list but got: {type(value)}"
                    )
            else:
                raise ValueError(
                    f"Invalid filter condition. Expected $and, $or or $not "
                    f"but got: {key}"
                )
        elif len(filters) > 1:
            # Then all keys have to be fields (they cannot be operators)
            for key in filters.keys():
                if key.startswith("$"):
                    raise ValueError(
                        f"Invalid filter condition. Expected a field but got: {key}"
                    )
            # These should all be fields and combined using an $and operator
            and_ = [
                self._handle_field_filter(field=k, value=v) for k, v in filters.items()
            ]
            if len(and_) > 1:
                all_clauses = [clause[0] for clause in and_]
                params = {}
                for clause in and_:
                    params.update(clause[1])
                return f"({' AND '.join(all_clauses)})", params
            elif len(and_) == 1:
                return and_[0]
            else:
                raise ValueError(
                    "Invalid filter condition. Expected a dictionary "
                    "but got an empty dictionary"
                )
        else:
            return "", {}

    def get_by_ids(self, ids: Sequence[str]) -> list[Document]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
        )

    def add_documents(
        self,
        documents: list[Document],
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> list[str]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
        )

    def delete(
        self,
        ids: Optional[list] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
        )

    @classmethod
    def from_texts(  # type: ignore[override]
        cls: type[AsyncPGVectorStore],
        texts: list[str],
        embedding: Embeddings,
        engine: PGEngine,
        table_name: str,
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        ignore_metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        **kwargs: Any,
    ) -> AsyncPGVectorStore:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
        )

    @classmethod
    def from_documents(  # type: ignore[override]
        cls: type[AsyncPGVectorStore],
        documents: list[Document],
        embedding: Embeddings,
        engine: PGEngine,
        table_name: str,
        ids: Optional[list] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[str]] = None,
        ignore_metadata_columns: Optional[list[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        **kwargs: Any,
    ) -> AsyncPGVectorStore:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
        )

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: Optional[int] = None,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
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
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
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
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
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
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGVectorStore. Use PGVectorStore interface instead."
        )
