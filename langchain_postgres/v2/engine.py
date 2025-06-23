from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import Thread
from typing import Any, Awaitable, Optional, TypedDict, TypeVar, Union

from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .hybrid_search_config import HybridSearchConfig

T = TypeVar("T")


class ColumnDict(TypedDict):
    name: str
    data_type: str
    nullable: bool


@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool = True

    def __post_init__(self) -> None:
        """Check if initialization parameters are valid.

        Raises:
            ValueError: If Column name is not string.
            ValueError: If data_type is not type string.
        """

        if not isinstance(self.name, str):
            raise ValueError("Column name must be type string")
        if not isinstance(self.data_type, str):
            raise ValueError("Column data_type must be type string")


class PGEngine:
    """A class for managing connections to a Postgres database."""

    _default_loop: Optional[asyncio.AbstractEventLoop] = None
    _default_thread: Optional[Thread] = None
    __create_key = object()

    def __init__(
        self,
        key: object,
        pool: AsyncEngine,
        loop: Optional[asyncio.AbstractEventLoop],
        thread: Optional[Thread],
    ) -> None:
        """PGEngine constructor.

        Args:
            key (object): Prevent direct constructor usage.
            pool (AsyncEngine): Async engine connection pool.
            loop (Optional[asyncio.AbstractEventLoop]): Async event loop used to create the engine.
            thread (Optional[Thread]): Thread used to create the engine async.

        Raises:
            Exception: If the constructor is called directly by the user.
        """

        if key != PGEngine.__create_key:
            raise Exception(
                "Only create class through 'from_connection_string' or 'from_engine' methods!"
            )
        self._pool = pool
        self._loop = loop
        self._thread = thread

    @classmethod
    def from_engine(
        cls: type[PGEngine],
        engine: AsyncEngine,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> PGEngine:
        """Create an PGEngine instance from an AsyncEngine."""
        return cls(cls.__create_key, engine, loop, None)

    @classmethod
    def from_connection_string(
        cls,
        url: str | URL,
        **kwargs: Any,
    ) -> PGEngine:
        """Create an PGEngine instance from arguments

        Args:
            url (Optional[str]): the URL used to connect to a database. Use url or set other arguments.

        Raises:
            ValueError: If not all database url arguments are specified

        Returns:
            PGEngine
        """
        # Running a loop in a background thread allows us to support
        # async methods from non-async environments
        if cls._default_loop is None:
            cls._default_loop = asyncio.new_event_loop()
            cls._default_thread = Thread(
                target=cls._default_loop.run_forever, daemon=True
            )
            cls._default_thread.start()

        engine = create_async_engine(url, **kwargs)
        return cls(cls.__create_key, engine, cls._default_loop, cls._default_thread)

    async def _run_as_async(self, coro: Awaitable[T]) -> T:
        """Run an async coroutine asynchronously"""
        # If a loop has not been provided, attempt to run in current thread
        if not self._loop:
            return await coro
        # Otherwise, run in the background thread
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        )

    def _run_as_sync(self, coro: Awaitable[T]) -> T:
        """Run an async coroutine synchronously"""
        if not self._loop:
            raise Exception(
                "Engine was initialized without a background loop and cannot call sync methods."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def close(self) -> None:
        """Dispose of connection pool"""
        await self._run_as_async(self._pool.dispose())

    def _escape_postgres_identifier(self, name: str) -> str:
        return name.replace('"', '""')

    def _validate_column_dict(self, col: ColumnDict) -> None:
        if not isinstance(col.get("name"), str):
            raise TypeError("The 'name' field must be a string.")
        if not isinstance(col.get("data_type"), str):
            raise TypeError("The 'data_type' field must be a string.")
        if not isinstance(col.get("nullable"), bool):
            raise TypeError("The 'nullable' field must be a boolean.")

    async def _ainit_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[Union[Column, ColumnDict]]] = None,
        metadata_json_column: str = "langchain_metadata",
        id_column: Union[str, Column, ColumnDict] = "langchain_id",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ) -> None:
        """
        Create a table for saving of vectors to be used with PGVectorStore.

        Args:
            table_name (str): The database table name.
            vector_size (int): Vector size for the embedding model to be used.
            schema_name (str): The schema name.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            embedding_column (str) : Name of the column to store vector embeddings.
                Default: "embedding".
            metadata_columns (Optional[list[Union[Column, ColumnDict]]]): A list of Columns to create for custom
                metadata. Default: None. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            id_column (Union[str, Column, ColumnDict]) :  Column to store ids.
                Default: "langchain_id" column name with data type UUID. Optional.
            overwrite_existing (bool): Whether to drop existing table. Default: False.
            store_metadata (bool): Whether to store metadata in the table.
                Default: True.
            hybrid_search_config (HybridSearchConfig): Hybrid search configuration.
                Default: None.

        Raises:
            :class:`DuplicateTableError <asyncpg.exceptions.DuplicateTableError>`: if table already exists.
            :class:`UndefinedObjectError <asyncpg.exceptions.UndefinedObjectError>`: if the data type of the id column is not a postgreSQL data type.
        """

        schema_name = self._escape_postgres_identifier(schema_name)
        table_name = self._escape_postgres_identifier(table_name)
        hybrid_search_default_column_name = content_column + "_tsv"
        content_column = self._escape_postgres_identifier(content_column)
        embedding_column = self._escape_postgres_identifier(embedding_column)
        if metadata_columns is None:
            metadata_columns = []
        else:
            for col in metadata_columns:
                if isinstance(col, Column):
                    col.name = self._escape_postgres_identifier(col.name)
                elif isinstance(col, dict):
                    self._validate_column_dict(col)
                    col["name"] = self._escape_postgres_identifier(col["name"])
        if isinstance(id_column, str):
            id_column = self._escape_postgres_identifier(id_column)
        elif isinstance(id_column, Column):
            id_column.name = self._escape_postgres_identifier(id_column.name)
        else:
            self._validate_column_dict(id_column)
            id_column["name"] = self._escape_postgres_identifier(id_column["name"])

        async with self._pool.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()

        if overwrite_existing:
            async with self._pool.connect() as conn:
                await conn.execute(
                    text(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"')
                )
                await conn.commit()

        if isinstance(id_column, str):
            id_data_type = "UUID"
            id_column_name = id_column
        elif isinstance(id_column, Column):
            id_data_type = id_column.data_type
            id_column_name = id_column.name
        else:
            id_data_type = id_column["data_type"]
            id_column_name = id_column["name"]

        hybrid_search_column = ""  # Default is no TSV column for hybrid search
        if hybrid_search_config:
            hybrid_search_column_name = (
                hybrid_search_config.tsv_column or hybrid_search_default_column_name
            )
            hybrid_search_column_name = self._escape_postgres_identifier(
                hybrid_search_column_name
            )
            hybrid_search_config.tsv_column = hybrid_search_column_name
            hybrid_search_column = f',"{self._escape_postgres_identifier(hybrid_search_column_name)}" TSVECTOR NOT NULL'

        query = f"""CREATE TABLE "{schema_name}"."{table_name}"(
            "{id_column_name}" {id_data_type} PRIMARY KEY,
            "{content_column}" TEXT NOT NULL,
            "{embedding_column}" vector({vector_size}) NOT NULL
            {hybrid_search_column}"""
        for column in metadata_columns:
            if isinstance(column, Column):
                nullable = "NOT NULL" if not column.nullable else ""
                query += f',\n"{column.name}" {column.data_type} {nullable}'
            elif isinstance(column, dict):
                nullable = "NOT NULL" if not column["nullable"] else ""
                query += f',\n"{column["name"]}" {column["data_type"]} {nullable}'
        if store_metadata:
            query += f""",\n"{metadata_json_column}" JSON"""
        query += "\n);"

        async with self._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def ainit_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[Union[Column, ColumnDict]]] = None,
        metadata_json_column: str = "langchain_metadata",
        id_column: Union[str, Column, ColumnDict] = "langchain_id",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ) -> None:
        """
        Create a table for saving of vectors to be used with PGVectorStore.

        Args:
            table_name (str): The database table name.
            vector_size (int): Vector size for the embedding model to be used.
            schema_name (str): The schema name.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            embedding_column (str) : Name of the column to store vector embeddings.
                Default: "embedding".
            metadata_columns (Optional[list[Union[Column, ColumnDict]]]): A list of Columns to create for custom
                metadata. Default: None. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            id_column (Union[str, Column, ColumnDict]) :  Column to store ids.
                Default: "langchain_id" column name with data type UUID. Optional.
            overwrite_existing (bool): Whether to drop existing table. Default: False.
            store_metadata (bool): Whether to store metadata in the table.
                Default: True.
            hybrid_search_config (HybridSearchConfig): Hybrid search configuration.
                Note that queries might be slow if the hybrid search column does not exist.
                For best hybrid search performance, consider creating a TSV column and adding GIN index.
                Default: None.
        """
        await self._run_as_async(
            self._ainit_vectorstore_table(
                table_name,
                vector_size,
                schema_name=schema_name,
                content_column=content_column,
                embedding_column=embedding_column,
                metadata_columns=metadata_columns,
                metadata_json_column=metadata_json_column,
                id_column=id_column,
                overwrite_existing=overwrite_existing,
                store_metadata=store_metadata,
                hybrid_search_config=hybrid_search_config,
            )
        )

    def init_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[Union[Column, ColumnDict]]] = None,
        metadata_json_column: str = "langchain_metadata",
        id_column: Union[str, Column, ColumnDict] = "langchain_id",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
        hybrid_search_config: Optional[HybridSearchConfig] = None,
    ) -> None:
        """
        Create a table for saving of vectors to be used with PGVectorStore.

        Args:
            table_name (str): The database table name.
            vector_size (int): Vector size for the embedding model to be used.
            schema_name (str): The schema name.
                Default: "public".
            content_column (str): Name of the column to store document content.
                Default: "page_content".
            embedding_column (str) : Name of the column to store vector embeddings.
                Default: "embedding".
            metadata_columns (Optional[list[Union[Column, ColumnDict]]]): A list of Columns to create for custom
                metadata. Default: None. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            id_column (Union[str, Column, ColumnDict]) :  Column to store ids.
                Default: "langchain_id" column name with data type UUID. Optional.
            overwrite_existing (bool): Whether to drop existing table. Default: False.
            store_metadata (bool): Whether to store metadata in the table.
                Default: True.
            hybrid_search_config (HybridSearchConfig): Hybrid search configuration.
                Note that queries might be slow if the hybrid search column does not exist.
                For best hybrid search performance, consider creating a TSV column and adding GIN index.
                Default: None.
        """
        self._run_as_sync(
            self._ainit_vectorstore_table(
                table_name,
                vector_size,
                schema_name=schema_name,
                content_column=content_column,
                embedding_column=embedding_column,
                metadata_columns=metadata_columns,
                metadata_json_column=metadata_json_column,
                id_column=id_column,
                overwrite_existing=overwrite_existing,
                store_metadata=store_metadata,
                hybrid_search_config=hybrid_search_config,
            )
        )

    async def _adrop_table(
        self,
        table_name: str,
        *,
        schema_name: str = "public",
    ) -> None:
        """Drop the vector store table"""
        query = f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}";'
        async with self._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def adrop_table(
        self,
        table_name: str,
        *,
        schema_name: str = "public",
    ) -> None:
        await self._run_as_async(
            self._adrop_table(table_name=table_name, schema_name=schema_name)
        )

    def drop_table(
        self,
        table_name: str,
        *,
        schema_name: str = "public",
    ) -> None:
        self._run_as_sync(
            self._adrop_table(table_name=table_name, schema_name=schema_name)
        )
