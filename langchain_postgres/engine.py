from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from threading import Thread
from typing import TYPE_CHECKING, Any, Awaitable, Optional, TypeVar, Union

from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

if TYPE_CHECKING:
    import asyncpg  # type: ignore

T = TypeVar("T")


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

        driver = "postgresql+asyncpg"
        if (isinstance(url, str) and not url.startswith(driver)) or (
            isinstance(url, URL) and url.drivername != driver
        ):
            raise ValueError("Driver must be type 'postgresql+asyncpg'")

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

    def escape_postgres_identifier(self, name: str) -> str:
        return name.replace('"', '""')

    async def _ainit_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        *,
        schema_name: str = "public",
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: Optional[list[Column]] = None,
        metadata_json_column: str = "langchain_metadata",
        id_column: Union[str, Column] = "langchain_id",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
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
            metadata_columns (Optional[list[Column]]): A list of Columns to create for custom
                metadata. Default: None. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            id_column (Union[str, Column]) :  Column to store ids.
                Default: "langchain_id" column name with data type UUID. Optional.
            overwrite_existing (bool): Whether to drop existing table. Default: False.
            store_metadata (bool): Whether to store metadata in the table.
                Default: True.

        Raises:
            :class:`DuplicateTableError <asyncpg.exceptions.DuplicateTableError>`: if table already exists.
            :class:`UndefinedObjectError <asyncpg.exceptions.UndefinedObjectError>`: if the data type of the id column is not a postgreSQL data type.
        """

        schema_name = self.escape_postgres_identifier(schema_name)
        table_name = self.escape_postgres_identifier(table_name)
        content_column = self.escape_postgres_identifier(content_column)
        embedding_column = self.escape_postgres_identifier(embedding_column)
        if metadata_columns is None:
            metadata_columns = []
        else:
            for col in metadata_columns:
                col.name = self.escape_postgres_identifier(col.name)
        if isinstance(id_column, str):
            id_column = self.escape_postgres_identifier(id_column)
        else:
            id_column.name = self.escape_postgres_identifier(id_column.name)

        async with self._pool.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()

        if overwrite_existing:
            async with self._pool.connect() as conn:
                await conn.execute(
                    text(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"')
                )
                await conn.commit()

        id_data_type = "UUID" if isinstance(id_column, str) else id_column.data_type
        id_column_name = id_column if isinstance(id_column, str) else id_column.name

        query = f"""CREATE TABLE "{schema_name}"."{table_name}"(
            "{id_column_name}" {id_data_type} PRIMARY KEY,
            "{content_column}" TEXT NOT NULL,
            "{embedding_column}" vector({vector_size}) NOT NULL"""
        for column in metadata_columns:
            nullable = "NOT NULL" if not column.nullable else ""
            query += f',\n"{column.name}" {column.data_type} {nullable}'
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
        metadata_columns: Optional[list[Column]] = None,
        metadata_json_column: str = "langchain_metadata",
        id_column: Union[str, Column] = "langchain_id",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
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
            metadata_columns (Optional[list[Column]]): A list of Columns to create for custom
                metadata. Default: None. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            id_column (Union[str, Column]) :  Column to store ids.
                Default: "langchain_id" column name with data type UUID. Optional.
            overwrite_existing (bool): Whether to drop existing table. Default: False.
            store_metadata (bool): Whether to store metadata in the table.
                Default: True.
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
        metadata_columns: Optional[list[Column]] = None,
        metadata_json_column: str = "langchain_metadata",
        id_column: Union[str, Column] = "langchain_id",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
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
            metadata_columns (Optional[list[Column]]): A list of Columns to create for custom
                metadata. Default: None. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: "langchain_metadata". Optional.
            id_column (Union[str, Column]) :  Column to store ids.
                Default: "langchain_id" column name with data type UUID. Optional.
            overwrite_existing (bool): Whether to drop existing table. Default: False.
            store_metadata (bool): Whether to store metadata in the table.
                Default: True.
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
            )
        )
