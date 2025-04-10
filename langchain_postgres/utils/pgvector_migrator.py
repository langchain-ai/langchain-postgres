import asyncio
import json
import warnings
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, TypeVar

from sqlalchemy import RowMapping, text
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError

from ..v2.engine import PGEngine
from ..v2.vectorstores import PGVectorStore

COLLECTIONS_TABLE = "langchain_pg_collection"
EMBEDDINGS_TABLE = "langchain_pg_embedding"

T = TypeVar("T")


async def __aget_collection_uuid(
    engine: PGEngine,
    collection_name: str,
) -> str:
    """
    Get the collection uuid for a collection present in PGVector tables.

    Args:
        engine (PGEngine): The PG engine corresponding to the Database.
        collection_name (str): The name of the collection to get the uuid for.
    Returns:
        The uuid corresponding to the collection.
    """
    query = f"SELECT name, uuid FROM {COLLECTIONS_TABLE} WHERE name = :collection_name"
    async with engine._pool.connect() as conn:
        result = await conn.execute(
            text(query), parameters={"collection_name": collection_name}
        )
        result_map = result.mappings()
        result_fetch = result_map.fetchone()
    if result_fetch is None:
        raise ValueError(f"Collection, {collection_name} not found.")
    return result_fetch.uuid


async def __aextract_pgvector_collection(
    engine: PGEngine,
    collection_name: str,
    batch_size: int = 1000,
) -> AsyncIterator[Sequence[RowMapping]]:
    """
    Extract all data belonging to a PGVector collection.

    Args:
        engine (PGEngine): The PG engine corresponding to the Database.
        collection_name (str): The name of the collection to get the data for.
        batch_size (int): The batch size for collection extraction.
            Default: 1000. Optional.

    Yields:
        The data present in the collection.
    """
    try:
        uuid_task = asyncio.create_task(__aget_collection_uuid(engine, collection_name))
        query = f"SELECT * FROM {EMBEDDINGS_TABLE} WHERE collection_id = :id"
        async with engine._pool.connect() as conn:
            uuid = await uuid_task
            result_proxy = await conn.execute(text(query), parameters={"id": uuid})
            while True:
                rows = result_proxy.fetchmany(size=batch_size)
                if not rows:
                    break
                yield [row._mapping for row in rows]
    except ValueError:
        raise ValueError(f"Collection, {collection_name} does not exist.")
    except SQLAlchemyError as e:
        raise ProgrammingError(
            statement=f"Failed to extract data from collection '{collection_name}': {e}",
            params={"id": uuid},
            orig=e,
        ) from e


async def __concurrent_batch_insert(
    data_batches: AsyncIterator[Sequence[RowMapping]],
    vector_store: PGVectorStore,
    max_concurrency: int = 100,
) -> None:
    pending: set[Any] = set()
    async for batch_data in data_batches:
        pending.add(
            asyncio.ensure_future(
                vector_store.aadd_embeddings(
                    texts=[data.document for data in batch_data],
                    embeddings=[json.loads(data.embedding) for data in batch_data],
                    metadatas=[data.cmetadata for data in batch_data],
                    ids=[data.id for data in batch_data],
                )
            )
        )
        if len(pending) >= max_concurrency:
            _, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
    if pending:
        await asyncio.wait(pending)


async def __amigrate_pgvector_collection(
    engine: PGEngine,
    collection_name: str,
    vector_store: PGVectorStore,
    delete_pg_collection: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """
    Migrate all data present in a PGVector collection to use separate tables for each collection.
    The new data format is compatible with the PGVectoreStore interface.

    Args:
        engine (PGEngine): The PG engine corresponding to the Database.
        collection_name (str): The collection to migrate.
        vector_store (PGVectorStore): The PGVectorStore object corresponding to the new collection table.
        delete_pg_collection (bool): An option to delete the original data upon migration.
            Default: False. Optional.
        insert_batch_size (int): Number of rows to insert at once in the table.
            Default: 1000.
    """
    destination_table = vector_store.get_table_name()

    # Get row count in PGVector collection
    uuid_task = asyncio.create_task(__aget_collection_uuid(engine, collection_name))
    query = (
        f"SELECT COUNT(*) FROM {EMBEDDINGS_TABLE} WHERE collection_id=:collection_id"
    )
    async with engine._pool.connect() as conn:
        uuid = await uuid_task
        result = await conn.execute(text(query), parameters={"collection_id": uuid})
        result_map = result.mappings()
        collection_data_len = result_map.fetchone()
    if collection_data_len is None:
        warnings.warn(f"Collection, {collection_name} contains no elements.")
        return

    # Extract data from the collection and batch insert into the new table
    data_batches = __aextract_pgvector_collection(
        engine, collection_name, batch_size=insert_batch_size
    )
    await __concurrent_batch_insert(data_batches, vector_store, max_concurrency=100)

    # Validate data migration
    query = f"SELECT COUNT(*) FROM {destination_table}"
    async with engine._pool.connect() as conn:
        result = await conn.execute(text(query))
        result_map = result.mappings()
        table_size = result_map.fetchone()
    if not table_size:
        raise ValueError(f"Table: {destination_table} does not exist.")

    if collection_data_len["count"] != table_size["count"]:
        raise ValueError(
            "All data not yet migrated.\n"
            f"Original row count: {collection_data_len['count']}\n"
            f"Collection table, {destination_table} row count: {table_size['count']}"
        )
    elif delete_pg_collection:
        # Delete PGVector data
        query = f"DELETE FROM {EMBEDDINGS_TABLE} WHERE collection_id=:collection_id"
        async with engine._pool.connect() as conn:
            await conn.execute(text(query), parameters={"collection_id": uuid})
            await conn.commit()

        query = f"DELETE FROM {COLLECTIONS_TABLE} WHERE name=:collection_name"
        async with engine._pool.connect() as conn:
            await conn.execute(
                text(query), parameters={"collection_name": collection_name}
            )
            await conn.commit()
        print(f"Successfully deleted PGVector collection, {collection_name}")


async def __alist_pgvector_collection_names(
    engine: PGEngine,
) -> list[str]:
    """Lists all collection names present in PGVector table."""
    try:
        query = f"SELECT name from {COLLECTIONS_TABLE}"
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            all_rows = result_map.fetchall()
        return [row["name"] for row in all_rows]
    except ProgrammingError as e:
        raise ValueError(
            "Please provide the correct collection table name: " + str(e)
        ) from e


async def aextract_pgvector_collection(
    engine: PGEngine,
    collection_name: str,
    batch_size: int = 1000,
) -> AsyncIterator[Sequence[RowMapping]]:
    """
    Extract all data belonging to a PGVector collection.

    Args:
        engine (PGEngine): The PG engine corresponding to the Database.
        collection_name (str): The name of the collection to get the data for.
        batch_size (int): The batch size for collection extraction.
            Default: 1000. Optional.

    Yields:
        The data present in the collection.
    """
    iterator = __aextract_pgvector_collection(engine, collection_name, batch_size)
    while True:
        try:
            result = await engine._run_as_async(iterator.__anext__())
            yield result
        except StopAsyncIteration:
            break


async def alist_pgvector_collection_names(
    engine: PGEngine,
) -> list[str]:
    """Lists all collection names present in PGVector table."""
    return await engine._run_as_async(__alist_pgvector_collection_names(engine))


async def amigrate_pgvector_collection(
    engine: PGEngine,
    collection_name: str,
    vector_store: PGVectorStore,
    delete_pg_collection: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """
    Migrate all data present in a PGVector collection to use separate tables for each collection.
    The new data format is compatible with the PGVectorStore interface.

    Args:
        engine (PGEngine): The PG engine corresponding to the Database.
        collection_name (str): The collection to migrate.
        vector_store (PGVectorStore): The PGVectorStore object corresponding to the new collection table.
        use_json_metadata (bool): An option to keep the PGVector metadata as json in the new table.
            Default: False. Optional.
        delete_pg_collection (bool): An option to delete the original data upon migration.
            Default: False. Optional.
        insert_batch_size (int): Number of rows to insert at once in the table.
            Default: 1000.
    """
    await engine._run_as_async(
        __amigrate_pgvector_collection(
            engine,
            collection_name,
            vector_store,
            delete_pg_collection,
            insert_batch_size,
        )
    )


def extract_pgvector_collection(
    engine: PGEngine,
    collection_name: str,
    batch_size: int = 1000,
) -> Iterator[Sequence[RowMapping]]:
    """
    Extract all data belonging to a PGVector collection.

    Args:
        engine (PGEngine): The PG engine corresponding to the Database.
        collection_name (str): The name of the collection to get the data for.
        batch_size (int): The batch size for collection extraction.
            Default: 1000. Optional.

    Yields:
        The data present in the collection.
    """
    iterator = __aextract_pgvector_collection(engine, collection_name, batch_size)
    while True:
        try:
            result = engine._run_as_sync(iterator.__anext__())
            yield result
        except StopAsyncIteration:
            break


def list_pgvector_collection_names(engine: PGEngine) -> list[str]:
    """Lists all collection names present in PGVector table."""
    return engine._run_as_sync(__alist_pgvector_collection_names(engine))


def migrate_pgvector_collection(
    engine: PGEngine,
    collection_name: str,
    vector_store: PGVectorStore,
    delete_pg_collection: Optional[bool] = False,
    insert_batch_size: int = 1000,
) -> None:
    """
    Migrate all data present in a PGVector collection to use separate tables for each collection.
    The new data format is compatible with the PGVectorStore interface.

    Args:
        engine (PGEngine): The PG engine corresponding to the Database.
        collection_name (str): The collection to migrate.
        vector_store (PGVectorStore): The PGVectorStore object corresponding to the new collection table.
        delete_pg_collection (bool): An option to delete the original data upon migration.
            Default: False. Optional.
        insert_batch_size (int): Number of rows to insert at once in the table.
            Default: 1000.
    """
    engine._run_as_sync(
        __amigrate_pgvector_collection(
            engine,
            collection_name,
            vector_store,
            delete_pg_collection,
            insert_batch_size,
        )
    )
