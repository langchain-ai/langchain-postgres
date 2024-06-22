# pylint: disable=too-many-lines
from __future__ import annotations

import contextlib
import enum
import logging
import uuid
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from typing import (
    cast as typing_cast,
)

import sqlalchemy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexes import DeleteResponse, UpsertResponse
from langchain_core.indexes.base import Index
from langchain_core.stores import K
from langchain_core.structured_query import StructuredQuery
from langchain_core.utils.iter import batch_iterate
from sqlalchemy import (
    SQLColumnExpression,
    and_,
    cast,
    create_engine,
    delete,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSON, JSONB, JSONPATH, UUID, insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import (
    Session,
    declarative_base,
    relationship,
    scoped_session,
    sessionmaker,
)


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

Base = declarative_base()  # type: Any


_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"


_classes: Any = None

COMPARISONS_TO_NATIVE = {
    "$eq": "==",
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


def _as_id_generator(ids: Optional[Sequence[str]]) -> Generator[str, None, None]:
    """Return an id generator."""
    if ids:
        for id in ids:
            yield id
    else:
        while True:
            yield str(uuid.uuid4())


def _get_embedding_collection_store(vector_dimension: Optional[int] = None) -> Any:
    global _classes
    if _classes is not None:
        return _classes

    from pgvector.sqlalchemy import Vector  # type: ignore

    class CollectionStore(Base):
        """Collection store."""

        __tablename__ = "langchain_pg_collection"

        uuid = sqlalchemy.Column(
            UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
        )
        name = sqlalchemy.Column(sqlalchemy.String, nullable=False, unique=True)
        cmetadata = sqlalchemy.Column(JSON)

        embeddings = relationship(
            "EmbeddingStore",
            back_populates="collection",
            passive_deletes=True,
        )

        @classmethod
        def get_by_name(
            cls, session: Session, name: str
        ) -> Optional["CollectionStore"]:
            return (
                session.query(cls)
                .filter(typing_cast(sqlalchemy.Column, cls.name) == name)
                .first()
            )

        @classmethod
        async def aget_by_name(
            cls, session: AsyncSession, name: str
        ) -> Optional["CollectionStore"]:
            return (
                (
                    await session.execute(
                        select(CollectionStore).where(
                            typing_cast(sqlalchemy.Column, cls.name) == name
                        )
                    )
                )
                .scalars()
                .first()
            )

        @classmethod
        def get_or_create(
            cls,
            session: Session,
            name: str,
            cmetadata: Optional[dict] = None,
        ) -> Tuple["CollectionStore", bool]:
            """Get or create a collection.
            Returns:
                 Where the bool is True if the collection was created.
            """  # noqa: E501
            created = False
            collection = cls.get_by_name(session, name)
            if collection:
                return collection, created

            collection = cls(name=name, cmetadata=cmetadata)
            session.add(collection)
            session.commit()
            created = True
            return collection, created

        @classmethod
        async def aget_or_create(
            cls,
            session: AsyncSession,
            name: str,
            cmetadata: Optional[dict] = None,
        ) -> Tuple["CollectionStore", bool]:
            """
            Get or create a collection.
            Returns [Collection, bool] where the bool is True if the collection was created.
            """  # noqa: E501
            created = False
            collection = await cls.aget_by_name(session, name)
            if collection:
                return collection, created

            collection = cls(name=name, cmetadata=cmetadata)
            session.add(collection)
            await session.commit()
            created = True
            return collection, created

    class EmbeddingStore(Base):
        """Embedding store."""

        __tablename__ = "langchain_pg_embedding"

        id = sqlalchemy.Column(
            sqlalchemy.String, nullable=True, primary_key=True, index=True, unique=True
        )

        collection_id = sqlalchemy.Column(
            UUID(as_uuid=True),
            sqlalchemy.ForeignKey(
                f"{CollectionStore.__tablename__}.uuid",
                ondelete="CASCADE",
            ),
        )
        collection = relationship(CollectionStore, back_populates="embeddings")

        embedding: Vector = sqlalchemy.Column(Vector(vector_dimension))
        document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
        cmetadata = sqlalchemy.Column(JSONB, nullable=True)

        __table_args__ = (
            sqlalchemy.Index(
                "ix_cmetadata_gin",
                "cmetadata",
                postgresql_using="gin",
                postgresql_ops={"cmetadata": "jsonb_path_ops"},
            ),
        )

    _classes = (EmbeddingStore, CollectionStore)

    return _classes


def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]


def _create_vector_extension(conn: Connection) -> None:
    statement = sqlalchemy.text(
        "SELECT pg_advisory_xact_lock(1573678846307946496);"
        "CREATE EXTENSION IF NOT EXISTS vector;"
    )
    conn.execute(statement)
    conn.commit()


DBConnection = Union[sqlalchemy.engine.Engine, str]


class PostgresDocumentIndex(Index):
    """Vectorstore implementation using Postgres as the backend.

    Currently, there is no mechanism for supporting data migration.

    So breaking changes in the vectorstore schema will require the user to recreate
    the tables and re-add the documents.

    If this is a concern, please use a different vectorstore. If
    not, this implementation should be fine for your use case.

    To use this vectorstore you need to have the `vector` extension installed.
    The `vector` extension is a Postgres extension that provides vector
    similarity search capabilities.

    ```sh
    docker run --name pgvector-container -e POSTGRES_PASSWORD=...
        -d pgvector/pgvector:pg16
    ```

    Example:
        .. code-block:: python

            from langchain_postgres.vectorstores import PGVector
            from langchain_openai.embeddings import OpenAIEmbeddings

            connection_string = "postgresql+psycopg://..."
            collection_name = "state_of_the_union_test"
            embeddings = OpenAIEmbeddings()
            vectorstore = PGVector.from_documents(
                embedding=embeddings,
                documents=docs,
                connection=connection_string,
                collection_name=collection_name,
                use_jsonb=True,
                async_mode=False,
            )


    This code has been ported over from langchain_community with minimal changes
    to allow users to easily transition from langchain_community to langchain_postgres.

    Some changes had to be made to address issues with the community implementation:
    * langchain_postgres now works with psycopg3. Please update your
      connection strings from `postgresql+psycopg2://...` to
      `postgresql+psycopg://langchain:langchain@...`
      (yes, the driver name is `psycopg` not `psycopg3`)
    * The schema of the embedding store and collection have been changed to make
      add_documents work correctly with user specified ids, specifically
      when overwriting existing documents.
      You will need to recreate the tables if you are using an existing database.
    * A Connection object has to be provided explicitly. Connections will not be
      picked up automatically based on env variables.
    * langchain_postgres now accept async connections. If you want to use the async
        version, you need to set `async_mode=True` when initializing the store or
        use an async engine.

    Supported filter operators:

    * $eq: Equality operator
    * $ne: Not equal operator
    * $lt: Less than operator
    * $lte: Less than or equal operator
    * $gt: Greater than operator
    * $gte: Greater than or equal operator
    * $in: In operator
    * $nin: Not in operator
    * $between: Between operator
    * $exists: Exists operator
    * $like: Like operator
    * $ilike: Case insensitive like operator
    * $and: Logical AND operator
    * $or: Logical OR operator
    * $not: Logical NOT operator

    Example:

    .. code-block:: python

        vectorstore.similarity_search('kitty', k=10, filter={
            'id': {'$in': [1, 5, 2, 9]}
        })
        #%% md

        If you provide a dict with multiple fields, but no operators,
        the top level will be interpreted as a logical **AND** filter

        vectorstore.similarity_search('ducks', k=10, filter={
            'id': {'$in': [1, 5, 2, 9]},
            'location': {'$in': ["pond", "market"]}
        })

    """

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        connection: Union[None, DBConnection, Engine, AsyncEngine, str] = None,
        embedding_length: Optional[int] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        collection_metadata: Optional[dict] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        engine_args: Optional[dict[str, Any]] = None,
        create_extension: bool = True,
        async_mode: bool = False,
    ) -> None:
        """Initialize the PGVector store.
        For an async version, use `PGVector.acreate()` instead.

        Args:
            connection: Postgres connection string or (async)engine.
            embeddings: Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface.
            embedding_length: The length of the embedding vector. (default: None)
                NOTE: This is not mandatory. Defining it will prevent vectors of
                any other size to be added to the embeddings table but, without it,
                the embeddings can't be indexed.
            collection_name: The name of the collection to use. (default: langchain)
                NOTE: This is not the name of the table, but the name of the collection.
                The tables will be created when initializing the store (if not exists)
                So, make sure the user has the right permissions to create tables.
            distance_strategy: The distance strategy to use. (default: COSINE)
            pre_delete_collection: If True, will delete the collection if it exists.
                (default: False). Useful for testing.
            engine_args: SQLAlchemy's create engine arguments.
            create_extension: If True, will create the vector extension if it
                doesn't exist. disabling creation is useful when using ReadOnly
                Databases.
        """
        self.async_mode = async_mode
        self.embedding_function = embeddings
        self._embedding_length = embedding_length
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self._distance_strategy = distance_strategy
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self._engine: Optional[Engine] = None
        self._async_engine: Optional[AsyncEngine] = None
        self._async_init = False

        if isinstance(connection, str):
            if async_mode:
                self._async_engine = create_async_engine(
                    connection, **(engine_args or {})
                )
            else:
                self._engine = create_engine(url=connection, **(engine_args or {}))
        elif isinstance(connection, Engine):
            self.async_mode = False
            self._engine = connection
        elif isinstance(connection, AsyncEngine):
            self.async_mode = True
            self._async_engine = connection
        else:
            raise ValueError(
                "connection should be a connection string or an instance of "
                "sqlalchemy.engine.Engine or sqlalchemy.ext.asyncio.engine.AsyncEngine"
            )
        self.session_maker: Union[scoped_session, async_sessionmaker]
        if self.async_mode:
            self.session_maker = async_sessionmaker(bind=self._async_engine)
        else:
            self.session_maker = scoped_session(sessionmaker(bind=self._engine))

        self.create_extension = create_extension

        if not self.async_mode:
            self.__post_init__()

    def __post_init__(
        self,
    ) -> None:
        """Initialize the store."""
        if self.create_extension:
            self.create_vector_extension()

        EmbeddingStore, CollectionStore = _get_embedding_collection_store(
            self._embedding_length
        )
        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore
        self.create_tables_if_not_exists()
        self.create_collection()

    async def __apost_init__(
        self,
    ) -> None:
        """Async initialize the store (use lazy approach)."""
        if self._async_init:  # Warning: possible race condition
            return
        self._async_init = True

        EmbeddingStore, CollectionStore = _get_embedding_collection_store(
            self._embedding_length
        )
        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore
        if self.create_extension:
            await self.acreate_vector_extension()

        await self.acreate_tables_if_not_exists()
        await self.acreate_collection()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def create_vector_extension(self) -> None:
        assert self._engine, "engine not found"
        try:
            with self._engine.connect() as conn:
                _create_vector_extension(conn)
        except Exception as e:
            raise Exception(f"Failed to create vector extension: {e}") from e

    async def acreate_vector_extension(self) -> None:
        assert self._async_engine, "_async_engine not found"

        async with self._async_engine.begin() as conn:
            await conn.run_sync(_create_vector_extension)

    def create_tables_if_not_exists(self) -> None:
        with self._make_sync_session() as session:
            Base.metadata.create_all(session.get_bind())
            session.commit()

    async def acreate_tables_if_not_exists(self) -> None:
        assert self._async_engine, "This method must be called with async_mode"
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def drop_tables(self) -> None:
        with self._make_sync_session() as session:
            Base.metadata.drop_all(session.get_bind())
            session.commit()

    async def adrop_tables(self) -> None:
        assert self._async_engine, "This method must be called with async_mode"
        await self.__apost_init__()  # Lazy async init
        async with self._async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        with self._make_sync_session() as session:
            self.CollectionStore.get_or_create(
                session, self.collection_name, cmetadata=self.collection_metadata
            )
            session.commit()

    async def acreate_collection(self) -> None:
        await self.__apost_init__()  # Lazy async init
        async with self._make_async_session() as session:
            if self.pre_delete_collection:
                await self._adelete_collection(session)
            await self.CollectionStore.aget_or_create(
                session, self.collection_name, cmetadata=self.collection_metadata
            )
            await session.commit()

    def _delete_collection(self, session: Session) -> None:
        collection = self.get_collection(session)
        if not collection:
            self.logger.warning("Collection not found")
            return
        session.delete(collection)

    async def _adelete_collection(self, session: AsyncSession) -> None:
        collection = await self.aget_collection(session)
        if not collection:
            self.logger.warning("Collection not found")
            return
        await session.delete(collection)

    def delete_collection(self) -> None:
        with self._make_sync_session() as session:
            collection = self.get_collection(session)
            if not collection:
                self.logger.warning("Collection not found")
                return
            session.delete(collection)
            session.commit()

    async def adelete_collection(self) -> None:
        await self.__apost_init__()  # Lazy async init
        async with self._make_async_session() as session:
            collection = await self.aget_collection(session)
            if not collection:
                self.logger.warning("Collection not found")
                return
            await session.delete(collection)
            await session.commit()

    def delete(
        self,
        ids: Optional[List[str]] = None,
        collection_only: bool = False,
        **kwargs: Any,
    ) -> None:
        """Delete vectors by ids or uuids.

        Args:
            ids: List of ids to delete.
            collection_only: Only delete ids in the collection.
        """
        with self._make_sync_session() as session:
            if ids is not None:
                self.logger.debug(
                    "Trying to delete vectors by ids (represented by the model "
                    "using the custom ids field)"
                )

                stmt = delete(self.EmbeddingStore)

                if collection_only:
                    collection = self.get_collection(session)
                    if not collection:
                        self.logger.warning("Collection not found")
                        return

                    stmt = stmt.where(
                        self.EmbeddingStore.collection_id == collection.uuid
                    )

                stmt = stmt.where(self.EmbeddingStore.id.in_(ids))
                session.execute(stmt)
            session.commit()

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        collection_only: bool = False,
        **kwargs: Any,
    ) -> None:
        """Async delete vectors by ids or uuids.

        Args:
            ids: List of ids to delete.
            collection_only: Only delete ids in the collection.
        """
        await self.__apost_init__()  # Lazy async init
        async with self._make_async_session() as session:
            if ids is not None:
                self.logger.debug(
                    "Trying to delete vectors by ids (represented by the model "
                    "using the custom ids field)"
                )

                stmt = delete(self.EmbeddingStore)

                if collection_only:
                    collection = await self.aget_collection(session)
                    if not collection:
                        self.logger.warning("Collection not found")
                        return

                    stmt = stmt.where(
                        self.EmbeddingStore.collection_id == collection.uuid
                    )

                stmt = stmt.where(self.EmbeddingStore.id.in_(ids))
                await session.execute(stmt)
            await session.commit()

    def get_collection(self, session: Session) -> Any:
        assert not self._async_engine, "This method must be called without async_mode"
        return self.CollectionStore.get_by_name(session, self.collection_name)

    async def aget_collection(self, session: AsyncSession) -> Any:
        assert self._async_engine, "This method must be called with async_mode"
        await self.__apost_init__()  # Lazy async init
        return await self.CollectionStore.aget_by_name(session, self.collection_name)

    def delete_by_ids(self, ids: Iterable[str]) -> DeleteResponse:
        """Delete documents by ids."""
        batch_size = 1_000
        for batch_ids in batch_iterate(size=batch_size, iterable=ids):
            with self._make_sync_session() as session:
                stmt = delete(self.EmbeddingStore).where(
                    and_(
                        self.EmbeddingStore.id.in_(batch_ids),
                        self.EmbeddingStore.collection_id  # Make into a single query
                        == self.get_collection(session).uuid,
                    )
                )
                session.execute(stmt)
                session.commit()

    def lazy_get_by_ids(self, ids: Iterable[str]) -> Iterable[Document]:
        """Lazy by get IDs."""
        batch_size = 100
        for batch_ids in batch_iterate(size=batch_size, iterable=ids):
            with self._make_sync_session() as session:
                stmt = select(self.EmbeddingStore).where(
                    self.EmbeddingStore.id.in_(list(batch_ids))
                )
                results = session.execute(stmt).scalars().all()
                for result in results:
                    yield Document(
                        page_content=result.document,
                        metadata=result.cmetadata,
                    )

    def lazy_get(
        self,
        *,
        ids: Optional[Iterable[str]] = None,
        filters: Union[
            StructuredQuery, Dict[str, Any], List[Dict[str, Any]], None
        ] = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """Lazy get."""
        batch_size = 100  # <-- Where to surface this? Exposing in lazy_get makes sense.

        if ids:
            raise NotImplementedError()

        if not filters:
            raise NotImplementedError()

        while True:
            num_results = 0
            with self._make_sync_session() as session:
                collection = self.get_collection(session)
                if not isinstance(filters, dict):
                    raise NotImplementedError()

                filter_by = self._create_filter_clause_with_uuid(
                    collection.uuid, filters
                )
                results: List[Any] = (
                    session.query(
                        self.EmbeddingStore,
                    )
                    .filter(*filter_by)
                    .order_by(self.EmbeddingStore.id)
                    .limit(batch_size + 1)
                    .all()
                )
                for result in results:
                    num_results += 1
                    yield Document(
                        page_content=result.document,
                        metadata=result.cmetadata,
                    )

                if num_results < batch_size:
                    break

    def upsert(
        self,
        # TODO: Iterable or Iterator?
        documents: Iterable[Document],
        *,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> UpsertResponse:
        """Upsert documents into the vectorstore.

        Args:
            documents: Iterable of documents to upsert.
            ids: Optional list of ids for the documents.
                 If not provided, will generate a new id for each document.
            kwargs: vectorstore specific parameters

        Returns:
            UpsertResponse
        """
        id_generator = _as_id_generator(ids)
        batch_size = 100  # <-- batch size

        for docs in batch_iterate(size=batch_size, iterable=documents):
            embeddings = self.embedding_function.embed_documents(
                [doc.page_content for doc in docs]
            )

            with self._make_sync_session() as session:  # type: ignore[arg-type]
                collection = self.get_collection(session)
                if not collection:
                    raise ValueError("Collection not found")

                data = []

                for embedding, doc in zip(embeddings, docs):
                    id_ = next(id_generator)
                    if not isinstance(id_, str):
                        raise TypeError()
                    data.append(
                        {
                            "id": id_,
                            "collection_id": collection.uuid,
                            "embedding": embedding,
                            "document": doc.page_content,
                            "cmetadata": doc.metadata or {},
                        }
                    )
                stmt = insert(self.EmbeddingStore).values(data)
                on_conflict_stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    # Conflict detection based on these columns
                    set_={
                        "embedding": stmt.excluded.embedding,
                        "document": stmt.excluded.document,
                        "cmetadata": stmt.excluded.cmetadata,
                    },
                )
                session.execute(on_conflict_stmt)
                session.commit()

        return UpsertResponse(
            succeeded=[],
            failed=[],
        )

    def _create_filter_clause_with_uuid(
        self, collection_uuid: str, filter: Optional[Dict[str, str]]
    ):
        """Create filter clause for the query."""
        filter_by = [self.EmbeddingStore.collection_id == collection_uuid]
        if filter:
            filter_clauses = self._create_filter_clause(filter)
            if filter_clauses is not None:
                filter_by.append(filter_clauses)
        return filter_by

    def _handle_field_filter(
        self,
        field: str,
        value: Any,
    ) -> SQLColumnExpression:
        """Create a filter for a specific field.

        Args:
            field: name of field
            value: value to filter
                If provided as is then this will be an equality filter
                If provided as a dictionary then this will be a filter, the key
                will be the operator and the value will be the value to filter by

        Returns:
            sqlalchemy expression
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

        if operator in COMPARISONS_TO_NATIVE:
            # Then we implement an equality filter
            # native is trusted input
            native = COMPARISONS_TO_NATIVE[operator]
            return func.jsonb_path_match(
                self.EmbeddingStore.cmetadata,
                cast(f"$.{field} {native} $value", JSONPATH),
                cast({"value": filter_value}, JSONB),
            )
        elif operator == "$between":
            # Use AND with two comparisons
            low, high = filter_value

            lower_bound = func.jsonb_path_match(
                self.EmbeddingStore.cmetadata,
                cast(f"$.{field} >= $value", JSONPATH),
                cast({"value": low}, JSONB),
            )
            upper_bound = func.jsonb_path_match(
                self.EmbeddingStore.cmetadata,
                cast(f"$.{field} <= $value", JSONPATH),
                cast({"value": high}, JSONB),
            )
            return sqlalchemy.and_(lower_bound, upper_bound)
        elif operator in {"$in", "$nin", "$like", "$ilike"}:
            # We'll do force coercion to text
            if operator in {"$in", "$nin"}:
                for val in filter_value:
                    if not isinstance(val, (str, int, float)):
                        raise NotImplementedError(
                            f"Unsupported type: {type(val)} for value: {val}"
                        )

                    if isinstance(val, bool):  # b/c bool is an instance of int
                        raise NotImplementedError(
                            f"Unsupported type: {type(val)} for value: {val}"
                        )

            queried_field = self.EmbeddingStore.cmetadata[field].astext

            if operator in {"$in"}:
                return queried_field.in_([str(val) for val in filter_value])
            elif operator in {"$nin"}:
                return ~queried_field.in_([str(val) for val in filter_value])
            elif operator in {"$like"}:
                return queried_field.like(filter_value)
            elif operator in {"$ilike"}:
                return queried_field.ilike(filter_value)
            else:
                raise NotImplementedError()
        elif operator == "$exists":
            if not isinstance(filter_value, bool):
                raise ValueError(
                    "Expected a boolean value for $exists "
                    f"operator, but got: {filter_value}"
                )
            condition = func.jsonb_exists(
                self.EmbeddingStore.cmetadata,
                field,
            )
            return condition if filter_value else ~condition
        else:
            raise NotImplementedError()

    def _create_filter_clause(self, filters: Any) -> Any:
        """Convert LangChain IR filter representation to matching SQLAlchemy clauses.

        At the top level, we still don't know if we're working with a field
        or an operator for the keys. After we've determined that we can
        call the appropriate logic to handle filter creation.

        Args:
            filters: Dictionary of filters to apply to the query.

        Returns:
            SQLAlchemy clause to apply to the query.
        """
        if isinstance(filters, dict):
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
                    return self._handle_field_filter(key, filters[key])

                if key.lower() == "$and":
                    if not isinstance(value, list):
                        raise ValueError(
                            f"Expected a list, but got {type(value)} for value: {value}"
                        )
                    and_ = [self._create_filter_clause(el) for el in value]
                    if len(and_) > 1:
                        return sqlalchemy.and_(*and_)
                    elif len(and_) == 1:
                        return and_[0]
                    else:
                        raise ValueError(
                            "Invalid filter condition. Expected a dictionary "
                            "but got an empty dictionary"
                        )
                elif key.lower() == "$or":
                    if not isinstance(value, list):
                        raise ValueError(
                            f"Expected a list, but got {type(value)} for value: {value}"
                        )
                    or_ = [self._create_filter_clause(el) for el in value]
                    if len(or_) > 1:
                        return sqlalchemy.or_(*or_)
                    elif len(or_) == 1:
                        return or_[0]
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
                        not_ = sqlalchemy.and_(
                            *[
                                sqlalchemy.not_(condition)
                                for condition in not_conditions
                            ]
                        )
                        return not_
                    elif isinstance(value, dict):
                        not_ = self._create_filter_clause(value)
                        return sqlalchemy.not_(not_)
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
                and_ = [self._handle_field_filter(k, v) for k, v in filters.items()]
                if len(and_) > 1:
                    return sqlalchemy.and_(*and_)
                elif len(and_) == 1:
                    return and_[0]
                else:
                    raise ValueError(
                        "Invalid filter condition. Expected a dictionary "
                        "but got an empty dictionary"
                    )
            else:
                raise ValueError("Got an empty dictionary for filters.")
        else:
            raise ValueError(
                f"Invalid type: Expected a dictionary but got type: {type(filters)}"
            )

    def __query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> Sequence[Any]:
        """Query the collection."""
        with self._make_sync_session() as session:  # type: ignore[arg-type]
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            filter_by = [self.EmbeddingStore.collection_id == collection.uuid]
            if filter:
                filter_clauses = self._create_filter_clause(filter)
                if filter_clauses is not None:
                    filter_by.append(filter_clauses)

            _type = self.EmbeddingStore

            results: List[Any] = (
                session.query(
                    self.EmbeddingStore,
                    self.distance_strategy(embedding).label("distance"),
                )
                .filter(*filter_by)
                .order_by(sqlalchemy.asc("distance"))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
                .limit(k)
                .all()
            )

        return results

    async def __aquery_collection(
        self,
        session: AsyncSession,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
    ) -> Sequence[Any]:
        """Query the collection."""
        async with self._make_async_session() as session:  # type: ignore[arg-type]
            collection = await self.aget_collection(session)
            if not collection:
                raise ValueError("Collection not found")

            _type = self.EmbeddingStore
            filter_by = self._create_filter_clause_with_uuid(collection.uuid, filter)

            stmt = (
                select(
                    self.EmbeddingStore,
                    self.distance_strategy(embedding).label("distance"),
                )
                .filter(*filter_by)
                .order_by(sqlalchemy.asc("distance"))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
                .limit(k)
            )

            results: Sequence[Any] = (await session.execute(stmt)).all()

            return results

    @contextlib.contextmanager
    def _make_sync_session(self) -> Generator[Session, None, None]:
        """Make an async session."""
        if self.async_mode:
            raise ValueError(
                "Attempting to use a sync method in when async mode is turned on. "
                "Please use the corresponding async method instead."
            )
        with self.session_maker() as session:
            yield typing_cast(Session, session)

    @contextlib.asynccontextmanager
    async def _make_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Make an async session."""
        if not self.async_mode:
            raise ValueError(
                "Attempting to use an async method in when sync mode is turned on. "
                "Please use the corresponding async method instead."
            )
        async with self.session_maker() as session:
            yield typing_cast(AsyncSession, session)

    ## TODO: I don't want to implement this
    def yield_keys(
        self, *, prefix: Optional[str] = None
    ) -> Union[Iterator[K], Iterator[str]]:
        """This should not be required to be implemented"""
        raise NotImplementedError()
