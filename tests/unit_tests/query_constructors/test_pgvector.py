import os
import re
from typing import Dict, Tuple

import pytest as pytest
import sqlalchemy
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from langchain_postgres import PGVectorTranslator, PGVector, EmbeddingIndexType, vectorstores
from langchain_postgres.vectorstores import _get_embedding_collection_store, IterativeScan
from tests.unit_tests.test_vectorstore import FakeEmbeddingsWithAdaDimension
from tests.utils import sync_session, async_session

from tests.utils import VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING

from sqlalchemy.orm import (
    declarative_base,
)

DEFAULT_TRANSLATOR = PGVectorTranslator()


EmbeddingStore, CollectionStore = _get_embedding_collection_store()


@pytest.fixture(scope='function')
def drop_tables():
    def drop():
        with sync_session() as session:
            session.execute(text("DROP SCHEMA public CASCADE"))
            session.execute(text("CREATE SCHEMA public"))
            session.execute(text(f"GRANT ALL ON SCHEMA public TO {os.environ.get('POSTGRES_USER', 'langchain')}"))
            session.execute(text("GRANT ALL ON SCHEMA public TO public"))
            session.commit()

            vectorstores._classes = None
            vectorstores.Base = declarative_base()

    drop()

    yield

    drop()


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=1)
    expected = {"foo": {"$lt": 1}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


@pytest.mark.skip("Not implemented")
def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.GT, attribute="abc", value=2.0),
        ],
    )
    expected = {
        "foo": {"$lt": 2},
        "bar": {"$eq": "baz"},
        "abc": {"$gt": 2.0},
    }
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


def test_visit_structured_query() -> None:
    query = "What is the capital of France?"
    structured_query = StructuredQuery(
        query=query,
        filter=None,
    )
    expected: Tuple[str, Dict] = (query, {})
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=1)
    structured_query = StructuredQuery(
        query=query,
        filter=comp,
    )
    expected = (query, {"filter": {"foo": {"$lt": 1}}})
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.GT, attribute="abc", value=2.0),
        ],
    )
    structured_query = StructuredQuery(
        query=query,
        filter=op,
    )
    expected = (
        query,
        {
            "filter": {
                "$and": [
                    {"foo": {"$lt": 2}},
                    {"bar": {"$eq": "baz"}},
                    {"abc": {"$gt": 2.0}},
                ]
            }
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_embedding_index_without_length():
    with pytest.raises(ValueError):
        PGVector(
            collection_name="test_collection",
            embeddings=FakeEmbeddingsWithAdaDimension(),
            connection=CONNECTION_STRING,
            embedding_index=EmbeddingIndexType.hnsw,
            embedding_index_ops="vector_cosine_ops",
        )


def test_embedding_index_without_ops():
    with pytest.raises(ValueError):
        PGVector(
            collection_name="test_collection",
            embeddings=FakeEmbeddingsWithAdaDimension(),
            connection=CONNECTION_STRING,
            embedding_length=1536,
            embedding_index=EmbeddingIndexType.hnsw,
        )


@pytest.mark.usefixtures("drop_tables")
def test_embedding_index_hnsw():
    PGVector(
        collection_name="test_collection",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_length=1536,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
    )

    with sync_session() as session:
        result = session.execute(text(f"""
            SELECT indexdef
            FROM pg_indexes
            WHERE tablename = '{EmbeddingStore.__tablename__}' AND indexname = 'ix_embedding_hnsw'
        """))

        assert result.fetchone()[0] == f"CREATE INDEX ix_embedding_hnsw ON public.{EmbeddingStore.__tablename__} USING hnsw (embedding vector_cosine_ops)"


@pytest.mark.asyncio
@pytest.mark.usefixtures("drop_tables")
async def test_embedding_index_hnsw_async():
    pgvector = PGVector(
        collection_name="test_collection",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        embedding_length=1536,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        connection=create_async_engine(CONNECTION_STRING),
        async_mode=True
    )

    await pgvector.__apost_init__()

    async with async_session() as session:
        result = await session.execute(text(f"""
            SELECT indexdef
            FROM pg_indexes
            WHERE tablename = '{EmbeddingStore.__tablename__}' AND indexname = 'ix_embedding_hnsw'
        """))

        assert result.fetchone()[0] == f"CREATE INDEX ix_embedding_hnsw ON public.{EmbeddingStore.__tablename__} USING hnsw (embedding vector_cosine_ops)"


@pytest.mark.usefixtures("drop_tables")
def test_embedding_index_ivfflat():
    PGVector(
        collection_name="test_collection",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_length=1536,
        embedding_index=EmbeddingIndexType.ivfflat,
        embedding_index_ops="vector_cosine_ops",
    )

    with sync_session() as session:
        result = session.execute(text(f"""
            SELECT indexdef
            FROM pg_indexes
            WHERE tablename = '{EmbeddingStore.__tablename__}' AND indexname = 'ix_embedding_ivfflat'
        """))

        assert result.fetchone()[0] == f"CREATE INDEX ix_embedding_ivfflat ON public.{EmbeddingStore.__tablename__} USING ivfflat (embedding vector_cosine_ops)"


@pytest.mark.asyncio
@pytest.mark.usefixtures("drop_tables")
async def test_embedding_index_ivfflat_async():
    pgvector = PGVector(
        collection_name="test_collection",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        embedding_length=1536,
        embedding_index=EmbeddingIndexType.ivfflat,
        embedding_index_ops="vector_cosine_ops",
        connection=create_async_engine(CONNECTION_STRING),
        async_mode=True
    )

    await pgvector.__apost_init__()

    async with async_session() as session:
        result = await session.execute(text(f"""
            SELECT indexdef
            FROM pg_indexes
            WHERE tablename = '{EmbeddingStore.__tablename__}' AND indexname = 'ix_embedding_ivfflat'
        """))

        assert result.fetchone()[0] == f"CREATE INDEX ix_embedding_ivfflat ON public.{EmbeddingStore.__tablename__} USING ivfflat (embedding vector_cosine_ops)"


def test_binary_quantization_without_hnsw():
    with pytest.raises(ValueError):
        PGVector(
            collection_name="test_collection",
            embeddings=FakeEmbeddingsWithAdaDimension(),
            connection=CONNECTION_STRING,
            embedding_length=1536,
            embedding_index=EmbeddingIndexType.ivfflat,
            embedding_index_ops="bit_hamming_ops",
            binary_quantization=True,
            binary_limit=200
        )


def test_binary_quantization_without_hamming_ops():
    with pytest.raises(ValueError):
        PGVector(
            collection_name="test_collection",
            embeddings=FakeEmbeddingsWithAdaDimension(),
            connection=CONNECTION_STRING,
            embedding_length=1536,
            embedding_index=EmbeddingIndexType.hnsw,
            binary_quantization=True,
            embedding_index_ops="vector_cosine_ops",
            binary_limit=200
        )


def test_binary_quantization_without_binary_limit():
    with pytest.raises(ValueError):
        PGVector(
            collection_name="test_collection",
            embeddings=FakeEmbeddingsWithAdaDimension(),
            connection=CONNECTION_STRING,
            embedding_length=1536,
            embedding_index=EmbeddingIndexType.hnsw,
            embedding_index_ops="bit_hamming_ops",
            binary_quantization=True,
        )


@pytest.mark.usefixtures("drop_tables")
def test_binary_quantization_index():
    PGVector(
        collection_name="test_collection",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_length=1536,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="bit_hamming_ops",
        binary_quantization=True,
        binary_limit=200
    )

    with sync_session() as session:
        result = session.execute(text(f"""
            SELECT indexdef
            FROM pg_indexes
            WHERE tablename = '{EmbeddingStore.__tablename__}' AND indexname = 'ix_embedding_hnsw'
        """))

        assert result.fetchone()[0] == f"CREATE INDEX ix_embedding_hnsw ON public.{EmbeddingStore.__tablename__} USING hnsw (((binary_quantize(embedding))::bit(1536)) bit_hamming_ops)"


def test_ef_construction_without_hnsw():
    with pytest.raises(ValueError):
        PGVector(
            collection_name="test_collection",
            embeddings=FakeEmbeddingsWithAdaDimension(),
            connection=CONNECTION_STRING,
            embedding_length=1536,
            embedding_index=EmbeddingIndexType.ivfflat,
            embedding_index_ops="vector_cosine_ops",
            ef_construction=256,
        )


def test_m_without_hnsw():
    with pytest.raises(ValueError):
        PGVector(
            collection_name="test_collection",
            embeddings=FakeEmbeddingsWithAdaDimension(),
            connection=CONNECTION_STRING,
            embedding_length=1536,
            embedding_index=EmbeddingIndexType.ivfflat,
            embedding_index_ops="vector_cosine_ops",
            m=16,
        )


@pytest.mark.usefixtures("drop_tables")
def test_ef_construction():
    PGVector(
        collection_name="test_collection",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_length=1536,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        ef_construction=256,
    )

    with sync_session() as session:
        result = session.execute(text(f"""
            SELECT indexdef
            FROM pg_indexes
            WHERE tablename = '{EmbeddingStore.__tablename__}' AND indexname = 'ix_embedding_hnsw'
        """))

        assert result.fetchone()[0] == f"CREATE INDEX ix_embedding_hnsw ON public.{EmbeddingStore.__tablename__} USING hnsw (embedding vector_cosine_ops) WITH (ef_construction='256')"


@pytest.mark.usefixtures("drop_tables")
def test_m():
    PGVector(
        collection_name="test_collection",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_length=1536,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        m=16,
    )

    with sync_session() as session:
        result = session.execute(text(f"""
            SELECT indexdef
            FROM pg_indexes
            WHERE tablename = '{EmbeddingStore.__tablename__}' AND indexname = 'ix_embedding_hnsw'
        """))

        assert result.fetchone()[0] == f"CREATE INDEX ix_embedding_hnsw ON public.{EmbeddingStore.__tablename__} USING hnsw (embedding vector_cosine_ops) WITH (m='16')"


@pytest.mark.usefixtures("drop_tables")
def test_ef_construction_and_m():
    PGVector(
        collection_name="test_collection",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_length=1536,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        ef_construction=256,
        m=16,
    )

    with sync_session() as session:
        result = session.execute(text(f"""
            SELECT indexdef
            FROM pg_indexes
            WHERE tablename = '{EmbeddingStore.__tablename__}' AND indexname = 'ix_embedding_hnsw'
        """))

        assert result.fetchone()[0] == f"CREATE INDEX ix_embedding_hnsw ON public.{EmbeddingStore.__tablename__} USING hnsw (embedding vector_cosine_ops) WITH (m='16', ef_construction='256')"


@pytest.mark.usefixtures("drop_tables")
def test_binary_quantization_with_ef_construction_and_m():
    PGVector(
        collection_name="test_collection",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_length=1536,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="bit_hamming_ops",
        binary_quantization=True,
        binary_limit=200,
        ef_construction=256,
        m=16
    )

    with sync_session() as session:
        result = session.execute(text(f"""
            SELECT indexdef
            FROM pg_indexes
            WHERE tablename = '{EmbeddingStore.__tablename__}' AND indexname = 'ix_embedding_hnsw'
        """))

        assert result.fetchone()[0] == f"CREATE INDEX ix_embedding_hnsw ON public.{EmbeddingStore.__tablename__} USING hnsw (((binary_quantize(embedding))::bit(1536)) bit_hamming_ops) WITH (m='16', ef_construction='256')"


get_partitioned_table = text(f"""
    SELECT 
        c.relname AS table_name,
        p.partstrat AS partition_strategy,
        pg_attribute.attname AS partition_key
    FROM 
        pg_class c
    JOIN 
        pg_partitioned_table p ON c.oid = p.partrelid
    LEFT JOIN 
        pg_attribute ON pg_attribute.attrelid = p.partrelid AND pg_attribute.attnum = ANY(p.partattrs)
    WHERE 
        c.relname = '{EmbeddingStore.__tablename__}'
""")


@pytest.mark.usefixtures("drop_tables")
def test_partitioning():
    PGVector(
        collection_name="test_collection",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_length=1536,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        enable_partitioning=True
    )

    with sync_session() as session:
        result = session.execute(get_partitioned_table)

        assert result.fetchone() == (EmbeddingStore.__tablename__, 'l', 'collection_id')

        result = session.execute(text(f"""
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = '{EmbeddingStore.__tablename__}'
        """))

        assert result.scalars().fetchall() == ['langchain_pg_embedding_pkey', 'ix_cmetadata_gin', 'ix_document_vector_gin', 'ix_embedding_hnsw']


@pytest.mark.asyncio
@pytest.mark.usefixtures("drop_tables")
async def test_partitioning_async():
    pgvector = PGVector(
        collection_name="test_collection",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=create_async_engine(CONNECTION_STRING),
        embedding_length=1536,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        enable_partitioning=True,
        async_mode=True
    )

    await pgvector.__apost_init__()

    async with async_session() as session:
        result = await session.execute(get_partitioned_table)

        assert result.fetchone() == (EmbeddingStore.__tablename__, 'l', 'collection_id')

        result = await session.execute(text(f"""
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = '{EmbeddingStore.__tablename__}'
        """))

        assert result.scalars().fetchall() == ['langchain_pg_embedding_pkey', 'ix_cmetadata_gin', 'ix_document_vector_gin', 'ix_embedding_hnsw']


get_partitions = text(f"""
    SELECT c.relname as partitioned_table_name,
           pg_get_expr(c.relpartbound, c.oid) as partition_bound
    FROM pg_class c
    JOIN pg_inherits i ON c.oid = i.inhrelid
    JOIN pg_class pc ON i.inhparent = pc.oid
    WHERE pc.relname = '{EmbeddingStore.__tablename__}'
""")


@pytest.mark.usefixtures("drop_tables")
def test_partitions_creation():
    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        enable_partitioning=True,
    )

    with sync_session() as session:
        collection1 = pgvector.get_collection(session)

    pgvector = PGVector(
        collection_name="test_collection2",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        enable_partitioning=True,
    )

    with sync_session() as session:
        collection2 = pgvector.get_collection(session)

    with sync_session() as session:
        result = session.execute(get_partitions)

        assert result.fetchall() == [
            (f"{EmbeddingStore.__tablename__}_{str(collection1.uuid).replace('-', '_')}", f"FOR VALUES IN ('{str(collection1.uuid)}')"),
            (f"{EmbeddingStore.__tablename__}_{str(collection2.uuid).replace('-', '_')}", f"FOR VALUES IN ('{str(collection2.uuid)}')")
        ]


@pytest.mark.asyncio
@pytest.mark.usefixtures("drop_tables")
async def test_partitions_creation_async():
    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=create_async_engine(CONNECTION_STRING),
        enable_partitioning=True,
        async_mode=True
    )
    await pgvector.__apost_init__()

    async with async_session() as session:
        collection1 = await pgvector.aget_collection(session)

    pgvector = PGVector(
        collection_name="test_collection2",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=create_async_engine(CONNECTION_STRING),
        enable_partitioning=True,
        async_mode=True
    )
    await pgvector.__apost_init__()

    async with async_session() as session:
        collection2 = await pgvector.aget_collection(session)

    async with async_session() as session:
        result = await session.execute(get_partitions)

        assert result.fetchall() == [
            (f"{EmbeddingStore.__tablename__}_{str(collection1.uuid).replace('-', '_')}", f"FOR VALUES IN ('{str(collection1.uuid)}')"),
            (f"{EmbeddingStore.__tablename__}_{str(collection2.uuid).replace('-', '_')}", f"FOR VALUES IN ('{str(collection2.uuid)}')")
        ]


@pytest.mark.usefixtures("drop_tables")
def test_partitions_deletion():
    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        enable_partitioning=True,
    )

    pgvector.delete_collection()

    with sync_session() as session:
        result = session.execute(get_partitions)

        assert result.fetchall() == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("drop_tables")
async def test_partitions_deletion_async():
    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=create_async_engine(CONNECTION_STRING),
        enable_partitioning=True,
        async_mode=True
    )
    await pgvector.__apost_init__()

    await pgvector.adelete_collection()

    async with async_session() as session:
        result = await session.execute(get_partitions)

        assert result.fetchall() == []


def test_iterative_scan_without_index():
    with pytest.raises(ValueError):
        PGVector(
            collection_name="test_collection1",
            embeddings=FakeEmbeddingsWithAdaDimension(),
            connection=CONNECTION_STRING,
            iterative_scan=IterativeScan.off
        )


@pytest.mark.usefixtures("drop_tables")
def test_iterative_scan():
    with sync_session() as session:
        with pytest.raises(sqlalchemy.exc.ProgrammingError):
            session.execute(text("SHOW hnsw.iterative_scan"))

    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        embedding_length=1536,
        iterative_scan=IterativeScan.off,
    )

    with sync_session() as session:
        pgvector._set_iterative_scan(session)

        result = session.execute(text("SHOW hnsw.iterative_scan"))
        assert result.fetchone()[0] == "off"

    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        embedding_length=1536,
        iterative_scan=IterativeScan.relaxed_order,
    )

    with sync_session() as session:
        pgvector._set_iterative_scan(session)

        result = session.execute(text("SHOW hnsw.iterative_scan"))
        assert result.fetchone()[0] == "relaxed_order"

    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        embedding_length=1536,
        iterative_scan=IterativeScan.strict_order,
    )

    with sync_session() as session:
        pgvector._set_iterative_scan(session)

        result = session.execute(text("SHOW hnsw.iterative_scan"))
        assert result.fetchone()[0] == "strict_order"


@pytest.mark.asyncio
@pytest.mark.usefixtures("drop_tables")
async def test_iterative_scan_async():
    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=create_async_engine(CONNECTION_STRING),
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        embedding_length=1536,
        iterative_scan=IterativeScan.strict_order,
        async_mode=True
    )

    await pgvector.__apost_init__()

    async with async_session() as session:
        await pgvector._aset_iterative_scan(session)

        result = await session.execute(text("SHOW hnsw.iterative_scan"))
        assert result.fetchone()[0] == "strict_order"


def test_max_scan_tuples_without_hnsw():
    with pytest.raises(ValueError):
        PGVector(
            collection_name="test_collection1",
            embeddings=FakeEmbeddingsWithAdaDimension(),
            connection=CONNECTION_STRING,
            max_scan_tuples=200
        )


@pytest.mark.usefixtures("drop_tables")
def test_max_scan_tuples():
    with sync_session() as session:
        with pytest.raises(sqlalchemy.exc.ProgrammingError):
            session.execute(text("SHOW hnsw.max_scan_tuples"))

    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        embedding_length=1536,
        max_scan_tuples=200
    )

    with sync_session() as session:
        pgvector._set_max_scan_tuples(session)

        result = session.execute(text("SHOW hnsw.max_scan_tuples"))
        assert result.fetchone()[0] == "200"


@pytest.mark.asyncio
@pytest.mark.usefixtures("drop_tables")
async def test_max_scan_tuples_async():
    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=create_async_engine(CONNECTION_STRING),
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        embedding_length=1536,
        max_scan_tuples=200,
        async_mode=True
    )

    await pgvector.__apost_init__()

    async with async_session() as session:
        await pgvector._aset_max_scan_tuples(session)

        result = await session.execute(text("SHOW hnsw.max_scan_tuples"))
        assert result.fetchone()[0] == "200"


def test_scan_mem_multiplier_without_hnsw():
    with pytest.raises(ValueError):
        PGVector(
            collection_name="test_collection1",
            embeddings=FakeEmbeddingsWithAdaDimension(),
            connection=CONNECTION_STRING,
            scan_mem_multiplier=200
        )


@pytest.mark.usefixtures("drop_tables")
def test_scan_mem_multiplier():
    with sync_session() as session:
        with pytest.raises(sqlalchemy.exc.ProgrammingError):
            session.execute(text("SHOW hnsw.scan_mem_multiplier"))

    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        embedding_length=1536,
        scan_mem_multiplier=2
    )

    with sync_session() as session:
        pgvector._set_scan_mem_multiplier(session)

        result = session.execute(text("SHOW hnsw.scan_mem_multiplier"))
        assert result.fetchone()[0] == "2"


@pytest.mark.asyncio
@pytest.mark.usefixtures("drop_tables")
async def test_scan_mem_multiplier_async():
    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=create_async_engine(CONNECTION_STRING),
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        embedding_length=1536,
        scan_mem_multiplier=2,
        async_mode=True
    )

    await pgvector.__apost_init__()

    async with async_session() as session:
        await pgvector._aset_scan_mem_multiplier(session)

        result = await session.execute(text("SHOW hnsw.scan_mem_multiplier"))
        assert result.fetchone()[0] == "2"


def normalize_sql(query):
    # Remove new lines, tabs, and multiple spaces
    query = re.sub(r'\s+', ' ', query).strip()
    # Normalize space around commas
    query = re.sub(r'\s*,\s*', ', ', query)
    # Normalize space around parentheses
    query = re.sub(r'\(\s*', '(', query)
    query = re.sub(r'\s*\)', ')', query)
    return query


@pytest.mark.usefixtures("drop_tables")
def test_binary_quantization_query():
    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_length=3,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="bit_hamming_ops",
        binary_quantization=True,
        binary_limit=200
    )

    with sync_session() as session:
        collection = pgvector.get_collection(session)

    stmt = pgvector._build_query_collection(
        collection=collection,
        embedding=[1.0, 0.0, -1.0],
        k=20,
        filter={"test_key": "test_value"},
        full_text_search=["word1", "word2 & word3"],
    )

    compiled = stmt.compile(dialect=sqlalchemy.dialects.postgresql.dialect())

    query = str(compiled)
    params = compiled.params

    expected_query = """
        SELECT
            binary_result.id,
            binary_result.collection_id,
            binary_result.embedding,
            binary_result.document,
            binary_result.cmetadata,
            binary_result.document_vector,
            binary_result.embedding <=> %(embedding_1)s AS distance
        FROM
            (
                SELECT
                    langchain_pg_embedding.id AS id,
                    langchain_pg_embedding.collection_id AS collection_id,
                    langchain_pg_embedding.embedding AS embedding,
                    langchain_pg_embedding.document AS document,
                    langchain_pg_embedding.cmetadata AS cmetadata,
                    langchain_pg_embedding.document_vector AS document_vector
                FROM
                    langchain_pg_embedding
                WHERE
                    langchain_pg_embedding.collection_id = %(collection_id_1)s::UUID
                    AND jsonb_path_match(langchain_pg_embedding.cmetadata, CAST(%(param_1)s AS JSONPATH), CAST(%(param_2)s AS JSONB))
                    AND document_vector @@ to_tsquery('word1 | word2 & word3')
                ORDER BY
                    CAST(binary_quantize(langchain_pg_embedding.embedding) AS BIT(3)) <~> binary_quantize(CAST(%(param_3)s AS VECTOR(3)))
                LIMIT
                    %(param_4)s
            ) AS binary_result
        ORDER BY
            distance ASC
        LIMIT
            %(param_5)s
    """.replace("\n", "").replace("    ", " ")

    assert normalize_sql(query) == normalize_sql(expected_query)
    assert params == {
        'embedding_1': [1.0, 0.0, -1.0],
        'collection_id_1': collection.uuid,
        'param_1': '$.test_key == $value',
        'param_2': {'value': 'test_value'},
        'param_3': [1.0, 0.0, -1.0],
        'param_4': 200,
        'param_5': 20
    }


@pytest.mark.usefixtures("drop_tables")
def test_relaxed_order_query():
    pgvector = PGVector(
        collection_name="test_collection1",
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        embedding_length=3,
        embedding_index=EmbeddingIndexType.hnsw,
        embedding_index_ops="vector_cosine_ops",
        iterative_scan=IterativeScan.relaxed_order
    )

    with sync_session() as session:
        collection = pgvector.get_collection(session)

    stmt = pgvector._build_query_collection(
        collection=collection,
        embedding=[1.0, 0.0, -1.0],
        k=20,
        filter={"test_key": "test_value"},
        full_text_search=["word1", "word2 & word3"],
    )

    compiled = stmt.compile(dialect=sqlalchemy.dialects.postgresql.dialect())

    query = str(compiled)
    params = compiled.params

    expected_query = """
        WITH relaxed_results AS MATERIALIZED (
            SELECT
                langchain_pg_embedding.id AS id,
                langchain_pg_embedding.collection_id AS collection_id,
                langchain_pg_embedding.embedding AS embedding,
                langchain_pg_embedding.document AS document,
                langchain_pg_embedding.cmetadata AS cmetadata,
                langchain_pg_embedding.document_vector AS document_vector,
                langchain_pg_embedding.embedding <=> %(embedding_1)s AS distance
            FROM
                langchain_pg_embedding
            WHERE
                langchain_pg_embedding.collection_id = %(collection_id_1)s::UUID
                AND jsonb_path_match(
                    langchain_pg_embedding.cmetadata,
                    CAST(%(param_1)s AS JSONPATH),
                    CAST(%(param_2)s AS JSONB)
                )
                AND document_vector @@ to_tsquery('word1 | word2 & word3')
            ORDER BY
                distance ASC
            LIMIT
                %(param_3)s
        )
        SELECT
            relaxed_results.id,
            relaxed_results.collection_id,
            relaxed_results.embedding,
            relaxed_results.document,
            relaxed_results.cmetadata,
            relaxed_results.document_vector,
            relaxed_results.distance
        FROM
            relaxed_results
        ORDER BY
            distance
    """

    assert normalize_sql(query) == normalize_sql(expected_query)
    assert params == {
        'embedding_1': [1.0, 0.0, -1.0],
        'collection_id_1': collection.uuid,
        'param_1': '$.test_key == $value',
        'param_2': {'value': 'test_value'},
        'param_3': 20
    }