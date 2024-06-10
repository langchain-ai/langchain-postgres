"""Test PGVector functionality."""
import contextlib
from typing import Any, AsyncGenerator, Dict, Generator, List

import pytest
from langchain_core.documents import Document
from sqlalchemy import select

from langchain_postgres.vectorstores import (
    SUPPORTED_OPERATORS,
    PGVector,
)
from tests.unit_tests.fake_embeddings import FakeEmbeddings
from tests.unit_tests.fixtures.filtering_test_cases import (
    DOCUMENTS,
    TYPE_1_FILTERING_TEST_CASES,
    TYPE_2_FILTERING_TEST_CASES,
    TYPE_3_FILTERING_TEST_CASES,
    TYPE_4_FILTERING_TEST_CASES,
    TYPE_5_FILTERING_TEST_CASES,
)
from tests.utils import VECTORSTORE_CONNECTION_STRING as CONNECTION_STRING

ADA_TOKEN_COUNT = 1536


class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(0.0)]


def test_pgvector() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_async_pgvector() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_pgvector_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = PGVector.from_embeddings(
        text_embeddings=text_embedding_pairs,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_async_pgvector_embeddings() -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = await PGVector.afrom_embeddings(
        text_embeddings=text_embedding_pairs,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_pgvector_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


@pytest.mark.asyncio
async def test_async_pgvector_with_metadatas() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_pgvector_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


@pytest.mark.asyncio
async def test_async_pgvector_with_metadatas_with_scores() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = await docsearch.asimilarity_search_with_score("foo", k=1)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_pgvector_with_filter_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "0"})
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


@pytest.mark.asyncio
async def test_async_pgvector_with_filter_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = await docsearch.asimilarity_search_with_score(
        "foo", k=1, filter={"page": "0"}
    )
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]


def test_pgvector_with_filter_distant_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "2"})
    assert output == [
        (Document(page_content="baz", metadata={"page": "2"}), 0.0013003906671379406)
    ]


@pytest.mark.asyncio
async def test_async_pgvector_with_filter_distant_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = await docsearch.asimilarity_search_with_score(
        "foo", k=1, filter={"page": "2"}
    )
    assert output == [
        (Document(page_content="baz", metadata={"page": "2"}), 0.0013003906671379406)
    ]


def test_pgvector_with_filter_no_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "5"})
    assert output == []


@pytest.mark.asyncio
async def test_async_pgvector_with_filter_no_match() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = await docsearch.asimilarity_search_with_score(
        "foo", k=1, filter={"page": "5"}
    )
    assert output == []


def test_pgvector_collection_with_metadata() -> None:
    """Test end to end collection construction"""
    pgvector = PGVector(
        collection_name="test_collection",
        collection_metadata={"foo": "bar"},
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    with pgvector.session_maker() as session:
        collection = pgvector.get_collection(session)
        if collection is None:
            assert False, "Expected a CollectionStore object but received None"
        else:
            assert collection.name == "test_collection"
            assert collection.cmetadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_async_pgvector_collection_with_metadata() -> None:
    """Test end to end collection construction"""
    pgvector = PGVector(
        collection_name="test_collection",
        collection_metadata={"foo": "bar"},
        embeddings=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
        async_mode=True,
    )
    async with pgvector.session_maker() as session:
        collection = await pgvector.aget_collection(session)
        if collection is None:
            assert False, "Expected a CollectionStore object but received None"
        else:
            assert collection.name == "test_collection"
            assert collection.cmetadata == {"foo": "bar"}


def test_pgvector_delete_docs() -> None:
    """Add and delete documents."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    vectorstore = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        ids=["1", "2", "3"],
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    vectorstore.delete(["1", "2"])
    with vectorstore.session_maker() as session:
        records = list(session.query(vectorstore.EmbeddingStore).all())
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert sorted(record.id for record in records) == ["3"]  # type: ignore

    vectorstore.delete(["2", "3"])  # Should not raise on missing ids
    with vectorstore.session_maker() as session:
        records = list(session.query(vectorstore.EmbeddingStore).all())
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert sorted(record.id for record in records) == []  # type: ignore


def test_pgvector_delete_collection() -> None:
    """Add and delete documents."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    vectorstore = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        ids=["1", "2", "3"],
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    vectorstore.delete(collection_only=True)


@pytest.mark.asyncio
async def test_async_pgvector_delete_docs() -> None:
    """Add and delete documents."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    vectorstore = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        ids=["1", "2", "3"],
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    await vectorstore.adelete(["1", "2"])
    async with vectorstore.session_maker() as session:
        records = (
            (await session.execute(select(vectorstore.EmbeddingStore))).scalars().all()
        )
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert sorted(record.id for record in records) == ["3"]  # type: ignore

    await vectorstore.adelete(["2", "3"])  # Should not raise on missing ids
    async with vectorstore.session_maker() as session:
        records = (
            (await session.execute(select(vectorstore.EmbeddingStore))).scalars().all()
        )
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert sorted(record.id for record in records) == []  # type: ignore


def test_pgvector_index_documents() -> None:
    """Test adding duplicate documents results in overwrites."""
    documents = [
        Document(
            page_content="there are cats in the pond",
            metadata={"id": 1, "location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"id": 2, "location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="fresh apples are available at the market",
            metadata={"id": 3, "location": "market", "topic": "food"},
        ),
        Document(
            page_content="the market also sells fresh oranges",
            metadata={"id": 4, "location": "market", "topic": "food"},
        ),
        Document(
            page_content="the new art exhibit is fascinating",
            metadata={"id": 5, "location": "museum", "topic": "art"},
        ),
    ]

    vectorstore = PGVector.from_documents(
        documents=documents,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        ids=[doc.metadata["id"] for doc in documents],
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    with vectorstore.session_maker() as session:
        records = list(session.query(vectorstore.EmbeddingStore).all())
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert sorted(record.id for record in records) == [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]

    # Try to overwrite the first document
    documents = [
        Document(
            page_content="new content in the zoo",
            metadata={"id": 1, "location": "zoo", "topic": "zoo"},
        ),
    ]

    vectorstore.add_documents(documents, ids=[doc.metadata["id"] for doc in documents])

    with vectorstore.session_maker() as session:
        records = list(session.query(vectorstore.EmbeddingStore).all())
        ordered_records = sorted(records, key=lambda x: x.id)
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert [record.id for record in ordered_records] == [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]

        assert ordered_records[0].cmetadata == {
            "id": 1,
            "location": "zoo",
            "topic": "zoo",
        }


@pytest.mark.asyncio
async def test_async_pgvector_index_documents() -> None:
    """Test adding duplicate documents results in overwrites."""
    documents = [
        Document(
            page_content="there are cats in the pond",
            metadata={"id": 1, "location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"id": 2, "location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="fresh apples are available at the market",
            metadata={"id": 3, "location": "market", "topic": "food"},
        ),
        Document(
            page_content="the market also sells fresh oranges",
            metadata={"id": 4, "location": "market", "topic": "food"},
        ),
        Document(
            page_content="the new art exhibit is fascinating",
            metadata={"id": 5, "location": "museum", "topic": "art"},
        ),
    ]

    vectorstore = await PGVector.afrom_documents(
        documents=documents,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        ids=[doc.metadata["id"] for doc in documents],
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    async with vectorstore.session_maker() as session:
        records = (
            (await session.execute(select(vectorstore.EmbeddingStore))).scalars().all()
        )
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert sorted(record.id for record in records) == [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]

    # Try to overwrite the first document
    documents = [
        Document(
            page_content="new content in the zoo",
            metadata={"id": 1, "location": "zoo", "topic": "zoo"},
        ),
    ]

    await vectorstore.aadd_documents(
        documents, ids=[doc.metadata["id"] for doc in documents]
    )

    async with vectorstore.session_maker() as session:
        records = (
            (await session.execute(select(vectorstore.EmbeddingStore))).scalars().all()
        )
        ordered_records = sorted(records, key=lambda x: x.id)
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert [record.id for record in ordered_records] == [
            "1",
            "2",
            "3",
            "4",
            "5",
        ]

        assert ordered_records[0].cmetadata == {
            "id": 1,
            "location": "zoo",
            "topic": "zoo",
        }


def test_pgvector_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9996744261675065),
        (Document(page_content="baz", metadata={"page": "2"}), 0.9986996093328621),
    ]


@pytest.mark.asyncio
async def test_async_pgvector_relevance_score() -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    output = await docsearch.asimilarity_search_with_relevance_scores("foo", k=3)
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.9996744261675065),
        (Document(page_content="baz", metadata={"page": "2"}), 0.9986996093328621),
    ]


def test_pgvector_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.999},
    )
    output = retriever.get_relevant_documents("summer")
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
    ]


@pytest.mark.asyncio
async def test_async_pgvector_retriever_search_threshold() -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.999},
    )
    output = await retriever.aget_relevant_documents("summer")
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
    ]


def test_pgvector_retriever_search_threshold_custom_normalization_fn() -> None:
    """Test searching with threshold and custom normalization function"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
        relevance_score_fn=lambda d: d * 0,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    output = retriever.get_relevant_documents("foo")
    assert output == []


@pytest.mark.asyncio
async def test_async_pgvector_retriever_search_threshold_custom_normalization_fn() -> (
    None
):
    """Test searching with threshold and custom normalization function"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
        relevance_score_fn=lambda d: d * 0,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    output = await retriever.aget_relevant_documents("foo")
    assert output == []


def test_pgvector_max_marginal_relevance_search() -> None:
    """Test max marginal relevance search."""
    texts = ["foo", "bar", "baz"]
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.max_marginal_relevance_search("foo", k=1, fetch_k=3)
    assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_async_pgvector_max_marginal_relevance_search() -> None:
    """Test max marginal relevance search."""
    texts = ["foo", "bar", "baz"]
    docsearch = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = await docsearch.amax_marginal_relevance_search("foo", k=1, fetch_k=3)
    assert output == [Document(page_content="foo")]


def test_pgvector_max_marginal_relevance_search_with_score() -> None:
    """Test max marginal relevance search with relevance scores."""
    texts = ["foo", "bar", "baz"]
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.max_marginal_relevance_search_with_score("foo", k=1, fetch_k=3)
    assert output == [(Document(page_content="foo"), 0.0)]


@pytest.mark.asyncio
async def test_async_pgvector_max_marginal_relevance_search_with_score() -> None:
    """Test max marginal relevance search with relevance scores."""
    texts = ["foo", "bar", "baz"]
    docsearch = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = await docsearch.amax_marginal_relevance_search_with_score(
        "foo", k=1, fetch_k=3
    )
    assert output == [(Document(page_content="foo"), 0.0)]


def test_pgvector_with_custom_connection() -> None:
    """Test construction using a custom connection."""
    texts = ["foo", "bar", "baz"]
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


@pytest.mark.asyncio
async def test_async_pgvector_with_custom_connection() -> None:
    """Test construction using a custom connection."""
    texts = ["foo", "bar", "baz"]
    docsearch = await PGVector.afrom_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    output = await docsearch.asimilarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_pgvector_with_custom_engine_args() -> None:
    """Test construction using custom engine arguments."""
    texts = ["foo", "bar", "baz"]
    engine_args = {
        "pool_size": 5,
        "max_overflow": 10,
        "pool_recycle": -1,
        "pool_use_lifo": False,
        "pool_pre_ping": False,
        "pool_timeout": 30,
    }
    docsearch = PGVector.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
        engine_args=engine_args,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


# We should reuse this test-case across other integrations
# Add database fixture using pytest
@pytest.fixture
def pgvector() -> Generator[PGVector, None, None]:
    """Create a PGVector instance."""
    with get_vectorstore() as vector_store:
        yield vector_store


@pytest.mark.asyncio
@pytest.fixture
async def async_pgvector() -> AsyncGenerator[PGVector, None]:
    """Create an async PGVector instance."""
    store = await PGVector.afrom_documents(
        documents=DOCUMENTS,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
        relevance_score_fn=lambda d: d * 0,
        use_jsonb=True,
    )
    try:
        yield store
    # Do clean up
    finally:
        await store.adrop_tables()


@contextlib.contextmanager
def get_vectorstore() -> Generator[PGVector, None, None]:
    """Get a pre-populated-vectorstore"""
    store = PGVector.from_documents(
        documents=DOCUMENTS,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
        relevance_score_fn=lambda d: d * 0,
        use_jsonb=True,
    )
    try:
        yield store
    finally:
        store.drop_tables()


@contextlib.asynccontextmanager
async def aget_vectorstore() -> AsyncGenerator[PGVector, None]:
    """Get a pre-populated-vectorstore"""
    store = await PGVector.afrom_documents(
        documents=DOCUMENTS,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
        relevance_score_fn=lambda d: d * 0,
        use_jsonb=True,
    )
    try:
        yield store
    finally:
        await store.adrop_tables()


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_1_FILTERING_TEST_CASES)
def test_pgvector_with_with_metadata_filters_1(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    with get_vectorstore() as pgvector:
        docs = pgvector.similarity_search("meow", k=5, filter=test_filter)
        assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.asyncio
@pytest.mark.parametrize("test_filter, expected_ids", TYPE_1_FILTERING_TEST_CASES)
async def test_async_pgvector_with_with_metadata_filters_1(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    async with aget_vectorstore() as pgvector:
        docs = await pgvector.asimilarity_search("meow", k=5, filter=test_filter)
        assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_2_FILTERING_TEST_CASES)
def test_pgvector_with_with_metadata_filters_2(
    pgvector: PGVector,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = pgvector.similarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.asyncio
@pytest.mark.parametrize("test_filter, expected_ids", TYPE_2_FILTERING_TEST_CASES)
async def test_async_pgvector_with_with_metadata_filters_2(
    async_pgvector: PGVector,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = await async_pgvector.asimilarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_3_FILTERING_TEST_CASES)
def test_pgvector_with_with_metadata_filters_3(
    pgvector: PGVector,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = pgvector.similarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.asyncio
@pytest.mark.parametrize("test_filter, expected_ids", TYPE_3_FILTERING_TEST_CASES)
async def test_async_pgvector_with_with_metadata_filters_3(
    async_pgvector: PGVector,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = await async_pgvector.asimilarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_4_FILTERING_TEST_CASES)
def test_pgvector_with_with_metadata_filters_4(
    pgvector: PGVector,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = pgvector.similarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.asyncio
@pytest.mark.parametrize("test_filter, expected_ids", TYPE_4_FILTERING_TEST_CASES)
async def test_async_pgvector_with_with_metadata_filters_4(
    async_pgvector: PGVector,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = await async_pgvector.asimilarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_5_FILTERING_TEST_CASES)
def test_pgvector_with_with_metadata_filters_5(
    pgvector: PGVector,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = pgvector.similarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.asyncio
@pytest.mark.parametrize("test_filter, expected_ids", TYPE_5_FILTERING_TEST_CASES)
async def test_async_pgvector_with_with_metadata_filters_5(
    async_pgvector: PGVector,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = await async_pgvector.asimilarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize(
    "invalid_filter",
    [
        ["hello"],
        {
            "id": 2,
            "$name": "foo",
        },
        {"$or": {}},
        {"$and": {}},
        {"$between": {}},
        {"$eq": {}},
        {"$exists": {}},
        {"$exists": 1},
        {"$not": 2},
    ],
)
def test_invalid_filters(pgvector: PGVector, invalid_filter: Any) -> None:
    """Verify that invalid filters raise an error."""
    with pytest.raises(ValueError):
        pgvector._create_filter_clause(invalid_filter)


def test_validate_operators() -> None:
    """Verify that all operators have been categorized."""
    assert sorted(SUPPORTED_OPERATORS) == [
        "$and",
        "$between",
        "$eq",
        "$exists",
        "$gt",
        "$gte",
        "$ilike",
        "$in",
        "$like",
        "$lt",
        "$lte",
        "$ne",
        "$nin",
        "$or",
        "$not",
    ]
