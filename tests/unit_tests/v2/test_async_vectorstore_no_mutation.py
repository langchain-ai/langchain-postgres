"""Regression tests pinning that `AsyncPGVectorStore.asimilarity_search*`
methods do not mutate the instance/caller-provided `hybrid_search_config`.

Without copying the config before back-filling `fts_query`, the very first
search query "sticks" on the shared `HybridSearchConfig` object and every
later call reuses it. Verified against #288.

These tests bypass `AsyncPGVectorStore.create(...)` (which needs a real
Postgres engine) by stubbing the few attributes the relevant methods touch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock

import pytest

from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig


@dataclass
class _FakeEmbeddingService:
    """Minimal async embedder — emits a fixed dummy vector per call."""

    vector: list[float]

    async def aembed_query(self, text: str) -> list[float]:
        return list(self.vector)


def _make_store(hybrid_config: HybridSearchConfig) -> AsyncPGVectorStore:
    """Build a barebones `AsyncPGVectorStore` without touching the DB."""
    store = object.__new__(AsyncPGVectorStore)
    store.embedding_service = _FakeEmbeddingService(vector=[0.1, 0.2, 0.3])
    store.hybrid_search_config = hybrid_config
    store.asimilarity_search_by_vector = AsyncMock(return_value=[])
    store.asimilarity_search_with_score_by_vector = AsyncMock(return_value=[])
    return store


@pytest.mark.asyncio
async def test_asimilarity_search_does_not_mutate_instance_hybrid_search_config() -> (
    None
):
    """First call's `query` must not leak into the instance config's
    `fts_query`, so subsequent searches with different queries see their
    own values."""
    config = HybridSearchConfig(fts_query="")
    store = _make_store(config)

    await store.asimilarity_search("first query")
    await store.asimilarity_search("second query")

    # Instance-level config still has its original empty fts_query.
    assert config.fts_query == ""
    # And each downstream call got its own config copy with the right query.
    calls = store.asimilarity_search_by_vector.await_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["hybrid_search_config"].fts_query == "first query"
    assert calls[1].kwargs["hybrid_search_config"].fts_query == "second query"
    # The instance config object is not the one passed downstream.
    assert calls[0].kwargs["hybrid_search_config"] is not config


@pytest.mark.asyncio
async def test_asimilarity_search_with_score_does_not_mutate_instance_config() -> None:
    """Same invariant for `asimilarity_search_with_score`, which has the
    same back-fill block."""
    config = HybridSearchConfig(fts_query="")
    store = _make_store(config)

    await store.asimilarity_search_with_score("first query")
    await store.asimilarity_search_with_score("second query")

    assert config.fts_query == ""
    calls = store.asimilarity_search_with_score_by_vector.await_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["hybrid_search_config"].fts_query == "first query"
    assert calls[1].kwargs["hybrid_search_config"].fts_query == "second query"


@pytest.mark.asyncio
async def test_asimilarity_search_does_not_mutate_caller_provided_config() -> None:
    """A `HybridSearchConfig` passed via kwargs should also be left alone —
    callers may reuse the same object across calls."""
    caller_config = HybridSearchConfig(fts_query="")
    store = _make_store(HybridSearchConfig(fts_query=""))

    await store.asimilarity_search("query A", hybrid_search_config=caller_config)
    await store.asimilarity_search("query B", hybrid_search_config=caller_config)

    assert caller_config.fts_query == ""


@pytest.mark.asyncio
async def test_asimilarity_search_preserves_existing_fts_query() -> None:
    """If the caller already set `fts_query`, the method must leave it
    alone (the back-fill branch only fires when `fts_query` is empty)."""
    caller_config = HybridSearchConfig(fts_query="user-specified")
    store = _make_store(HybridSearchConfig(fts_query=""))

    await store.asimilarity_search("ignored", hybrid_search_config=caller_config)

    calls: list[Any] = store.asimilarity_search_by_vector.await_args_list
    assert calls[0].kwargs["hybrid_search_config"].fts_query == "user-specified"
