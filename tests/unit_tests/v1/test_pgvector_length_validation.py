"""Regression tests pinning that `PGVector.add_embeddings` /
`aadd_embeddings` raise `ValueError` when input lengths disagree.

Previously these methods built the SQL payload with
`zip(texts, metadatas, embeddings, ids_)`, which silently truncates to
the shortest argument and then returns the full `ids_` list — so an
upstream embedder bug that returned fewer embeddings than texts would
yield N IDs to the caller but only M rows in the database, with no
exception. See #300.

These tests bypass `PGVector.__init__` (no Postgres needed) by stubbing
the few attributes the validation path touches.
"""

from __future__ import annotations

import pytest

from langchain_postgres.vectorstores import PGVector


def _make_store() -> PGVector:
    """Build a barebones `PGVector` that's just enough for the early
    length-validation block to execute."""
    store = object.__new__(PGVector)
    # Sync path: `assert not self._async_engine` runs before validation.
    store._async_engine = False
    return store


def test_add_embeddings_raises_when_embeddings_shorter_than_texts() -> None:
    store = _make_store()
    with pytest.raises(ValueError, match="3 texts but 1 embeddings"):
        store.add_embeddings(
            texts=["a", "b", "c"],
            embeddings=[[0.1, 0.2]],
        )


def test_add_embeddings_raises_when_metadatas_length_mismatches() -> None:
    store = _make_store()
    with pytest.raises(ValueError, match="2 texts but 3 metadatas"):
        store.add_embeddings(
            texts=["a", "b"],
            embeddings=[[0.1], [0.2]],
            metadatas=[{}, {}, {}],
        )


def test_add_embeddings_raises_when_ids_length_mismatches() -> None:
    store = _make_store()
    with pytest.raises(ValueError, match="2 texts but 1 ids"):
        store.add_embeddings(
            texts=["a", "b"],
            embeddings=[[0.1], [0.2]],
            ids=["only-one"],
        )


def test_add_embeddings_does_not_raise_when_lengths_match() -> None:
    """Validation must not fire when shapes line up. We don't reach the DB
    layer here — the test uses an attribute access on a stubbed session
    helper to detect that validation passed and execution continued."""
    store = _make_store()
    # Validation lives before any DB session is opened, so reaching the
    # `with self._make_sync_session()` line should raise AttributeError on
    # our stub. That is the signal we want — the ValueError path was NOT
    # taken.
    with pytest.raises(AttributeError):
        store.add_embeddings(
            texts=["a", "b"],
            embeddings=[[0.1], [0.2]],
            metadatas=[{}, {}],
            ids=["x", "y"],
        )


@pytest.mark.asyncio
async def test_aadd_embeddings_raises_when_embeddings_shorter_than_texts() -> None:
    store = _make_store()
    # `__apost_init__` short-circuits when `_async_init` is True, so the
    # async validation block runs without touching the DB engine.
    store._async_init = True
    with pytest.raises(ValueError, match="3 texts but 2 embeddings"):
        await store.aadd_embeddings(
            texts=["a", "b", "c"],
            embeddings=[[0.1], [0.2]],
        )
