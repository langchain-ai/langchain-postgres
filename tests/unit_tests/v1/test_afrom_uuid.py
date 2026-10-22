"""Auto-generated ids from the async constructors must be random UUIDv4.

These are pure-Python tests: constructing ``PGVector`` with ``async_mode=True``
defers ``__post_init__``, so no database connection is opened, and
``aadd_embeddings`` is stubbed out to capture the ids the constructor generates.
"""

import uuid
from typing import Any, List, Optional
from unittest.mock import patch

import pytest

from langchain_postgres.vectorstores import PGVector
from tests.unit_tests.fake_embeddings import FakeEmbeddings

# A well-formed connection string. ``create_async_engine`` is lazy, so this does
# not open a socket, and ``aadd_embeddings`` is patched before any I/O happens.
FAKE_ASYNC_CONNECTION = "postgresql+psycopg://user:pw@localhost:5432/db"


@pytest.mark.asyncio
async def test_afrom_texts_generates_uuid4_ids() -> None:
    """``afrom_texts`` should mint random (v4) ids, matching the sync path."""
    captured: dict = {}

    async def _capture(
        *args: Any, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> None:
        captured["ids"] = ids

    with patch.object(PGVector, "aadd_embeddings", _capture):
        await PGVector.afrom_texts(
            texts=["foo", "bar", "baz"],
            embedding=FakeEmbeddings(),
            connection=FAKE_ASYNC_CONNECTION,
        )

    ids = captured["ids"]
    assert ids is not None and len(ids) == 3
    # uuid1 embeds the host MAC address and a timestamp; the sync `from_texts`
    # uses uuid4, so the async path must too.
    assert all(uuid.UUID(i).version == 4 for i in ids)
