"""Error-handling tests for the pgvector migrator (no live Postgres).

The migrator's ``except SQLAlchemyError`` handler must surface the original DB
error as a ``ProgrammingError``. It previously referenced a local ``uuid`` that
is only bound *inside* the ``try`` (after the collection uuid is awaited), so a
``SQLAlchemyError`` raised before that binding made the handler itself raise
``UnboundLocalError`` and mask the real error.
"""

import importlib
from typing import Any

import pytest
from sqlalchemy.exc import ProgrammingError

migrator = importlib.import_module("langchain_postgres.utils.pgvector_migrator")
_extract = getattr(migrator, "__aextract_pgvector_collection")


class _FailingConn:
    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        # Simulate e.g. a missing/renamed table: a SQLAlchemyError subclass.
        raise ProgrammingError("SELECT ...", {}, Exception("relation does not exist"))


class _Ctx:
    async def __aenter__(self) -> _FailingConn:
        return _FailingConn()

    async def __aexit__(self, *args: Any) -> bool:
        return False


class _Pool:
    def connect(self) -> _Ctx:
        return _Ctx()


class _Engine:
    _pool = _Pool()


@pytest.mark.asyncio
async def test_extract_masks_nothing_when_db_errors_before_uuid_bound() -> None:
    """A DB error before the uuid is resolved must raise ProgrammingError.

    Before the fix this raised ``UnboundLocalError`` (the handler referenced an
    unbound ``uuid``), masking the original error.
    """
    gen = _extract(_Engine(), "my_collection")
    with pytest.raises(ProgrammingError):
        await gen.__anext__()
