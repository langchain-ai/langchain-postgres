"""Regression tests for the lazy import of `PostgresChatMessageHistory`.

`langchain_postgres/__init__.py` used to eagerly import
`chat_message_histories.PostgresChatMessageHistory`, which top-level-imports
`psycopg`. That meant `from langchain_postgres import PGEngine` (or any
other driver-agnostic v2 symbol) transitively loaded the LGPL-licensed
`psycopg`, breaking downstreams that ship only the v2 API on a non-LGPL
PostgreSQL driver such as `asyncpg`. See #296.

These tests assert:

- accessing `PostgresChatMessageHistory` via package attribute lookup still
  works (no backward-compat regression);
- a fresh subprocess that only imports `langchain_postgres` does **not**
  transitively load `psycopg`;
- the same subprocess gains a `psycopg` import once the lazy attribute is
  actually touched.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_postgres_chat_message_history_accessible_via_package() -> None:
    import langchain_postgres
    from langchain_postgres.chat_message_histories import (
        PostgresChatMessageHistory as _Direct,
    )

    assert langchain_postgres.PostgresChatMessageHistory is _Direct


def test_unknown_attribute_raises_attribute_error() -> None:
    import langchain_postgres

    try:
        _ = langchain_postgres.DefinitelyDoesNotExist  # type: ignore[attr-defined]
    except AttributeError as exc:
        assert "DefinitelyDoesNotExist" in str(exc)
    else:  # pragma: no cover
        msg = "Expected AttributeError"
        raise AssertionError(msg)


def _run_python(snippet: str) -> str:
    """Run `snippet` in a fresh subprocess and return its stdout (one line)."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    return result.stdout.strip()


def test_psycopg_not_imported_when_only_v2_symbols_used() -> None:
    """`from langchain_postgres import PGEngine, PGVectorStore` must NOT
    transitively load `psycopg`."""
    out = _run_python(
        """
        import sys
        from langchain_postgres import PGEngine, PGVectorStore  # noqa: F401
        print("psycopg" in sys.modules)
        """
    )
    assert out == "False", (
        f"psycopg should not be imported eagerly; got sys.modules check={out!r}"
    )


def test_psycopg_is_imported_when_lazy_symbol_is_accessed() -> None:
    """Accessing the lazy symbol must still pull `psycopg` in (so the v1
    chat-history API keeps working)."""
    out = _run_python(
        """
        import sys
        import langchain_postgres
        _ = langchain_postgres.PostgresChatMessageHistory
        print("psycopg" in sys.modules)
        """
    )
    assert out == "True", (
        f"psycopg should be imported once the lazy attr is touched; got {out!r}"
    )
