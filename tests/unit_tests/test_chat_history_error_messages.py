"""The guard messages must name a method that exists."""

import pytest

from langchain_postgres.chat_message_histories import PostgresChatMessageHistory


@pytest.fixture
def history() -> PostgresChatMessageHistory:
    """A history with neither connection, so every guard fires."""
    obj = PostgresChatMessageHistory.__new__(PostgresChatMessageHistory)
    obj._connection = None
    obj._aconnection = None
    obj._table_name = "t"
    obj._session_id = "s"
    return obj


def test_clear_points_at_aclear(history: PostgresChatMessageHistory) -> None:
    """`clear` must not tell the caller to use `clear`."""
    with pytest.raises(ValueError) as excinfo:
        history.clear()

    assert "aclear" in str(excinfo.value)


async def test_aclear_points_at_clear(history: PostgresChatMessageHistory) -> None:
    with pytest.raises(ValueError) as excinfo:
        await history.aclear()

    assert "sync clear" in str(excinfo.value)


def test_get_messages_points_at_aget_messages(
    history: PostgresChatMessageHistory,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        history.get_messages()

    assert "aget_messages" in str(excinfo.value)
