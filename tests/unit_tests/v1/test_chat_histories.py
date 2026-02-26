import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_postgres.chat_message_histories import PostgresChatMessageHistory
from tests.utils import asyncpg_client, syncpg_client


def _assert_messages_content(
    actual: list, expected_contents: list[tuple[str, str]]
) -> None:
    """Assert messages match by type and content, ignoring additional_kwargs."""
    assert len(actual) == len(expected_contents)
    for msg, (msg_type, content) in zip(actual, expected_contents):
        assert msg.type == msg_type
        assert msg.content == content


def _assert_recent_timestamp(iso_str: str, tolerance_seconds: int = 10) -> datetime:
    """Assert an ISO timestamp string is recent (within tolerance of now)."""
    parsed = datetime.fromisoformat(iso_str)
    assert parsed.tzinfo is not None, "Timestamp must be timezone-aware"
    now = datetime.now(timezone.utc)
    delta = abs((now - parsed).total_seconds())
    assert delta < tolerance_seconds, (
        f"Timestamp {iso_str} is {delta:.1f}s from now, expected < {tolerance_seconds}s"
    )
    return parsed


def test_sync_chat_history() -> None:
    table_name = "chat_history"
    session_id = str(uuid.UUID(int=123))
    with syncpg_client() as sync_connection:
        PostgresChatMessageHistory.drop_table(sync_connection, table_name)
        PostgresChatMessageHistory.create_tables(sync_connection, table_name)

        chat_history = PostgresChatMessageHistory(
            table_name, session_id, sync_connection=sync_connection
        )

        messages = chat_history.messages
        assert messages == []

        assert chat_history is not None

        # Get messages from the chat history
        messages = chat_history.messages
        assert messages == []

        chat_history.add_messages(
            [
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ]
        )

        # Get messages from the chat history
        messages = chat_history.messages
        assert len(messages) == 3
        _assert_messages_content(
            messages,
            [("system", "Meow"), ("ai", "woof"), ("human", "bark")],
        )

        chat_history.add_messages(
            [
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ]
        )

        messages = chat_history.messages
        assert len(messages) == 6
        _assert_messages_content(
            messages,
            [
                ("system", "Meow"),
                ("ai", "woof"),
                ("human", "bark"),
                ("system", "Meow"),
                ("ai", "woof"),
                ("human", "bark"),
            ],
        )

        chat_history.clear()
        assert chat_history.messages == []


async def test_async_chat_history() -> None:
    """Test the async chat history."""
    async with asyncpg_client() as async_connection:
        table_name = "chat_history"
        session_id = str(uuid.UUID(int=125))
        await PostgresChatMessageHistory.adrop_table(async_connection, table_name)
        await PostgresChatMessageHistory.acreate_tables(async_connection, table_name)

        chat_history = PostgresChatMessageHistory(
            table_name, session_id, async_connection=async_connection
        )

        messages = await chat_history.aget_messages()
        assert messages == []

        # Add messages
        await chat_history.aadd_messages(
            [
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ]
        )
        # Get the messages
        messages = await chat_history.aget_messages()
        assert len(messages) == 3
        _assert_messages_content(
            messages,
            [("system", "Meow"), ("ai", "woof"), ("human", "bark")],
        )

        # Add more messages
        await chat_history.aadd_messages(
            [
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ]
        )
        # Get the messages
        messages = await chat_history.aget_messages()
        assert len(messages) == 6
        _assert_messages_content(
            messages,
            [
                ("system", "Meow"),
                ("ai", "woof"),
                ("human", "bark"),
                ("system", "Meow"),
                ("ai", "woof"),
                ("human", "bark"),
            ],
        )

        # clear
        await chat_history.aclear()
        assert await chat_history.aget_messages() == []


def test_sync_message_timestamps() -> None:
    """Test that retrieved messages include recent created_at timestamps."""
    table_name = "chat_history"
    session_id = str(uuid.UUID(int=200))
    with syncpg_client() as sync_connection:
        PostgresChatMessageHistory.drop_table(sync_connection, table_name)
        PostgresChatMessageHistory.create_tables(sync_connection, table_name)

        chat_history = PostgresChatMessageHistory(
            table_name, session_id, sync_connection=sync_connection
        )

        chat_history.add_messages(
            [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there"),
            ]
        )

        messages = chat_history.get_messages()
        assert len(messages) == 2

        timestamps = []
        for msg in messages:
            assert "created_at" in msg.additional_kwargs
            ts = _assert_recent_timestamp(msg.additional_kwargs["created_at"])
            timestamps.append(ts)

        # Timestamps should be monotonically non-decreasing
        assert timestamps[1] >= timestamps[0]


async def test_async_message_timestamps() -> None:
    """Test that async retrieved messages include recent created_at timestamps."""
    async with asyncpg_client() as async_connection:
        table_name = "chat_history"
        session_id = str(uuid.UUID(int=201))
        await PostgresChatMessageHistory.adrop_table(async_connection, table_name)
        await PostgresChatMessageHistory.acreate_tables(async_connection, table_name)

        chat_history = PostgresChatMessageHistory(
            table_name, session_id, async_connection=async_connection
        )

        await chat_history.aadd_messages(
            [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there"),
            ]
        )

        messages = await chat_history.aget_messages()
        assert len(messages) == 2

        timestamps = []
        for msg in messages:
            assert "created_at" in msg.additional_kwargs
            ts = _assert_recent_timestamp(msg.additional_kwargs["created_at"])
            timestamps.append(ts)

        assert timestamps[1] >= timestamps[0]


def test_sync_session_info() -> None:
    """Test that add_messages updates session info."""
    table_name = "chat_history"
    session_id = str(uuid.UUID(int=202))
    with syncpg_client() as sync_connection:
        PostgresChatMessageHistory.drop_table(sync_connection, table_name)
        PostgresChatMessageHistory.create_tables(sync_connection, table_name)

        chat_history = PostgresChatMessageHistory(
            table_name, session_id, sync_connection=sync_connection
        )

        # Before any messages, session info should be None
        assert chat_history.get_session_info() is None
        assert chat_history.get_last_message_time() is None

        chat_history.add_messages([HumanMessage(content="First")])
        info = chat_history.get_session_info()
        assert info is not None
        assert isinstance(info["created_at"], datetime)
        assert isinstance(info["last_message_time"], datetime)
        assert info["message_count"] == 1
        # Session created_at should be recent
        assert (
            abs((datetime.now(timezone.utc) - info["created_at"]).total_seconds()) < 10
        )
        # created_at and last_message_time should match on first insert
        assert info["created_at"] == info["last_message_time"]
        first_created_at = info["created_at"]
        first_time = info["last_message_time"]

        # Add more messages — last_message_time advances, created_at stays
        chat_history.add_messages(
            [HumanMessage(content="Second"), HumanMessage(content="Third")]
        )
        info2 = chat_history.get_session_info()
        assert info2 is not None
        assert info2["created_at"] == first_created_at  # unchanged
        assert info2["last_message_time"] >= first_time
        assert info2["message_count"] == 3

        # last_message_time should be consistent with the message timestamps
        messages = chat_history.get_messages()
        last_msg_ts = datetime.fromisoformat(
            messages[-1].additional_kwargs["created_at"]
        )
        assert abs((info2["last_message_time"] - last_msg_ts).total_seconds()) < 2

        # Clear messages — session row should persist with original metadata
        chat_history.clear()
        assert chat_history.messages == []
        persisted = chat_history.get_session_info()
        assert persisted is not None
        assert persisted["created_at"] == first_created_at
        assert persisted["last_message_time"] == info2["last_message_time"]
        assert persisted["message_count"] == 3  # count persists after clear


async def test_async_session_info() -> None:
    """Test that aadd_messages updates session info."""
    async with asyncpg_client() as async_connection:
        table_name = "chat_history"
        session_id = str(uuid.UUID(int=203))
        await PostgresChatMessageHistory.adrop_table(async_connection, table_name)
        await PostgresChatMessageHistory.acreate_tables(async_connection, table_name)

        chat_history = PostgresChatMessageHistory(
            table_name, session_id, async_connection=async_connection
        )

        # Before any messages
        assert await chat_history.aget_session_info() is None
        assert await chat_history.aget_last_message_time() is None

        await chat_history.aadd_messages([HumanMessage(content="First")])
        info = await chat_history.aget_session_info()
        assert info is not None
        assert info["message_count"] == 1
        assert info["created_at"] == info["last_message_time"]
        first_created_at = info["created_at"]
        assert abs((datetime.now(timezone.utc) - first_created_at).total_seconds()) < 10

        await chat_history.aadd_messages([HumanMessage(content="Second")])
        info2 = await chat_history.aget_session_info()
        assert info2 is not None
        assert info2["created_at"] == first_created_at  # unchanged
        assert info2["last_message_time"] >= info["last_message_time"]
        assert info2["message_count"] == 2


def test_sync_use_timestamp_with_retrieval() -> None:
    """Test that explicit timestamps via use_timestamp survive round-trip."""
    table_name = "chat_history"
    session_id = str(uuid.UUID(int=204))
    with syncpg_client() as sync_connection:
        PostgresChatMessageHistory.drop_table(sync_connection, table_name)
        PostgresChatMessageHistory.create_tables(sync_connection, table_name)

        chat_history = PostgresChatMessageHistory(
            table_name, session_id, sync_connection=sync_connection
        )

        known_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        msg = HumanMessage(
            content="Timestamped",
            additional_kwargs={"created_at": known_time.isoformat()},
        )

        chat_history.add_messages([msg], use_timestamp=True)

        messages = chat_history.get_messages()
        assert len(messages) == 1
        retrieved_ts = messages[0].additional_kwargs["created_at"]
        retrieved_dt = datetime.fromisoformat(retrieved_ts)
        assert retrieved_dt == known_time

        # Session info should reflect the explicit timestamp
        info = chat_history.get_session_info()
        assert info is not None
        assert info["last_message_time"] == known_time
        assert info["created_at"] == known_time
        assert info["message_count"] == 1


def test_sync_empty_session_info() -> None:
    """Test that get_session_info returns None for empty session."""
    table_name = "chat_history"
    session_id = str(uuid.UUID(int=205))
    with syncpg_client() as sync_connection:
        PostgresChatMessageHistory.drop_table(sync_connection, table_name)
        PostgresChatMessageHistory.create_tables(sync_connection, table_name)

        chat_history = PostgresChatMessageHistory(
            table_name, session_id, sync_connection=sync_connection
        )

        assert chat_history.get_session_info() is None
        assert chat_history.get_last_message_time() is None


def test_sync_multiple_sessions_independent() -> None:
    """Test that session info is tracked independently per session."""
    table_name = "chat_history"
    session_a = str(uuid.UUID(int=300))
    session_b = str(uuid.UUID(int=301))
    with syncpg_client() as sync_connection:
        PostgresChatMessageHistory.drop_table(sync_connection, table_name)
        PostgresChatMessageHistory.create_tables(sync_connection, table_name)

        history_a = PostgresChatMessageHistory(
            table_name, session_a, sync_connection=sync_connection
        )
        history_b = PostgresChatMessageHistory(
            table_name, session_b, sync_connection=sync_connection
        )

        history_a.add_messages([HumanMessage(content="Session A")])
        info_a = history_a.get_session_info()
        assert info_a is not None
        assert info_a["message_count"] == 1

        # Session B has no messages yet
        assert history_b.get_session_info() is None

        history_b.add_messages([HumanMessage(content="Session B")])
        info_b = history_b.get_session_info()
        assert info_b is not None
        assert info_b["message_count"] == 1
        assert info_b["last_message_time"] >= info_a["last_message_time"]

        # Session A unchanged
        assert history_a.get_session_info() == info_a
