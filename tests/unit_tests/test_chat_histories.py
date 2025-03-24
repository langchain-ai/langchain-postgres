import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_postgres.chat_message_histories import PostgresChatMessageHistory
from psycopg_pool import AsyncConnectionPool
from tests.utils import asyncpg_client, syncpg_client, DSN


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
        assert messages == [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]

        chat_history.add_messages(
            [
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ]
        )

        messages = chat_history.messages
        assert len(messages) == 6
        assert messages == [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]

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
        assert messages == [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]

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
        assert messages == [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]

        # clear
        await chat_history.aclear()
        assert await chat_history.aget_messages() == []


async def test_async_chat_history_with_pool() -> None:
    """Test the async chat history using a connection pool."""
    # Initialize the connection pool
    pool = AsyncConnectionPool(conninfo=DSN)
    try:
        table_name = "chat_history"
        session_id = str(uuid.uuid4())

        # Create tables using a connection from the pool
        async with pool.connection() as async_connection:
            await PostgresChatMessageHistory.adrop_table(async_connection, table_name)
            await PostgresChatMessageHistory.acreate_tables(async_connection, table_name)

        # Create PostgresChatMessageHistory with conn_pool
        chat_history = PostgresChatMessageHistory(
            session_id=session_id,
            table_name=table_name,
            conn_pool=pool,
        )

        # Ensure the chat history is empty
        messages = await chat_history.aget_messages()
        assert messages == []

        # Add messages to the chat history
        await chat_history.aadd_messages(
            [
                SystemMessage(content="System message"),
                AIMessage(content="AI response"),
                HumanMessage(content="Human message"),
            ]
        )

        # Retrieve messages from the chat history
        messages = await chat_history.aget_messages()
        assert len(messages) == 3
        assert messages == [
            SystemMessage(content="System message"),
            AIMessage(content="AI response"),
            HumanMessage(content="Human message"),
        ]

        # Add more messages
        await chat_history.aadd_messages(
            [
                SystemMessage(content="Another system message"),
                AIMessage(content="Another AI response"),
                HumanMessage(content="Another human message"),
            ]
        )

        # Verify all messages are retrieved
        messages = await chat_history.aget_messages()
        assert len(messages) == 6
        assert messages == [
            SystemMessage(content="System message"),
            AIMessage(content="AI response"),
            HumanMessage(content="Human message"),
            SystemMessage(content="Another system message"),
            AIMessage(content="Another AI response"),
            HumanMessage(content="Another human message"),
        ]

        # Clear the chat history
        await chat_history.aclear()
        messages = await chat_history.aget_messages()
        assert messages == []
    finally:
        # Close the connection pool
        await pool.close()
