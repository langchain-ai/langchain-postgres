import uuid
from typing import AsyncIterator

import pytest
import pytest_asyncio
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from sqlalchemy import text

from langchain_postgres import PGEngine, PostgresChatMessageHistory
from langchain_postgres.v2.async_chat_message_history import (
    AsyncPGChatMessageHistory,
)
from tests.utils import VECTORSTORE_CONNECTION_STRING, asyncpg_client

TABLE_NAME = "message_store" + str(uuid.uuid4())
TABLE_NAME_ASYNC = "message_store" + str(uuid.uuid4())


async def aexecute(engine: PGEngine, query: str) -> None:
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()


@pytest_asyncio.fixture
async def async_engine() -> AsyncIterator[PGEngine]:
    async_engine = PGEngine.from_connection_string(url=VECTORSTORE_CONNECTION_STRING)
    await async_engine._ainit_chat_history_table(table_name=TABLE_NAME_ASYNC)
    yield async_engine
    # use default table for AsyncPGChatMessageHistory
    query = f'DROP TABLE IF EXISTS "{TABLE_NAME_ASYNC}"'
    await aexecute(async_engine, query)
    await async_engine.close()


@pytest.mark.asyncio
async def test_chat_message_history_async(
    async_engine: PGEngine,
) -> None:
    history = await AsyncPGChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=TABLE_NAME_ASYNC
    )
    msg1 = HumanMessage(content="hi!")
    msg2 = AIMessage(content="whats up?")
    await history.aadd_message(msg1)
    await history.aadd_message(msg2)
    messages = await history._aget_messages()

    # verify messages are correct
    assert messages[0].content == "hi!"
    assert type(messages[0]) is HumanMessage
    assert messages[1].content == "whats up?"
    assert type(messages[1]) is AIMessage

    # verify clear() clears message history
    await history.aclear()
    assert len(await history._aget_messages()) == 0


@pytest.mark.asyncio
async def test_chat_message_history_sync_messages(
    async_engine: PGEngine,
) -> None:
    history1 = await AsyncPGChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=TABLE_NAME_ASYNC
    )
    history2 = await AsyncPGChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=TABLE_NAME_ASYNC
    )
    msg1 = HumanMessage(content="hi!")
    msg2 = AIMessage(content="whats up?")
    await history1.aadd_message(msg1)
    await history2.aadd_message(msg2)

    assert len(await history1._aget_messages()) == 2
    assert len(await history2._aget_messages()) == 2

    # verify clear() clears message history
    await history2.aclear()
    assert len(await history2._aget_messages()) == 0


@pytest.mark.asyncio
async def test_chat_table_async(async_engine: PGEngine) -> None:
    with pytest.raises(ValueError):
        await AsyncPGChatMessageHistory.create(
            engine=async_engine, session_id="test", table_name="doesnotexist"
        )


@pytest.mark.asyncio
async def test_v1_schema_support(async_engine: PGEngine) -> None:
    table_name = "chat_history"
    session_id = str(uuid.UUID(int=125))
    async with asyncpg_client() as async_connection:
        await PostgresChatMessageHistory.adrop_table(async_connection, table_name)
        await PostgresChatMessageHistory.acreate_tables(async_connection, table_name)

        chat_history = PostgresChatMessageHistory(
            table_name, session_id, async_connection=async_connection
        )

        await chat_history.aadd_messages(
            [
                SystemMessage(content="Meow"),
                AIMessage(content="woof"),
                HumanMessage(content="bark"),
            ]
        )

    history = await AsyncPGChatMessageHistory.create(
        engine=async_engine, session_id=session_id, table_name=table_name
    )

    messages = await history._aget_messages()

    assert len(messages) == 3

    msg1 = HumanMessage(content="hi!")
    await history.aadd_message(msg1)

    messages = await history._aget_messages()

    assert len(messages) == 4

    await async_engine._adrop_table(table_name=table_name)


async def test_incorrect_schema(async_engine: PGEngine) -> None:
    table_name = "incorrect_schema_" + str(uuid.uuid4())
    await async_engine._ainit_vectorstore_table(table_name=table_name, vector_size=1024)
    with pytest.raises(IndexError):
        await AsyncPGChatMessageHistory.create(
            engine=async_engine, session_id="test", table_name=table_name
        )
    query = f'DROP TABLE IF EXISTS "{table_name}"'
    await aexecute(async_engine, query)
