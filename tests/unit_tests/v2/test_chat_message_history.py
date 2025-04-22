import uuid
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from sqlalchemy import text

from langchain_postgres import PGChatMessageHistory, PGEngine
from tests.utils import VECTORSTORE_CONNECTION_STRING

TABLE_NAME = "message_store" + str(uuid.uuid4())
TABLE_NAME_ASYNC = "message_store" + str(uuid.uuid4())


async def aexecute(
    engine: PGEngine,
    query: str,
) -> None:
    async def run(engine: PGEngine, query: str) -> None:
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await engine._run_as_async(run(engine, query))


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[PGEngine]:
    engine = PGEngine.from_connection_string(url=VECTORSTORE_CONNECTION_STRING)
    engine.init_chat_history_table(table_name=TABLE_NAME)
    yield engine
    # use default table for PGChatMessageHistory
    query = f'DROP TABLE IF EXISTS "{TABLE_NAME}"'
    await aexecute(engine, query)
    await engine.close()


@pytest_asyncio.fixture
async def async_engine() -> AsyncIterator[PGEngine]:
    async_engine = PGEngine.from_connection_string(url=VECTORSTORE_CONNECTION_STRING)
    await async_engine.ainit_chat_history_table(table_name=TABLE_NAME_ASYNC)
    yield async_engine
    # use default table for PGChatMessageHistory
    query = f'DROP TABLE IF EXISTS "{TABLE_NAME_ASYNC}"'
    await aexecute(async_engine, query)
    await async_engine.close()


def test_chat_message_history(engine: PGEngine) -> None:
    history = PGChatMessageHistory.create_sync(
        engine=engine, session_id="test", table_name=TABLE_NAME
    )
    history.add_user_message("hi!")
    history.add_ai_message("whats up?")
    messages = history.messages

    # verify messages are correct
    assert messages[0].content == "hi!"
    assert type(messages[0]) is HumanMessage
    assert messages[1].content == "whats up?"
    assert type(messages[1]) is AIMessage

    # verify clear() clears message history
    history.clear()
    assert len(history.messages) == 0


def test_chat_table(engine: Any) -> None:
    with pytest.raises(ValueError):
        PGChatMessageHistory.create_sync(
            engine=engine, session_id="test", table_name="doesnotexist"
        )


@pytest.mark.asyncio
async def test_chat_schema(engine: Any) -> None:
    doc_table_name = "test_table" + str(uuid.uuid4())
    engine.init_document_table(table_name=doc_table_name)
    with pytest.raises(IndexError):
        PGChatMessageHistory.create_sync(
            engine=engine, session_id="test", table_name=doc_table_name
        )

    query = f'DROP TABLE IF EXISTS "{doc_table_name}"'
    await aexecute(engine, query)


@pytest.mark.asyncio
async def test_chat_message_history_async(
    async_engine: PGEngine,
) -> None:
    history = await PGChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=TABLE_NAME_ASYNC
    )
    msg1 = HumanMessage(content="hi!")
    msg2 = AIMessage(content="whats up?")
    await history.aadd_message(msg1)
    await history.aadd_message(msg2)
    messages = history.messages

    # verify messages are correct
    assert messages[0].content == "hi!"
    assert type(messages[0]) is HumanMessage
    assert messages[1].content == "whats up?"
    assert type(messages[1]) is AIMessage

    # verify clear() clears message history
    await history.aclear()
    assert len(history.messages) == 0


@pytest.mark.asyncio
async def test_chat_message_history_sync_messages(
    async_engine: PGEngine,
) -> None:
    history1 = await PGChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=TABLE_NAME_ASYNC
    )
    history2 = await PGChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=TABLE_NAME_ASYNC
    )
    msg1 = HumanMessage(content="hi!")
    msg2 = AIMessage(content="whats up?")
    await history1.aadd_message(msg1)
    await history2.aadd_message(msg2)

    assert len(history1.messages) == 2
    assert len(history2.messages) == 2

    # verify clear() clears message history
    await history2.aclear()
    assert len(history2.messages) == 0


@pytest.mark.asyncio
async def test_chat_message_history_set_messages(
    async_engine: PGEngine,
) -> None:
    history = await PGChatMessageHistory.create(
        engine=async_engine, session_id="test", table_name=TABLE_NAME_ASYNC
    )
    msg1 = HumanMessage(content="hi!")
    msg2 = AIMessage(content="bye -_-")
    # verify setting messages property adds to message history
    history.messages = [msg1, msg2]
    assert len(history.messages) == 2


@pytest.mark.asyncio
async def test_chat_table_async(async_engine: PGEngine) -> None:
    with pytest.raises(ValueError):
        await PGChatMessageHistory.create(
            engine=async_engine, session_id="test", table_name="doesnotexist"
        )


@pytest.mark.asyncio
async def test_cross_env_chat_message_history(engine: PGEngine) -> None:
    history = PGChatMessageHistory.create_sync(
        engine=engine, session_id="test_cross", table_name=TABLE_NAME
    )
    await history.aadd_message(HumanMessage(content="hi!"))
    messages = history.messages
    assert messages[0].content == "hi!"
    history.clear()

    history = await PGChatMessageHistory.create(
        engine=engine, session_id="test_cross", table_name=TABLE_NAME
    )
    history.add_message(HumanMessage(content="hi!"))
    messages = history.messages
    assert messages[0].content == "hi!"
    history.clear()
