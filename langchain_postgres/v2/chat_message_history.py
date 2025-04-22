from __future__ import annotations

from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from .async_chat_message_history import AsyncPGChatMessageHistory
from .engine import PGEngine


class PGChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a PostgreSQL database."""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: PGEngine,
        history: AsyncPGChatMessageHistory,
    ):
        """PGChatMessageHistory constructor.

        Args:
            key (object): Key to prevent direct constructor usage.
            engine (PGEngine): Database connection pool.
            history (AsyncPGChatMessageHistory): Async only implementation.

        Raises:
            Exception: If constructor is directly called by the user.
        """
        if key != PGChatMessageHistory.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._engine = engine
        self.__history = history

    @classmethod
    async def create(
        cls,
        engine: PGEngine,
        session_id: str,
        table_name: str,
        schema_name: str = "public",
    ) -> PGChatMessageHistory:
        """Create a new PGChatMessageHistory instance.

        Args:
            engine (PGEngine): PGEngine to use.
            session_id (str): Retrieve the table content with this session ID.
            table_name (str): Table name that stores the chat message history.
            schema_name (str): The schema name where the table is located (default: "public").

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            PGChatMessageHistory: A newly created instance of PGChatMessageHistory.
        """
        coro = AsyncPGChatMessageHistory.create(
            engine, session_id, table_name, schema_name
        )
        history = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, history)

    @classmethod
    def create_sync(
        cls,
        engine: PGEngine,
        session_id: str,
        table_name: str,
        schema_name: str = "public",
    ) -> PGChatMessageHistory:
        """Create a new PGChatMessageHistory instance.

        Args:
            engine (PGEngine): PGEngine to use.
            session_id (str): Retrieve the table content with this session ID.
            table_name (str): Table name that stores the chat message history.
            schema_name: The schema name where the table is located (default: "public").

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            PGChatMessageHistory: A newly created instance of PGChatMessageHistory.
        """
        coro = AsyncPGChatMessageHistory.create(
            engine, session_id, table_name, schema_name
        )
        history = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, history)

    @property
    def messages(self) -> list[BaseMessage]:
        """Fetches all messages stored in Postgres."""
        return self._engine._run_as_sync(self.__history._aget_messages())

    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        """Clear the stored messages and appends a list of messages to the record in Postgres."""
        self.clear()
        self.add_messages(value)

    async def aadd_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Postgres"""
        await self._engine._run_as_async(self.__history.aadd_message(message))

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Postgres"""
        self._engine._run_as_sync(self.__history.aadd_message(message))

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append a list of messages to the record in Postgres"""
        await self._engine._run_as_async(self.__history.aadd_messages(messages))

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append a list of messages to the record in Postgres"""
        self._engine._run_as_sync(self.__history.aadd_messages(messages))

    async def aclear(self) -> None:
        """Clear session memory from Postgres"""
        await self._engine._run_as_async(self.__history.aclear())

    def clear(self) -> None:
        """Clear session memory from Postgres"""
        self._engine._run_as_sync(self.__history.aclear())
