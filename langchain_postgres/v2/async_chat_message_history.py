from __future__ import annotations

import json
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from sqlalchemy import RowMapping, text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import PGEngine


class AsyncPGChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a PostgreSQL database."""

    __create_key = object()

    def __init__(
        self,
        key: object,
        pool: AsyncEngine,
        session_id: str,
        table_name: str,
        store_message: bool,
        schema_name: str = "public",
    ):
        """AsyncPGChatMessageHistory constructor.

        Args:
            key (object): Key to prevent direct constructor usage.
            engine (PGEngine): Database connection pool.
            session_id (str): Retrieve the table content with this session ID.
            table_name (str): Table name that stores the chat message history.
            store_message (bool): Whether to store the whole message or store data & type seperately
            schema_name (str): The schema name where the table is located (default: "public").

        Raises:
            Exception: If constructor is directly called by the user.
        """
        if key != AsyncPGChatMessageHistory.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self.pool = pool
        self.session_id = session_id
        self.table_name = table_name
        self.schema_name = schema_name
        self.store_message = store_message

    @classmethod
    async def create(
        cls,
        engine: PGEngine,
        session_id: str,
        table_name: str,
        schema_name: str = "public",
    ) -> AsyncPGChatMessageHistory:
        """Create a new AsyncPGChatMessageHistory instance.

        Args:
            engine (PGEngine): PGEngine to use.
            session_id (str): Retrieve the table content with this session ID.
            table_name (str): Table name that stores the chat message history.
            schema_name (str): The schema name where the table is located (default: "public").

        Raises:
            IndexError: If the table provided does not contain required schema.

        Returns:
            AsyncPGChatMessageHistory: A newly created instance of AsyncPGChatMessageHistory.
        """
        column_names = await engine._aload_table_schema(table_name, schema_name)

        required_columns = ["id", "session_id", "data", "type"]
        supported_columns = ["id", "session_id", "message", "created_at"]

        if not (all(x in column_names for x in required_columns)):
            if not (all(x in column_names for x in supported_columns)):
                raise IndexError(
                    f"Table '{schema_name}'.'{table_name}' has incorrect schema. Got "
                    f"column names '{column_names}' but required column names "
                    f"'{required_columns}'.\nPlease create table with following schema:"
                    f"\nCREATE TABLE {schema_name}.{table_name} ("
                    "\n    id INT AUTO_INCREMENT PRIMARY KEY,"
                    "\n    session_id TEXT NOT NULL,"
                    "\n    data JSONB NOT NULL,"
                    "\n    type TEXT NOT NULL"
                    "\n);"
                )

        store_message = True if "message" in column_names else False

        return cls(
            cls.__create_key,
            engine._pool,
            session_id,
            table_name,
            store_message,
            schema_name,
        )

    def _insert_query(self, message: BaseMessage) -> tuple[str, dict]:
        if self.store_message:
            query = f"""INSERT INTO "{self.schema_name}"."{self.table_name}"(session_id, message) VALUES (:session_id, :message)"""
            params = {
                "message": json.dumps(message_to_dict(message)),
                "session_id": self.session_id,
            }
        else:
            query = f"""INSERT INTO "{self.schema_name}"."{self.table_name}"(session_id, data, type) VALUES (:session_id, :data, :type)"""
            params = {
                "data": json.dumps(message.model_dump()),
                "session_id": self.session_id,
                "type": message.type,
            }

        return query, params

    async def aadd_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Postgres"""
        query, params = self._insert_query(message)
        async with self.pool.connect() as conn:
            await conn.execute(text(query), params)
            await conn.commit()

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append a list of messages to the record in Postgres"""
        for message in messages:
            await self.aadd_message(message)

    async def aclear(self) -> None:
        """Clear session memory from Postgres"""
        query = f"""DELETE FROM "{self.schema_name}"."{self.table_name}" WHERE session_id = :session_id;"""
        async with self.pool.connect() as conn:
            await conn.execute(text(query), {"session_id": self.session_id})
            await conn.commit()

    def _select_query(self) -> str:
        if self.store_message:
            return f"""SELECT message FROM "{self.schema_name}"."{self.table_name}" WHERE session_id = :session_id ORDER BY id;"""
        else:
            return f"""SELECT data, type FROM "{self.schema_name}"."{self.table_name}" WHERE session_id = :session_id ORDER BY id;"""

    def _convert_to_messages(self, rows: Sequence[RowMapping]) -> list[BaseMessage]:
        if self.store_message:
            items = [row["message"] for row in rows]
            messages = messages_from_dict(items)
        else:
            items = [{"data": row["data"], "type": row["type"]} for row in rows]
            messages = messages_from_dict(items)
        return messages

    async def _aget_messages(self) -> list[BaseMessage]:
        """Retrieve the messages from Postgres."""
        query = self._select_query()
        async with self.pool.connect() as conn:
            result = await conn.execute(text(query), {"session_id": self.session_id})
            result_map = result.mappings()
            results = result_map.fetchall()
        if not results:
            return []

        messages = self._convert_to_messages(results)
        return messages

    def clear(self) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGChatMessageHistory. Use PGChatMessageHistory interface instead."
        )

    def add_message(self, message: BaseMessage) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGChatMessageHistory. Use PGChatMessageHistory interface instead."
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPGChatMessageHistory. Use PGChatMessageHistory interface instead."
        )
