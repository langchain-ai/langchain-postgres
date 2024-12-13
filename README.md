# langchain-postgres

[![Release Notes](https://img.shields.io/github/release/langchain-ai/langchain-postgres)](https://github.com/langchain-ai/langchain-postgres/releases)
[![CI](https://github.com/langchain-ai/langchain-postgres/actions/workflows/ci.yml/badge.svg)](https://github.com/langchain-ai/langchain-postgres/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://twitter.com/langchainai)
[![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.gg/6adMQxSpJS)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langchain-postgres)](https://github.com/langchain-ai/langchain-postgres/issues)

The `langchain-postgres` package implementations of core LangChain abstractions using `Postgres`.

The package is released under the MIT license. 

Feel free to use the abstraction as provided or else modify them / extend them as appropriate for your own application.

## Requirements

- [psycopg3](https://www.psycopg.org/psycopg3/): The PostgreSQL driver.
- [psycopg_pool](https://www.psycopg.org/psycopg3/docs/advanced/pool.html): For connection pooling support.

## Installation

```bash
pip install -U langchain-postgres
```

## Change Log

**0.0.7:**

- Added support for asynchronous connection pooling in `PostgresChatMessageHistory`.
- Adjusted parameter order in `PostgresChatMessageHistory` to make `session_id` the first parameter.

## Usage

### ChatMessageHistory

The chat message history abstraction helps to persist chat message history in a Postgres table.

`PostgresChatMessageHistory` is parameterized using a `session_id` and an optional `table_name` (default is `"chat_history"`).

- **`session_id`:** A unique identifier for the chat session. It can be assigned using `uuid.uuid4()`.
- **`table_name`:** The name of the table in the database where the chat messages will be stored.

#### **Synchronous Usage**

```python
import uuid

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
import psycopg

# Establish a synchronous connection to the database
conn_info = "postgresql://user:password@host:port/dbname"  # Replace with your connection info
sync_connection = psycopg.connect(conn_info)

# Create the table schema (only needs to be done once)
table_name = "chat_history"
PostgresChatMessageHistory.create_tables(sync_connection, table_name)

session_id = str(uuid.uuid4())

# Initialize the chat history manager
chat_history = PostgresChatMessageHistory(
    session_id=session_id,
    table_name=table_name,
    sync_connection=sync_connection
)

# Add messages to the chat history
chat_history.add_messages([
    SystemMessage(content="Meow"),
    AIMessage(content="woof"),
    HumanMessage(content="bark"),
])

print(chat_history.messages)
```

#### **Asynchronous Usage with Connection Pooling**

```python
import uuid
import asyncio

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
from psycopg_pool import AsyncConnectionPool

# Asynchronous main function
async def main():
    # Database connection string
    conn_info = "postgresql://user:password@host:port/dbname"  # Replace with your connection info

    # Initialize the connection pool
    pool = AsyncConnectionPool(conninfo=conn_info)

    try:
        # Create the table schema (only needs to be done once)
        async with pool.connection() as async_connection:
            table_name = "chat_history"
            await PostgresChatMessageHistory.adrop_table(async_connection, table_name)
            await PostgresChatMessageHistory.acreate_tables(async_connection, table_name)

        session_id = str(uuid.uuid4())

        # Initialize the chat history manager with the connection pool
        chat_history = PostgresChatMessageHistory(
            session_id=session_id,
            table_name=table_name,
            conn_pool=pool
        )

        # Add messages to the chat history asynchronously
        await chat_history.aadd_messages([
            SystemMessage(content="System message"),
            AIMessage(content="AI response"),
            HumanMessage(content="Human message"),
        ])

        # Retrieve messages from the chat history
        messages = await chat_history.aget_messages()
        print(messages)
    finally:
        # Close the connection pool
        await pool.close()

# Run the async main function
asyncio.run(main())
```

### Vectorstore

See example for the [PGVector vectorstore here](https://github.com/langchain-ai/langchain-postgres/blob/main/examples/vectorstore.ipynb)
