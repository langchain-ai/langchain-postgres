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

The package currently only supports the [psycogp3](https://www.psycopg.org/psycopg3/) driver.

## Installation

```bash
pip install -U langchain-postgres
```

## Change Log

0.0.6: 
- Remove langgraph as a dependency as it was causing dependency conflicts.
- Base interface for checkpointer changed in langgraph, so existing implementation would've broken regardless.

## Usage

### ChatMessageHistory

The chat message history abstraction helps to persist chat message history 
in a postgres table.

PostgresChatMessageHistory is parameterized using a `table_name` and a `session_id`.

The `table_name` is the name of the table in the database where 
the chat messages will be stored.

The `session_id` is a unique identifier for the chat session. It can be assigned
by the caller using `uuid.uuid4()`.

```python
import uuid

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory
import psycopg

# Establish a synchronous connection to the database
# (or use psycopg.AsyncConnection for async)
conn_info = ... # Fill in with your connection info
sync_connection = psycopg.connect(conn_info)

# Create the table schema (only needs to be done once)
table_name = "chat_history"
PostgresChatMessageHistory.create_tables(sync_connection, table_name)

session_id = str(uuid.uuid4())

# Initialize the chat history manager
chat_history = PostgresChatMessageHistory(
    table_name,
    session_id,
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


### Vectorstore

See example for the [PGVector vectorstore here](https://github.com/langchain-ai/langchain-postgres/blob/main/examples/vectorstore.ipynb)
