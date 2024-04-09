# langchain-postgres

The `langchain-postgres` package implementations of core LangChain abstractions using `Postgres`.

The package is released under the MIT license. 

Feel free to use the abstraction as provided or else modify them / extend them as appropriate for your own application.

## Requirements

The package currently only supports the [psycogp3](https://www.psycopg.org/psycopg3/) driver.

## Installation

```bash
pip install -U langchain-postgres
```

## Usage

### PostgresSaver (LangGraph Checkpointer) 

The LangGraph checkpointer can be used to add memory to your LangGraph application.

`PostgresSaver` is an implementation of the checkpointer saver using
Postgres as the backend.

Currently, only the psycopg3 driver is supported.

Sync usage:

```python 
from psycopg_pool import ConnectionPool
from langchain_postgres import (
    PostgresSaver, PickleCheckpointSerializer
)

pool = ConnectionPool(
    # Example configuration
    conninfo="postgresql://langchain:langchain@localhost:6024/langchain",
    max_size=20,
)

PostgresSaver.create_tables(pool)

checkpointer = PostgresSaver(
    serializer=PickleCheckpointSerializer(),
    sync_connection=pool,
)

# Set up the langgraph workflow with the checkpointer
workflow = ... # Fill in with your workflow
app = workflow.compile(checkpointer=checkpointer)

# Use with the sync methods of `app` (e.g., `app.stream())

pool.close() # Remember to close the connection pool.
```

Async usage:

```python
from psycopg_pool import AsyncConnectionPool
from langchain_postgres import (
    PostgresSaver, PickleCheckpointSerializer
)

pool = AsyncConnectionPool(
    # Example configuration
    conninfo="postgresql://langchain:langchain@localhost:6024/langchain",
    max_size=20,
)

# Create the tables in postgres (only needs to be done once)
await PostgresSaver.acreate_tables(pool)

checkpointer = PostgresSaver(
    serializer=PickleCheckpointSerializer(),
    async_connection=pool,
)

# Set up the langgraph workflow with the checkpointer
workflow = ... # Fill in with your workflow
app = workflow.compile(checkpointer=checkpointer)

# Use with the async methods of `app` (e.g., `app.astream()`)

await pool.close() # Remember to close the connection pool.
```

#### Testing

If testing with the postgres checkpointer it may be useful to both create and
drop the tables before and after the tests.

```python
from psycopg_pool import ConnectionPool
from langchain_postgres import (
    PostgresSaver 
)
with ConnectionPool(
    # Example configuration
    conninfo="postgresql://langchain:langchain@localhost:6024/langchain",
    max_size=20,
) as conn:
    PostgresSaver.create_tables(conn)
    PostgresSaver.drop_tables(conn)
    # Run your unit tests with langgraph
```


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
