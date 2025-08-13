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

The package supports the [asyncpg](https://github.com/MagicStack/asyncpg) and [psycopg3](https://www.psycopg.org/psycopg3/) drivers.

## Installation

```bash
pip install -U langchain-postgres
```

## Vectorstore

> [!WARNING]
> In v0.0.14+, `PGVector` is deprecated. Please migrate to `PGVectorStore`
> for improved performance and manageability.
> See the [migration guide](https://github.com/langchain-ai/langchain-postgres/blob/main/examples/migrate_pgvector_to_pgvectorstore.ipynb) for details on how to migrate from `PGVector` to `PGVectorStore`.

### Documentation

* [Quickstart](https://github.com/langchain-ai/langchain-postgres/blob/main/examples/pg_vectorstore.ipynb)
* [How-to](https://github.com/langchain-ai/langchain-postgres/blob/main/examples/pg_vectorstore_how_to.ipynb)

### Example

```python
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_postgres import PGEngine, PGVectorStore

# Replace the connection string with your own Postgres connection string
CONNECTION_STRING = "postgresql+psycopg3://langchain:langchain@localhost:6024/langchain"
engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

# Replace the vector size with your own vector size
VECTOR_SIZE = 768
embedding = DeterministicFakeEmbedding(size=VECTOR_SIZE)

TABLE_NAME = "my_doc_collection"

engine.init_vectorstore_table(
    table_name=TABLE_NAME,
    vector_size=VECTOR_SIZE,
)

store = PGVectorStore.create_sync(
    engine=engine,
    table_name=TABLE_NAME,
    embedding_service=embedding,
)

docs = [
    Document(page_content="Apples and oranges"),
    Document(page_content="Cars and airplanes"),
    Document(page_content="Train")
]

store.add_documents(docs)

query = "I'd like a fruit."
docs = store.similarity_search(query)
print(docs)
```

> [!TIP]
> All synchronous functions have corresponding asynchronous functions

### Hybrid Search with PGVectorStore

With PGVectorStore you can use hybrid search for more comprehensive and relevant search results.

```python
hybrid_docs = await vector_store.asimilarity_search("products", k=5, hybrid_search_config=hybrid_search_config)
```

For a detailed guide on how to use hybrid search, see the [documentation](/examples/pg_vectorstore_how_to.ipynb#hybrid-search-with-pgvectorstore ).

## ChatMessageHistory

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

## Google Cloud Integrations

[Google Cloud](https://python.langchain.com/docs/integrations/providers/google/) provides Vector Store, Chat Message History, and Data Loader integrations for [AlloyDB](https://cloud.google.com/alloydb) and [Cloud SQL](https://cloud.google.com/sql) for PostgreSQL databases via the following PyPi packages:

* [`langchain-google-alloydb-pg`](https://github.com/googleapis/langchain-google-alloydb-pg-python)

* [`langchain-google-cloud-sql-pg`](https://github.com/googleapis/langchain-google-cloud-sql-pg-python)

Using the Google Cloud integrations provides the following benefits:

- **Enhanced Security**: Securely connect to Google Cloud databases utilizing IAM for authorization and database authentication without needing to manage SSL certificates, configure firewall rules, or enable authorized networks.
- **Simplified and Secure Connections:** Connect to Google Cloud databases effortlessly using the instance name instead of complex connection strings. The integrations creates a secure connection pool that can be easily shared across your application using the `engine` object.

| Vector Store             | Metadata filtering | Async support  | Schema Flexibility | Improved metadata handling | Hybrid Search |
|--------------------------|--------------------|----------------|--------------------|----------------------------|---------------|
| Google AlloyDB           |          ✓         |        ✓       |         ✓          |             ✓              |       ✗       |
| Google Cloud SQL Postgres|          ✓         |        ✓       |         ✓          |             ✓              |       ✗       |

