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

### HNSW index

```python
from langchain_postgres import PGVector, EmbeddingIndexType

PGVector(
    collection_name="test_collection",
    embeddings=FakeEmbedding(),
    connection=CONNECTION_STRING,
    embedding_length=1536,
    embedding_index=EmbeddingIndexType.hnsw,
    embedding_index_ops="vector_cosine_ops",
)
```

- Embedding length is required for HNSW index.
- Allowed values for `embedding_index_ops` are described in the [pgvector HNSW](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw).

Can set `ef_construction` and `m` parameters for HNSW index.
Refer to the [pgvector HNSW Index Options](https://github.com/pgvector/pgvector?tab=readme-ov-file#index-options) to better understand these parameters.

```python
from langchain_postgres import PGVector, EmbeddingIndexType

PGVector(
    collection_name="test_collection",
    embeddings=FakeEmbedding(),
    connection=CONNECTION_STRING,
    embedding_length=1536,
    embedding_index=EmbeddingIndexType.hnsw,
    embedding_index_ops="vector_cosine_ops",
    ef_construction=200,
    m=48,
)
```

### IVFFlat index

```python
from langchain_postgres import PGVector, EmbeddingIndexType

PGVector(
    collection_name="test_collection",
    embeddings=FakeEmbedding(),
    connection=CONNECTION_STRING,
    embedding_length=1536,
    embedding_index=EmbeddingIndexType.ivfflat,
    embedding_index_ops="vector_cosine_ops",
)
```

- Embedding length is required for HNSW index.
- Allowed values for `embedding_index_ops` are described in the [pgvector IVFFlat](https://github.com/pgvector/pgvector?tab=readme-ov-file#ivfflat).

### Binary Quantization

```python
from langchain_postgres import PGVector, EmbeddingIndexType

PGVector(
    collection_name="test_collection",
    embeddings=FakeEmbedding(),
    connection=CONNECTION_STRING,
    embedding_length=1536,
    embedding_index=EmbeddingIndexType.hnsw,
    embedding_index_ops="bit_hamming_ops",
    binary_quantization=True,
    binary_limit=200,
)
```

- Works only with HNSW index with `bit_hamming_ops`.
- `binary_limit` increases the limit in the inner binary search. A higher value will increase the recall at the cost of speed.

Refer to the [pgvector Binary Quantization](https://github.com/pgvector/pgvector?tab=readme-ov-file#binary-quantization) to better understand.

### Partitioning

```python
from langchain_postgres import PGVector

PGVector(
    collection_name="test_collection",
    embeddings=FakeEmbedding(),
    connection=CONNECTION_STRING,
    enable_partitioning=True,
)
```

- Create partitions of `langchain_pg_embedding` table by `collection_id`. Useful with a large number of embeddings with different collection.

Refer to the [pgvector Partitioning](https://github.com/pgvector/pgvector?tab=readme-ov-file#filtering)

### Iterative Scan

```python
from langchain_postgres import PGVector, EmbeddingIndexType, IterativeScan

PGVector(
    collection_name="test_collection",
    embeddings=FakeEmbedding(),
    connection=CONNECTION_STRING,
    embedding_length=1536,
    embedding_index=EmbeddingIndexType.hnsw,
    embedding_index_ops="vector_cosine_ops",
    iterative_scan=IterativeScan.relaxed_order
)
```

- `iterative_scan` can be set to `IterativeScan.relaxed_order` or `IterativeScan.strict_order` or disabled with `IterativeScan.off`.
- Requires an HNSW or IVFFlat index.

Refer to the [pgvector Iterative Scan](https://github.com/pgvector/pgvector?tab=readme-ov-file#iterative-index-scans) to better understand.

### Iterative Scan Options for HNSW index

```python
from langchain_postgres import PGVector, EmbeddingIndexType, IterativeScan

PGVector(
    collection_name="test_collection",
    embeddings=FakeEmbedding(),
    connection=CONNECTION_STRING,
    embedding_length=1536,
    embedding_index=EmbeddingIndexType.hnsw,
    embedding_index_ops="vector_cosine_ops",
    iterative_scan=IterativeScan.relaxed_order,
    max_scan_tuples=40000,
    scan_mem_multiplier=2
)
```

- `max_scan_tuples` control when the scan ends when `iterative_scan` is enabled.
- `scan_mem_multiplier` specify the max amount of memory to use for the scan.

Refer to the [pgvector Iterative Scan Options](https://github.com/pgvector/pgvector?tab=readme-ov-file#iterative-scan-options) to better understand.

### Full Text Search

Can be used by specifying `full_text_search` parameter.

```python
from langchain_postgres import PGVector

vectorstore = PGVector(
    collection_name="test_collection",
    embeddings=FakeEmbedding(),
    connection=CONNECTION_STRING,
)

vectorstore.similarity_search(
    "hello world",
    full_text_search=["foo", "bar & baz"]
)
```

This adds the following statement to the `WHERE` clause:
```sql
AND document_vector @@ to_tsquery('foo | bar & baz')
```

Can be used with retrievers like this: 
```python
from langchain_postgres import PGVector

vectorstore = PGVector(
    collection_name="test_collection",
    embeddings=FakeEmbedding(),
    connection=CONNECTION_STRING,
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "full_text_search": ["foo", "bar & baz"]
    }
)
```

Refer to Postgres [Full Text Search](https://www.postgresql.org/docs/current/textsearch.html) for more information.

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
