# Migrate a `PGVector` vector store to `PGVectorStore`

This guide shows how to migrate from the [`PGVector`](https://github.com/langchain-ai/langchain-postgres/blob/main/langchain_postgres/vectorstores.py) vector store class to the [`PGVectorStore`](https://github.com/langchain-ai/langchain-postgres/blob/main/langchain_postgres/vectorstore.py) class.

## Why migrate?

This guide explains how to migrate your vector data from a PGVector-style database (two tables) to an PGVectoStore-style database (one table per collection) for improved performance and manageability.

Migrating to the PGVectorStore interface provides the following benefits:

- **Simplified management**: A single table contains data corresponding to a single collection, making it easier to query, update, and maintain.
- **Improved metadata handling**: It stores metadata in columns instead of JSON, resulting in significant performance improvements.
- **Schema flexibility**: The interface allows users to add tables into any database schema.
- **Improved performance**: The single-table schema can lead to faster query execution, especially for large collections.
- **Clear separation**: Clearly separate table and extension creation, allowing for distinct permissions and streamlined workflows.
- **Secure Connections:** The PGVectorStore interface creates a secure connection pool that can be easily shared across your application using the `engine` object.

## Migration process

> **_NOTE:_**  The langchain-core library is installed to use the Fake embeddings service. To use a different embedding service, you'll need to install the appropriate library for your chosen provider. Choose embeddings services from [LangChain's Embedding models](https://python.langchain.com/v0.2/docs/integrations/text_embedding/).

While you can use the  existing PGVector database, we **strongly recommend** migrating your data to the PGVectorStore-style schema to take full advantage of the performance benefits.

### (Recommended) Data migration

1. **Create a PG engine.**

    ```python
    from langchain_postgres import PGEngine

    # Replace these variable values
    engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
    ```

    > **_NOTE:_** All sync methods have corresponding async methods.

2. **Create a new table to migrate existing data.**

    ```python
    # Vertex AI embeddings uses a vector size of 768.
    # Adjust this according to your embeddings service.
    VECTOR_SIZE = 768

    engine.init_vectorstore_table(
        table_name="destination_table",
        vector_size=VECTOR_SIZE,
    )
    ```

    **(Optional) Customize your table.**

    When creating your vectorstore table, you have the flexibility to define custom metadata and ID columns. This is particularly useful for:

    - **Filtering**: Metadata columns allow you to easily filter your data within the vectorstore. For example, you might store the document source, date, or author as metadata for efficient retrieval.
    - **Non-UUID Identifiers**: By default, the id_column uses UUIDs. If you need to use a different type of ID (e.g., an integer or string), you can define a custom id_column.

    ```python
    metadata_columns = [
        Column(f"col_0_{collection_name}", "VARCHAR"),
        Column(f"col_1_{collection_name}", "VARCHAR"),
    ]
    engine.init_vectorstore_table(
        table_name="destination_table",
        vector_size=VECTOR_SIZE,
        metadata_columns=metadata_columns,
        id_column=Column("langchain_id", "VARCHAR"),
    )
    ```

3. **Create a vector store object to interact with the new data.**

    > **_NOTE:_** The `FakeEmbeddings` embedding service is only used to initialise a vector store object, not to generate any embeddings. The embeddings are copied directly from the PGVector table.

    ```python
    from langchain_postgres import PGVectorStore
    from langchain_core.embeddings import FakeEmbeddings

    destination_vector_store = PGVectorStore.create_sync(
        engine,
        embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
        table_name="destination_table",
    )
    ```

    If you have any customisations on the metadata or the id columns, add them to the vector store as follows:

    ```python
    from langchain_postgres import PGVectorStore
    from langchain_core.embeddings import FakeEmbeddings

    destination_vector_store = PGVectorStore.create_sync(
        engine,
        embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
        table_name="destination_table",
        metadata_columns=[col.name for col in metadata_columns],
        id_column="langchain_id",
    )
    ```

4. **Migrate the data to the new table.**

    ```python
    from langchain_postgres.utils.pgvector_migrator import amigrate_pgvector_collection

    migrate_pgvector_collection(
        engine,
        # Set collection name here
        collection_name="collection_name",
        vector_store=destination_vector_store,
        # This deletes data from the original table upon migration. You can choose to turn it off.
        delete_pg_collection=True,
    )
    ```

    The data will only be deleted from the original table once all of it has been successfully copied to the destination table.

> **TIP:** If you would like to migrate multiple collections, you can use the `alist_pgvector_collection_names` method to get the names of all collections, allowing you to iterate through them.
>
> ```python
> from langchain_postgres.utils.pgvector_migrator import alist_pgvector_collection_names
>
> all_collection_names = list_pgvector_collection_names(engine)
> print(all_collection_names)
> ```

### (Not Recommended) Use PGVectorStore interface on PGVector databases

If you choose not to migrate your data, you can still use the PGVectorStore interface with your existing PGVector database. However, you won't benefit from the performance improvements of the PGVectorStore-style schema.

1. **Create an PGVectorStore engine.**

    ```python
    from langchain_postgres import PGEngine

    # Replace these variable values
    engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
    ```

    > **_NOTE:_** All sync methods have corresponding async methods.

2. **Create a vector store object to interact with the data.**

    Use the embeddings service used by your database. See [langchain docs](https://python.langchain.com/docs/integrations/text_embedding/) for reference.

    ```python
    from langchain_postgres import PGVectorStore
    from langchain_core.embeddings import FakeEmbeddings

    vector_store = PGVectorStore.create_sync(
        engine=engine,
        table_name="langchain_pg_embedding",
        embedding_service=FakeEmbeddings(size=VECTOR_SIZE),
        content_column="document",
        metadata_json_column="cmetadata",
        metadata_columns=["collection_id"],
        id_column="id",
    )
    ```

3. **Perform similarity search.**

    Filter by collection id:

    ```python
    vector_store.similarity_search("query", k=5, filter=f"collection_id='{uuid}'")
    ```

    Filter by collection id and metadata:

    ```python
    vector_store.similarity_search(
        "query", k=5, filter=f"collection_id='{uuid}' and cmetadata->>'col_name' = 'value'"
    )
    ```