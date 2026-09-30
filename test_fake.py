with open("tests/unit_tests/v2/test_async_pg_vectorstore.py", "r") as f:
    content = f.read()

new_test = """
    async def test_init_does_not_raise_for_standard_embeddings(self) -> None:
        from langchain_core.embeddings import DeterministicFakeEmbedding
        create_key = getattr(AsyncPGVectorStore, "_AsyncPGVectorStore__create_key")
        
        # Should not raise ValueError because DeterministicFakeEmbedding 
        # doesn't implement embed_query_inline at all
        vs = AsyncPGVectorStore(
            create_key,
            engine=None,  # type: ignore
            embedding_service=DeterministicFakeEmbedding(size=2),
            table_name="test_table",
        )
        assert vs.embedding_service is not None
"""

content = content + new_test
with open("tests/unit_tests/v2/test_async_pg_vectorstore.py", "w") as f:
    f.write(content)
