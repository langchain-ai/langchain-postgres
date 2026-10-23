"""`from_texts`/`from_documents` must tolerate the extra kwargs their signature accepts.

`VectorStore.from_texts` is declared with `**kwargs`, but the sync overrides
forwarded them into `create_sync`, which accepts none — so any extra keyword
raised `TypeError` while the async siblings ignored it.
"""

from unittest.mock import MagicMock, patch

from langchain_postgres.v2.vectorstores import PGVectorStore


def test_from_texts_tolerates_extra_kwargs() -> None:
    store = MagicMock()
    with patch.object(PGVectorStore, "create_sync", return_value=store) as create:
        PGVectorStore.from_texts(
            texts=["a"],
            embedding=MagicMock(),
            engine=MagicMock(),
            table_name="t",
            some_provider_flag=1,
        )

    assert "some_provider_flag" not in create.call_args.kwargs
    store.add_texts.assert_called_once()


def test_from_documents_tolerates_extra_kwargs() -> None:
    store = MagicMock()
    with patch.object(PGVectorStore, "create_sync", return_value=store) as create:
        PGVectorStore.from_documents(
            documents=[],
            embedding=MagicMock(),
            engine=MagicMock(),
            table_name="t",
            some_provider_flag=1,
        )

    assert "some_provider_flag" not in create.call_args.kwargs
    store.add_documents.assert_called_once()
