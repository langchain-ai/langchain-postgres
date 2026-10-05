"""Length validation for the sequences `add_embeddings` zips together."""

import pytest

from langchain_postgres.vectorstores import PGVector


def test_mismatched_embeddings_raise() -> None:
    """A short `embeddings` list must raise instead of writing a partial batch.

    The insert payload is zipped from four parallel sequences, so a provider
    that quietly returns fewer vectors than texts used to write only part of
    the batch while the full `ids` list was still returned to the caller.
    """
    with pytest.raises(ValueError, match="Got 1 embeddings for 3 texts"):
        PGVector._validate_parallel_lengths(
            ["a", "b", "c"], [[0.1]], [{}, {}, {}], ["1", "2", "3"]
        )


def test_mismatched_metadatas_raise() -> None:
    with pytest.raises(ValueError, match="Got 2 metadatas for 3 texts"):
        PGVector._validate_parallel_lengths(
            ["a", "b", "c"], [[0.1], [0.2], [0.3]], [{}, {}], ["1", "2", "3"]
        )


def test_mismatched_ids_raise() -> None:
    with pytest.raises(ValueError, match="Got 2 ids for 3 texts"):
        PGVector._validate_parallel_lengths(
            ["a", "b", "c"], [[0.1], [0.2], [0.3]], [{}, {}, {}], ["1", "2"]
        )


def test_matching_lengths_pass() -> None:
    PGVector._validate_parallel_lengths(
        ["a", "b", "c"], [[0.1], [0.2], [0.3]], [{}, {}, {}], ["1", "2", "3"]
    )


def test_empty_batch_passes() -> None:
    PGVector._validate_parallel_lengths([], [], [], [])
