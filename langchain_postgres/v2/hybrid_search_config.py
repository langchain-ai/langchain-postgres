from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from sqlalchemy import RowMapping


def weighted_sum_ranking(
    primary_search_results: Sequence[RowMapping],
    secondary_search_results: Sequence[RowMapping],
    primary_results_weight: float = 0.5,
    secondary_results_weight: float = 0.5,
    fetch_top_k: int = 4,
) -> Sequence[dict[str, Any]]:
    """
    Ranks documents using a weighted sum of scores from two sources.

    Args:
        primary_search_results: A list of (document, distance) tuples from
            the primary search.
        secondary_search_results: A list of (document, distance) tuples from
            the secondary search.
        primary_results_weight: The weight for the primary source's scores.
            Defaults to 0.5.
        secondary_results_weight: The weight for the secondary source's scores.
            Defaults to 0.5.
        fetch_top_k: The number of documents to fetch after merging the results.
            Defaults to 4.

    Returns:
        A list of (document, distance) tuples, sorted by weighted_score in
        descending order.
    """

    # stores computed metric with provided distance metric and weights
    weighted_scores: dict[str, dict[str, Any]] = {}

    # Process results from primary source
    for row in primary_search_results:
        values = list(row.values())
        doc_id = str(values[0])  # first value is doc_id
        distance = float(values[-1])  # type: ignore # last value is distance
        row_values = dict(row)
        row_values["distance"] = primary_results_weight * distance
        weighted_scores[doc_id] = row_values

    # Process results from secondary source,
    # adding to existing scores or creating new ones
    for row in secondary_search_results:
        values = list(row.values())
        doc_id = str(values[0])  # first value is doc_id
        distance = float(values[-1])  # type: ignore # last value is distance
        primary_score = (
            weighted_scores[doc_id]["distance"] if doc_id in weighted_scores else 0.0
        )
        row_values = dict(row)
        row_values["distance"] = distance * secondary_results_weight + primary_score
        weighted_scores[doc_id] = row_values

    # Sort the results by weighted score in descending order
    ranked_results = sorted(
        weighted_scores.values(), key=lambda item: item["distance"], reverse=True
    )
    return ranked_results[:fetch_top_k]


def reciprocal_rank_fusion(
    primary_search_results: Sequence[RowMapping],
    secondary_search_results: Sequence[RowMapping],
    rrf_k: float = 60,
    fetch_top_k: int = 4,
) -> Sequence[dict[str, Any]]:
    """
    Ranks documents using Reciprocal Rank Fusion (RRF) of scores from two sources.

    Args:
        primary_search_results: A list of (document, distance) tuples from
            the primary search.
        secondary_search_results: A list of (document, distance) tuples from
            the secondary search.
        rrf_k: The RRF parameter k.
            Defaults to 60.
        fetch_top_k: The number of documents to fetch after merging the results.
            Defaults to 4.

    Returns:
        A list of (document_id, rrf_score) tuples, sorted by rrf_score
        in descending order.
    """
    rrf_scores: dict[str, dict[str, Any]] = {}

    # Process results from primary source
    for rank, row in enumerate(
        sorted(primary_search_results, key=lambda item: item["distance"], reverse=True)
    ):
        values = list(row.values())
        doc_id = str(values[0])
        row_values = dict(row)
        primary_score = rrf_scores[doc_id]["distance"] if doc_id in rrf_scores else 0.0
        primary_score += 1.0 / (rank + rrf_k)
        row_values["distance"] = primary_score
        rrf_scores[doc_id] = row_values

    # Process results from secondary source
    for rank, row in enumerate(
        sorted(
            secondary_search_results, key=lambda item: item["distance"], reverse=True
        )
    ):
        values = list(row.values())
        doc_id = str(values[0])
        row_values = dict(row)
        secondary_score = (
            rrf_scores[doc_id]["distance"] if doc_id in rrf_scores else 0.0
        )
        secondary_score += 1.0 / (rank + rrf_k)
        row_values["distance"] = secondary_score
        rrf_scores[doc_id] = row_values

    # Sort the results by rrf score in descending order
    # Sort the results by weighted score in descending order
    ranked_results = sorted(
        rrf_scores.values(), key=lambda item: item["distance"], reverse=True
    )
    # Extract only the RowMapping for the top results
    return ranked_results[:fetch_top_k]


@dataclass
class HybridSearchConfig(ABC):
    """
    AlloyDB Vector Store Hybrid Search Config.

    Queries might be slow if the hybrid search column does not exist.
    For best hybrid search performance, consider creating a TSV column
    and adding GIN index.
    """

    tsv_column: Optional[str] = ""
    tsv_lang: Optional[str] = "pg_catalog.english"
    fts_query: Optional[str] = ""
    fusion_function: Callable[
        [Sequence[RowMapping], Sequence[RowMapping], Any], Sequence[Any]
    ] = weighted_sum_ranking  # Updated default
    fusion_function_parameters: dict[str, Any] = field(default_factory=dict)
    primary_top_k: int = 4
    secondary_top_k: int = 4
    index_name: str = "langchain_tsv_index"
    index_type: str = "GIN"
