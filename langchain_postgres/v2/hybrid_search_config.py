from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from sqlalchemy import RowMapping

from .indexes import DistanceStrategy


def _normalize_scores(
    results: Sequence[dict[str, Any]], is_distance_metric: bool
) -> Sequence[dict[str, Any]]:
    """Normalizes scores to a 0-1 scale, where 1 is best."""
    if not results:
        return []

    # Get scores from the last column of each result
    scores = [float(list(item.values())[-1]) for item in results]
    min_score, max_score = min(scores), max(scores)
    score_range = max_score - min_score

    if score_range == 0:
        # All documents are of the highest quality (1.0)
        for item in results:
            item["normalized_score"] = 1.0
        return list(results)

    for item in results:
        # Access the score again from the last column for calculation
        score = list(item.values())[-1]
        normalized = (score - min_score) / score_range
        if is_distance_metric:
            # For distance, a lower score is better, so we invert the result.
            item["normalized_score"] = 1.0 - normalized
        else:
            # For similarity (like keyword search), a higher score is better.
            item["normalized_score"] = normalized

    return list(results)


def weighted_sum_ranking(
    primary_search_results: Sequence[RowMapping],
    secondary_search_results: Sequence[RowMapping],
    primary_results_weight: float = 0.5,
    secondary_results_weight: float = 0.5,
    fetch_top_k: int = 4,
    **kwargs: Any,
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

    distance_strategy = kwargs.get(
        "distance_strategy", DistanceStrategy.COSINE_DISTANCE
    )
    is_primary_distance = distance_strategy != DistanceStrategy.INNER_PRODUCT

    # Normalize both sets of results onto a 0-1 scale
    normalized_primary = _normalize_scores(
        [dict(row) for row in primary_search_results],
        is_distance_metric=is_primary_distance,
    )

    # Keyword search relevance is a similarity score (higher is better)
    normalized_secondary = _normalize_scores(
        [dict(row) for row in secondary_search_results], is_distance_metric=False
    )

    # stores computed metric with provided distance metric and weights
    weighted_scores: dict[str, dict[str, Any]] = {}

    # Process primary results
    for item in normalized_primary:
        doc_id = str(list(item.values())[0])
        # Set the 'distance' key with the weighted primary score
        item["distance"] = item["normalized_score"] * primary_results_weight
        weighted_scores[doc_id] = item

    # Process secondary results
    for item in normalized_secondary:
        doc_id = str(list(item.values())[0])
        secondary_weighted_score = item["normalized_score"] * secondary_results_weight

        if doc_id in weighted_scores:
            # Add to the existing 'distance' score
            weighted_scores[doc_id]["distance"] += secondary_weighted_score
        else:
            # Set the 'distance' key for the new item
            item["distance"] = secondary_weighted_score
            weighted_scores[doc_id] = item

    ranked_results = sorted(
        weighted_scores.values(), key=lambda item: item["distance"], reverse=True
    )

    for result in ranked_results:
        result.pop("normalized_score", None)

    return ranked_results[:fetch_top_k]


def reciprocal_rank_fusion(
    primary_search_results: Sequence[RowMapping],
    secondary_search_results: Sequence[RowMapping],
    rrf_k: float = 60,
    fetch_top_k: int = 4,
    **kwargs: Any,
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
    distance_strategy = kwargs.get(
        "distance_strategy", DistanceStrategy.COSINE_DISTANCE
    )
    rrf_scores: dict[str, dict[str, Any]] = {}

    # Process results from primary source
    # Determine sorting order based on the vector distance strategy.
    # For COSINE & EUCLIDEAN(distance), we sort ascending (reverse=False).
    # For INNER_PRODUCT (similarity), we sort descending (reverse=True).
    is_similarity_metric = distance_strategy == DistanceStrategy.INNER_PRODUCT
    sorted_primary = sorted(
        primary_search_results,
        key=lambda item: item["distance"],
        reverse=is_similarity_metric,
    )

    for rank, row in enumerate(sorted_primary):
        doc_id = str(list(row.values())[0])
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = dict(row)
            rrf_scores[doc_id]["distance"] = 0.0
        # Add the "normalized" rank score
        rrf_scores[doc_id]["distance"] += 1.0 / (rank + rrf_k)

    # Process results from secondary source
    # Keyword search relevance is always "higher is better" -> sort descending
    sorted_secondary = sorted(
        secondary_search_results,
        key=lambda item: item["distance"],
        reverse=True,
    )

    for rank, row in enumerate(sorted_secondary):
        doc_id = str(list(row.values())[0])
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = dict(row)
            rrf_scores[doc_id]["distance"] = 0.0
        # Add the rank score from this list to the existing score
        rrf_scores[doc_id]["distance"] += 1.0 / (rank + rrf_k)

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
