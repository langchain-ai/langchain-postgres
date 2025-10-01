from typing import cast

import pytest
from sqlalchemy import RowMapping

from langchain_postgres.v2.hybrid_search_config import (
    reciprocal_rank_fusion,
    weighted_sum_ranking,
)
from langchain_postgres.v2.indexes import DistanceStrategy


# Helper to create mock input items that mimic RowMapping for the fusion functions
def get_row(doc_id: str, score: float, content: str = "content") -> RowMapping:
    """
    Simulates a RowMapping-like dictionary.
    The fusion functions expect to extract doc_id as the first value and
    the initial score/distance as the last value when casting values from RowMapping.
    They then operate on dictionaries, using the 'distance' key for the fused score.
    """
    # Python dicts maintain insertion order (Python 3.7+).
    # This structure ensures list(row.values())[0] is doc_id and
    # list(row.values())[-1] is score.
    row_dict = {"id_val": doc_id, "content_field": content, "distance": score}
    return cast(RowMapping, row_dict)


class TestWeightedSumRanking:
    def test_empty_inputs(self) -> None:
        results = weighted_sum_ranking([], [])
        assert results == []

    def test_primary_only(self) -> None:
        primary = [get_row("p1", 0.8), get_row("p2", 0.6)]
        # Expected scores: p1 = 0.8 * 0.5 = 0.4, p2 = 0.6 * 0.5 = 0.3
        results = weighted_sum_ranking(  # type: ignore
            primary,  # type: ignore
            [],
            primary_results_weight=0.5,
            secondary_results_weight=0.5,
        )
        assert len(results) == 2
        assert results[0]["id_val"] == "p1"
        assert results[0]["distance"] == pytest.approx(0.4)
        assert results[1]["id_val"] == "p2"
        assert results[1]["distance"] == pytest.approx(0.3)

    def test_secondary_only(self) -> None:
        secondary = [get_row("s1", 0.9), get_row("s2", 0.7)]
        # Expected scores: s1 = 0.9 * 0.5 = 0.45, s2 = 0.7 * 0.5 = 0.35
        results = weighted_sum_ranking(
            [],
            secondary,  # type: ignore
            primary_results_weight=0.5,
            secondary_results_weight=0.5,
        )
        assert len(results) == 2
        assert results[0]["id_val"] == "s1"
        assert results[0]["distance"] == pytest.approx(0.45)
        assert results[1]["id_val"] == "s2"
        assert results[1]["distance"] == pytest.approx(0.35)

    def test_mixed_results_default_weights(self) -> None:
        primary = [get_row("common", 0.8), get_row("p_only", 0.7)]
        secondary = [get_row("common", 0.9), get_row("s_only", 0.6)]
        # Weights are 0.5, 0.5
        # common_score = (0.8 * 0.5) + (0.9 * 0.5) = 0.4 + 0.45 = 0.85
        # p_only_score = (0.7 * 0.5) = 0.35
        # s_only_score = (0.6 * 0.5) = 0.30
        # Order: common (0.85), p_only (0.35), s_only (0.30)

        results = weighted_sum_ranking(primary, secondary)  # type: ignore
        assert len(results) == 3
        assert results[0]["id_val"] == "common"
        assert results[0]["distance"] == pytest.approx(0.85)
        assert results[1]["id_val"] == "p_only"
        assert results[1]["distance"] == pytest.approx(0.35)
        assert results[2]["id_val"] == "s_only"
        assert results[2]["distance"] == pytest.approx(0.30)

    def test_mixed_results_custom_weights(self) -> None:
        primary = [get_row("d1", 1.0)]  # p_w=0.2 -> 0.2
        secondary = [get_row("d1", 0.5)]  # s_w=0.8 -> 0.4
        # Expected: d1_score = (1.0 * 0.2) + (0.5 * 0.8) = 0.2 + 0.4 = 0.6

        results = weighted_sum_ranking(
            primary,  # type: ignore
            secondary,  # type: ignore
            primary_results_weight=0.2,
            secondary_results_weight=0.8,
        )
        assert len(results) == 1
        assert results[0]["id_val"] == "d1"
        assert results[0]["distance"] == pytest.approx(0.6)

    def test_fetch_top_k(self) -> None:
        primary = [get_row(f"p{i}", (10 - i) / 10.0) for i in range(5)]
        # Scores: 1.0, 0.9, 0.8, 0.7, 0.6
        # Weighted (0.5): 0.5, 0.45, 0.4, 0.35, 0.3
        results = weighted_sum_ranking(primary, [], fetch_top_k=2)  # type: ignore
        assert len(results) == 2
        assert results[0]["id_val"] == "p0"
        assert results[0]["distance"] == pytest.approx(0.5)
        assert results[1]["id_val"] == "p1"
        assert results[1]["distance"] == pytest.approx(0.45)


class TestReciprocalRankFusion:
    def test_empty_inputs(self) -> None:
        """Tests that the function handles empty inputs gracefully."""
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_primary_only(self) -> None:
        """Tests RRF with only primary results using default cosine (lower is better)."""
        primary = [get_row("p1", 0.8), get_row("p2", 0.6)]
        rrf_k = 60
        # --- Calculation (Cosine: lower is better) ---
        # Sorted order: p2 (0.6) -> rank 0; p1 (0.8) -> rank 1
        # p2_score = 1 / (0 + 60)
        # p1_score = 1 / (1 + 60)
        results = reciprocal_rank_fusion(primary, [], rrf_k=rrf_k)  # type: ignore
        assert len(results) == 2
        assert results[0]["id_val"] == "p2"
        assert results[0]["distance"] == pytest.approx(1.0 / (0 + rrf_k))
        assert results[1]["id_val"] == "p1"
        assert results[1]["distance"] == pytest.approx(1.0 / (1 + rrf_k))

    def test_secondary_only(self) -> None:
        """Tests RRF with only secondary results (higher is better)."""
        secondary = [get_row("s1", 0.9), get_row("s2", 0.7)]
        rrf_k = 60
        # --- Calculation (Keyword: higher is better) ---
        # Sorted order: s1 (0.9) -> rank 0; s2 (0.7) -> rank 1
        results = reciprocal_rank_fusion([], secondary, rrf_k=rrf_k)  # type: ignore
        assert len(results) == 2
        assert results[0]["id_val"] == "s1"
        assert results[0]["distance"] == pytest.approx(1.0 / (0 + rrf_k))
        assert results[1]["id_val"] == "s2"
        assert results[1]["distance"] == pytest.approx(1.0 / (1 + rrf_k))

    def test_mixed_results_default_k(self) -> None:
        """Tests fusion with default cosine (lower better) and keyword (higher better)."""
        primary = [
            get_row("common", 0.8),
            get_row("p_only", 0.7),
        ]  # Order: p_only, common
        secondary = [
            get_row("common", 0.9),
            get_row("s_only", 0.6),
        ]  # Order: common, s_only
        rrf_k = 60
        # --- Calculation ---
        # common: rank 1 in P (1/61) + rank 0 in S (1/60) -> highest score
        # p_only: rank 0 in P (1/60)
        # s_only: rank 1 in S (1/61)
        results = reciprocal_rank_fusion(primary, secondary, rrf_k=rrf_k)  # type: ignore
        assert len(results) == 3
        assert results[0]["id_val"] == "common"
        assert results[0]["distance"] == pytest.approx(1 / 61 + 1 / 60)
        assert results[1]["id_val"] == "p_only"
        assert results[1]["distance"] == pytest.approx(1 / 60)
        assert results[2]["id_val"] == "s_only"
        assert results[2]["distance"] == pytest.approx(1 / 61)

    def test_fetch_top_k_rrf(self) -> None:
        """Tests that fetch_top_k limits results correctly after fusion."""
        # Using cosine distance (lower is better)
        primary = [get_row(f"p{i}", (10 - i) / 10.0) for i in range(5)]
        # Scores: [1.0, 0.9, 0.8, 0.7, 0.6]
        # Sorted order: p4 (0.6), p3 (0.7), p2 (0.8), ...
        results = reciprocal_rank_fusion(primary, [], fetch_top_k=2)  # type: ignore
        assert len(results) == 2
        assert results[0]["id_val"] == "p4"
        assert results[1]["id_val"] == "p3"

    def test_rrf_content_preservation(self) -> None:
        """Tests that the data from the first time a document is seen is kept."""
        primary = [get_row("doc1", 0.9, content="Primary Content")]
        secondary = [get_row("doc1", 0.8, content="Secondary Content")]
        # RRF processes primary first. When "doc1" is seen, its data is stored.
        # It will not be overwritten by the "doc1" from the secondary list.
        results = reciprocal_rank_fusion(primary, secondary)  # type: ignore
        assert len(results) == 1
        assert results[0]["id_val"] == "doc1"
        assert results[0]["content_field"] == "Primary Content"

        # If only in secondary
        results_prim_only = reciprocal_rank_fusion([], secondary, rrf_k=60)  # type: ignore
        assert results_prim_only[0]["content_field"] == "Secondary Content"

    def test_reordering_from_inputs_rrf(self) -> None:
        """Tests that RRF can produce a ranking different from the inputs."""
        primary = [get_row("docA", 0.9), get_row("docB", 0.8), get_row("docC", 0.1)]
        secondary = [get_row("docC", 0.9), get_row("docB", 0.5), get_row("docA", 0.2)]
        rrf_k = 1.0
        # --- Calculation (Primary sorted ascending, Secondary descending) ---
        # Primary ranks: docC (0), docB (1), docA (2)
        # Secondary ranks: docC (0), docB (1), docA (2)
        # docC_score = 1/(0+1) [P] + 1/(0+1) [S] = 2.0
        # docB_score = 1/(1+1) [P] + 1/(1+1) [S] = 1.0
        # docA_score = 1/(2+1) [P] + 1/(2+1) [S] = 2/3
        results = reciprocal_rank_fusion(primary, secondary, rrf_k=rrf_k)  # type: ignore
        assert len(results) == 3
        assert results[0]["id_val"] == "docC"
        assert results[0]["distance"] == pytest.approx(2.0)
        assert results[1]["id_val"] == "docB"
        assert results[1]["distance"] == pytest.approx(1.0)
        assert results[2]["id_val"] == "docA"
        assert results[2]["distance"] == pytest.approx(2.0 / 3.0)

    # --------------------------------------------------------------------------
    ## New Tests for Other Strategies and Edge Cases

    def test_mixed_results_max_inner_product(self) -> None:
        """Tests fusion with MAX_INNER_PRODUCT (higher is better) for primary."""
        primary = [get_row("best", 0.9), get_row("worst", 0.1)]  # Order: best, worst
        secondary = [get_row("best", 20.0), get_row("worst", 5.0)]  # Order: best, worst
        rrf_k = 10
        # best: rank 0 in P + rank 0 in S -> 1/10 + 1/10 = 0.2
        # worst: rank 1 in P + rank 1 in S -> 1/11 + 1/11
        results = reciprocal_rank_fusion(
            primary,  # type: ignore
            secondary,  # type: ignore
            rrf_k=rrf_k,
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        assert len(results) == 2
        assert results[0]["id_val"] == "best"
        assert results[0]["distance"] == pytest.approx(0.2)
        assert results[1]["id_val"] == "worst"
        assert results[1]["distance"] == pytest.approx(2.0 / 11.0)

    def test_mixed_results_euclidean(self) -> None:
        """Tests fusion with EUCLIDEAN (lower is better) for primary."""
        primary = [
            get_row("closer", 10.5),
            get_row("farther", 25.5),
        ]  # Order: closer, farther
        secondary = [
            get_row("closer", 100.0),
            get_row("farther", 10.0),
        ]  # Order: closer, farther
        rrf_k = 10
        # closer: rank 0 in P + rank 0 in S -> 1/10 + 1/10 = 0.2
        # farther: rank 1 in P + rank 1 in S -> 1/11 + 1/11
        results = reciprocal_rank_fusion(
            primary,  # type: ignore
            secondary,  # type: ignore
            rrf_k=rrf_k,
            distance_strategy=DistanceStrategy.EUCLIDEAN,
        )
        assert len(results) == 2
        assert results[0]["id_val"] == "closer"
        assert results[0]["distance"] == pytest.approx(0.2)
        assert results[1]["id_val"] == "farther"
        assert results[1]["distance"] == pytest.approx(2.0 / 11.0)

    def test_rrf_with_identical_scores(self) -> None:
        """Tests that stable sort is preserved for identical scores."""
        # Python's sorted() is stable. p1 appears before p2 in the list.
        primary = [get_row("p1", 0.5), get_row("p2", 0.5)]
        rrf_k = 60
        # Expected order (stable sort): p1 (rank 0), p2 (rank 1)
        results = reciprocal_rank_fusion(primary, [])  # type: ignore
        assert results[0]["id_val"] == "p1"
        assert results[0]["distance"] == pytest.approx(1 / 60)
        assert results[1]["id_val"] == "p2"
        assert results[1]["distance"] == pytest.approx(1 / 61)

    def test_reordering_from_inputs_weighted_sum(self) -> None:
        """Tests that the fused ranking can be different from the inputs."""
        primary = [get_row("docA", 0.9), get_row("docB", 0.7)]
        secondary = [get_row("docB", 0.8), get_row("docA", 0.2)]
        # --- Calculation with normalization ---
        # Primary norm (inverted): docA=0.0, docB=1.0
        # Secondary norm: docB=1.0, docA=0.0
        # Weighted (0.5/0.5):
        # docA_score = (0.0 * 0.5) + (0.0 * 0.5) = 0.0
        # docB_score = (1.0 * 0.5) + (1.0 * 0.5) = 1.0
        results = weighted_sum_ranking(primary, secondary)
        assert len(results) == 2
        assert results[0]["id_val"] == "docB"
        assert results[0]["distance"] == pytest.approx(1.0)
        assert results[1]["id_val"] == "docA"
        assert results[1]["distance"] == pytest.approx(0.0)
