import pytest

from langchain_postgres.v2.hybrid_search_config import (
    reciprocal_rank_fusion,
    weighted_sum_ranking,
)
from langchain_postgres.v2.indexes import DistanceStrategy


# Helper to create mock input items that mimic RowMapping for the fusion functions
def get_row(doc_id: str, score: float, content: str = "content") -> dict:
    """
    Simulates a RowMapping-like dictionary.
    The fusion functions expect to extract doc_id as the first value and
    the initial score/distance as the last value when casting values from RowMapping.
    They then operate on dictionaries, using the 'distance' key for the fused score.
    """
    # Python dicts maintain insertion order (Python 3.7+).
    # This structure ensures list(row.values())[0] is doc_id and
    # list(row.values())[-1] is score.
    return {"id_val": doc_id, "content_field": content, "distance": score}


class TestWeightedSumRanking:
    def test_empty_inputs(self) -> None:
        """Tests that the function handles empty inputs gracefully."""
        results = weighted_sum_ranking([], [])
        assert results == []

    def test_primary_only_cosine_default(self) -> None:
        """Tests ranking with only primary results using default cosine distance."""
        primary = [get_row("p1", 0.8), get_row("p2", 0.6)]
        # --- Calculation (Cosine = lower is better) ---
        # Scores: [0.8, 0.6]. Range: 0.2. Min: 0.6.
        # p1 norm: 1.0 - ((0.8 - 0.6) / 0.2) = 0.0
        # p2 norm: 1.0 - ((0.6 - 0.6) / 0.2) = 1.0
        # Weighted (0.5): p1 = 0.0, p2 = 0.5
        # Order: p2, p1
        results = weighted_sum_ranking(
            primary,  # type: ignore
            [],
        )
        assert len(results) == 2
        assert results[0]["id_val"] == "p2"
        assert results[0]["distance"] == pytest.approx(0.5)
        assert results[1]["id_val"] == "p1"
        assert results[1]["distance"] == pytest.approx(0.0)

    def test_secondary_only(self) -> None:
        """Tests ranking with only secondary (keyword) results."""
        secondary = [get_row("s1", 15.0), get_row("s2", 5.0)]
        # --- Calculation (Keyword = higher is better) ---
        # Scores: [15.0, 5.0]. Range: 10.0. Min: 5.0.
        # s1 norm: (15.0 - 5.0) / 10.0 = 1.0
        # s2 norm: (5.0 - 5.0) / 10.0 = 0.0
        # Weighted (0.5): s1 = 0.5, s2 = 0.0
        # Order: s1, s2
        results = weighted_sum_ranking(
            [],
            secondary,  # type: ignore
        )
        assert len(results) == 2
        assert results[0]["id_val"] == "s1"
        assert results[0]["distance"] == pytest.approx(0.5)
        assert results[1]["id_val"] == "s2"
        assert results[1]["distance"] == pytest.approx(0.0)

    def test_mixed_results_cosine(self) -> None:
        """Tests combining cosine (lower is better) and keyword (higher is better) scores."""
        primary = [get_row("common", 0.8), get_row("p_only", 0.7)]
        secondary = [get_row("common", 9.0), get_row("s_only", 6.0)]
        # --- Calculation ---
        # Primary norm (inverted): common=0.0, p_only=1.0
        # Secondary norm: common=1.0, s_only=0.0
        # Weighted (0.5):
        # common = (0.0 * 0.5) + (1.0 * 0.5) = 0.5
        # p_only = (1.0 * 0.5) + 0 = 0.5
        # s_only = 0 + (0.0 * 0.5) = 0.0
        results = weighted_sum_ranking(
            primary,  # type: ignore
            secondary,  # type: ignore
        )
        assert len(results) == 3
        # Check that the top two results have the correct score and IDs (order may vary)
        top_ids = {res["id_val"] for res in results[:2]}
        assert top_ids == {"common", "p_only"}
        assert results[0]["distance"] == pytest.approx(0.5)
        assert results[1]["distance"] == pytest.approx(0.5)
        assert results[2]["id_val"] == "s_only"
        assert results[2]["distance"] == pytest.approx(0.0)

    def test_primary_max_inner_product(self) -> None:
        """Tests using MAX_INNER_PRODUCT (higher is better) for primary search."""
        primary = [get_row("best", 0.9), get_row("worst", 0.1)]
        secondary = [get_row("best", 20.0), get_row("worst", 5.0)]
        # --- Calculation ---
        # Primary norm (NOT inverted): best=1.0, worst=0.0
        # Secondary norm: best=1.0, worst=0.0
        # Weighted (0.5):
        # best = (1.0 * 0.5) + (1.0 * 0.5) = 1.0
        # worst = (0.0 * 0.5) + (0.0 * 0.5) = 0.0
        results = weighted_sum_ranking(
            primary,  # type: ignore
            secondary,  # type: ignore
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        assert len(results) == 2
        assert results[0]["id_val"] == "best"
        assert results[0]["distance"] == pytest.approx(1.0)
        assert results[1]["id_val"] == "worst"
        assert results[1]["distance"] == pytest.approx(0.0)

    def test_primary_euclidean(self) -> None:
        """Tests using EUCLIDEAN (lower is better) for primary search."""
        primary = [get_row("closer", 10.5), get_row("farther", 25.5)]
        secondary = [get_row("closer", 100.0), get_row("farther", 10.0)]
        # --- Calculation ---
        # Primary norm (inverted): closer=1.0, farther=0.0
        # Secondary norm: closer=1.0, farther=0.0
        # Weighted (0.5):
        # closer = (1.0 * 0.5) + (1.0 * 0.5) = 1.0
        # farther = (0.0 * 0.5) + (0.0 * 0.5) = 0.0
        results = weighted_sum_ranking(
            primary,  # type: ignore
            secondary,  # type: ignore
            distance_strategy=DistanceStrategy.EUCLIDEAN,
        )
        assert len(results) == 2
        assert results[0]["id_val"] == "closer"
        assert results[0]["distance"] == pytest.approx(1.0)
        assert results[1]["id_val"] == "farther"
        assert results[1]["distance"] == pytest.approx(0.0)

    def test_fetch_top_k(self) -> None:
        """Tests that fetch_top_k correctly limits the number of results."""
        primary = [get_row(f"p{i}", (10 - i) / 10.0) for i in range(5)]
        # p0=1.0, p1=0.9, p2=0.8, p3=0.7, p4=0.6
        # The best scores (lowest distance) are p4 and p3
        results = weighted_sum_ranking(
            primary,  # type: ignore
            [],
            fetch_top_k=2,
        )
        assert len(results) == 2
        assert results[0]["id_val"] == "p4"  # Has the best normalized score
        assert results[1]["id_val"] == "p3"


class TestReciprocalRankFusion:
    def test_empty_inputs(self) -> None:
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_primary_only(self) -> None:
        primary = [
            get_row("p1", 0.8),
            get_row("p2", 0.6),
        ]  # p1 rank 0, p2 rank 1
        rrf_k = 60
        # p1_score = 1 / (0 + 60)
        # p2_score = 1 / (1 + 60)
        results = reciprocal_rank_fusion(primary, [], rrf_k=rrf_k)  # type: ignore
        assert len(results) == 2
        assert results[0]["id_val"] == "p1"
        assert results[0]["distance"] == pytest.approx(1.0 / (0 + rrf_k))
        assert results[1]["id_val"] == "p2"
        assert results[1]["distance"] == pytest.approx(1.0 / (1 + rrf_k))

    def test_secondary_only(self) -> None:
        secondary = [
            get_row("s1", 0.9),
            get_row("s2", 0.7),
        ]  # s1 rank 0, s2 rank 1
        rrf_k = 60
        results = reciprocal_rank_fusion([], secondary, rrf_k=rrf_k)  # type: ignore
        assert len(results) == 2
        assert results[0]["id_val"] == "s1"
        assert results[0]["distance"] == pytest.approx(1.0 / (0 + rrf_k))
        assert results[1]["id_val"] == "s2"
        assert results[1]["distance"] == pytest.approx(1.0 / (1 + rrf_k))

    def test_mixed_results_default_k(self) -> None:
        primary = [get_row("common", 0.8), get_row("p_only", 0.7)]
        secondary = [get_row("common", 0.9), get_row("s_only", 0.6)]
        rrf_k = 60
        # common_score = (1/(0+k))_prim + (1/(0+k))_sec = 2/k
        # p_only_score = (1/(1+k))_prim = 1/(k+1)
        # s_only_score = (1/(1+k))_sec = 1/(k+1)
        results = reciprocal_rank_fusion(primary, secondary, rrf_k=rrf_k)  # type: ignore
        assert len(results) == 3
        assert results[0]["id_val"] == "common"
        assert results[0]["distance"] == pytest.approx(2.0 / rrf_k)
        # Check the next two elements, their order might vary due to tie in score
        next_ids = {results[1]["id_val"], results[2]["id_val"]}
        next_scores = {results[1]["distance"], results[2]["distance"]}
        assert next_ids == {"p_only", "s_only"}
        for score in next_scores:
            assert score == pytest.approx(1.0 / (1 + rrf_k))

    def test_fetch_top_k_rrf(self) -> None:
        primary = [get_row(f"p{i}", (10 - i) / 10.0) for i in range(5)]
        rrf_k = 1
        results = reciprocal_rank_fusion(primary, [], rrf_k=rrf_k, fetch_top_k=2)  # type: ignore
        assert len(results) == 2
        assert results[0]["id_val"] == "p0"
        assert results[0]["distance"] == pytest.approx(1.0 / (0 + rrf_k))
        assert results[1]["id_val"] == "p1"
        assert results[1]["distance"] == pytest.approx(1.0 / (1 + rrf_k))

    def test_rrf_content_preservation(self) -> None:
        primary = [get_row("doc1", 0.9, content="Primary Content")]
        secondary = [get_row("doc1", 0.8, content="Secondary Content")]
        # RRF processes primary then secondary. If a doc is in both,
        # the content from the secondary list will overwrite primary's.
        results = reciprocal_rank_fusion(primary, secondary, rrf_k=60)  # type: ignore
        assert len(results) == 1
        assert results[0]["id_val"] == "doc1"
        assert results[0]["content_field"] == "Secondary Content"

        # If only in primary
        results_prim_only = reciprocal_rank_fusion(primary, [], rrf_k=60)  # type: ignore
        assert results_prim_only[0]["content_field"] == "Primary Content"

    def test_reordering_from_inputs_rrf(self) -> None:
        """
        Tests that RRF fused ranking can be different from both primary and secondary
        input rankings.
        Primary Order: A, B, C
        Secondary Order: C, B, A
        Fused Order: (A, C) tied, then B
        """
        primary = [
            get_row("docA", 0.9),
            get_row("docB", 0.8),
            get_row("docC", 0.1),
        ]
        secondary = [
            get_row("docC", 0.9),
            get_row("docB", 0.5),
            get_row("docA", 0.2),
        ]
        rrf_k = 1.0  # Using 1.0 for k to simplify rank score calculation
        # docA_score = 1/(0+1) [P] + 1/(2+1) [S] = 1 + 1/3 = 4/3
        # docB_score = 1/(1+1) [P] + 1/(1+1) [S] = 1/2 + 1/2 = 1
        # docC_score = 1/(2+1) [P] + 1/(0+1) [S] = 1/3 + 1 = 4/3
        results = reciprocal_rank_fusion(primary, secondary, rrf_k=rrf_k)  # type: ignore
        assert len(results) == 3
        assert {results[0]["id_val"], results[1]["id_val"]} == {"docA", "docC"}
        assert results[0]["distance"] == pytest.approx(4.0 / 3.0)
        assert results[1]["distance"] == pytest.approx(4.0 / 3.0)
        assert results[2]["id_val"] == "docB"
        assert results[2]["distance"] == pytest.approx(1.0)

    def test_reordering_from_inputs_weighted_sum(self) -> None:
        """
        Tests that the fused ranking can be different from both primary and secondary
        input rankings.
        Primary Order: A (0.9), B (0.7)
        Secondary Order: B (0.8), A (0.2)
        Fusion (0.5/0.5 weights):
        docA_score = (0.9 * 0.5) + (0.2 * 0.5) = 0.45 + 0.10 = 0.55
        docB_score = (0.7 * 0.5) + (0.8 * 0.5) = 0.35 + 0.40 = 0.75
        Expected Fused Order: docB (0.75), docA (0.55)
        This is different from Primary (A,B) and Secondary (B,A) in terms of
        original score, but the fusion logic changes the effective contribution).
        """
        primary = [get_row("docA", 0.9), get_row("docB", 0.7)]
        secondary = [get_row("docB", 0.8), get_row("docA", 0.2)]

        results = weighted_sum_ranking(primary, secondary)  # type: ignore
        assert len(results) == 2
        assert results[0]["id_val"] == "docB"
        assert results[0]["distance"] == pytest.approx(1.0)
        assert results[1]["id_val"] == "docA"
        assert results[1]["distance"] == pytest.approx(0.0)
