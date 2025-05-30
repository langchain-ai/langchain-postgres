import pytest

from langchain_postgres.v2.hybrid_search_config import (
    reciprocal_rank_fusion,
    weighted_sum_ranking,
)


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
    def test_empty_inputs(self):
        results = weighted_sum_ranking([], [])
        assert results == []

    def test_primary_only(self):
        primary = [get_row("p1", 0.8), get_row("p2", 0.6)]
        # Expected scores: p1 = 0.8 * 0.5 = 0.4, p2 = 0.6 * 0.5 = 0.3
        results = weighted_sum_ranking(
            primary, [], primary_results_weight=0.5, secondary_results_weight=0.5
        )
        assert len(results) == 2
        assert results[0]["id_val"] == "p1"
        assert results[0]["distance"] == pytest.approx(0.4)
        assert results[1]["id_val"] == "p2"
        assert results[1]["distance"] == pytest.approx(0.3)

    def test_secondary_only(self):
        secondary = [get_row("s1", 0.9), get_row("s2", 0.7)]
        # Expected scores: s1 = 0.9 * 0.5 = 0.45, s2 = 0.7 * 0.5 = 0.35
        results = weighted_sum_ranking(
            [], secondary, primary_results_weight=0.5, secondary_results_weight=0.5
        )
        assert len(results) == 2
        assert results[0]["id_val"] == "s1"
        assert results[0]["distance"] == pytest.approx(0.45)
        assert results[1]["id_val"] == "s2"
        assert results[1]["distance"] == pytest.approx(0.35)

    def test_mixed_results_default_weights(self):
        primary = [get_row("common", 0.8), get_row("p_only", 0.7)]
        secondary = [get_row("common", 0.9), get_row("s_only", 0.6)]
        # Weights are 0.5, 0.5
        # common_score = (0.8 * 0.5) + (0.9 * 0.5) = 0.4 + 0.45 = 0.85
        # p_only_score = (0.7 * 0.5) = 0.35
        # s_only_score = (0.6 * 0.5) = 0.30
        # Order: common (0.85), p_only (0.35), s_only (0.30)

        results = weighted_sum_ranking(primary, secondary)
        assert len(results) == 3
        assert results[0]["id_val"] == "common"
        assert results[0]["distance"] == pytest.approx(0.85)
        assert results[1]["id_val"] == "p_only"
        assert results[1]["distance"] == pytest.approx(0.35)
        assert results[2]["id_val"] == "s_only"
        assert results[2]["distance"] == pytest.approx(0.30)

    def test_mixed_results_custom_weights(self):
        primary = [get_row("d1", 1.0)]  # p_w=0.2 -> 0.2
        secondary = [get_row("d1", 0.5)]  # s_w=0.8 -> 0.4
        # Expected: d1_score = (1.0 * 0.2) + (0.5 * 0.8) = 0.2 + 0.4 = 0.6

        results = weighted_sum_ranking(
            primary, secondary, primary_results_weight=0.2, secondary_results_weight=0.8
        )
        assert len(results) == 1
        assert results[0]["id_val"] == "d1"
        assert results[0]["distance"] == pytest.approx(0.6)

    def test_fetch_top_k(self):
        primary = [get_row(f"p{i}", (10 - i) / 10.0) for i in range(5)]
        # Scores: 1.0, 0.9, 0.8, 0.7, 0.6
        # Weighted (0.5): 0.5, 0.45, 0.4, 0.35, 0.3
        secondary = []
        results = weighted_sum_ranking(primary, secondary, fetch_top_k=2)
        assert len(results) == 2
        assert results[0]["id_val"] == "p0"
        assert results[0]["distance"] == pytest.approx(0.5)
        assert results[1]["id_val"] == "p1"
        assert results[1]["distance"] == pytest.approx(0.45)


class TestReciprocalRankFusion:
    def test_empty_inputs(self):
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_primary_only(self):
        primary = [
            get_row("p1", 0.8),
            get_row("p2", 0.6),
        ]  # p1 rank 0, p2 rank 1
        rrf_k = 60
        # p1_score = 1 / (0 + 60)
        # p2_score = 1 / (1 + 60)
        results = reciprocal_rank_fusion(primary, [], rrf_k=rrf_k)
        assert len(results) == 2
        assert results[0]["id_val"] == "p1"
        assert results[0]["distance"] == pytest.approx(1.0 / (0 + rrf_k))
        assert results[1]["id_val"] == "p2"
        assert results[1]["distance"] == pytest.approx(1.0 / (1 + rrf_k))

    def test_secondary_only(self):
        secondary = [
            get_row("s1", 0.9),
            get_row("s2", 0.7),
        ]  # s1 rank 0, s2 rank 1
        rrf_k = 60
        results = reciprocal_rank_fusion([], secondary, rrf_k=rrf_k)
        assert len(results) == 2
        assert results[0]["id_val"] == "s1"
        assert results[0]["distance"] == pytest.approx(1.0 / (0 + rrf_k))
        assert results[1]["id_val"] == "s2"
        assert results[1]["distance"] == pytest.approx(1.0 / (1 + rrf_k))

    def test_mixed_results_default_k(self):
        primary = [get_row("common", 0.8), get_row("p_only", 0.7)]
        secondary = [get_row("common", 0.9), get_row("s_only", 0.6)]
        rrf_k = 60
        # common_score = (1/(0+k))_prim + (1/(0+k))_sec = 2/k
        # p_only_score = (1/(1+k))_prim = 1/(k+1)
        # s_only_score = (1/(1+k))_sec = 1/(k+1)
        results = reciprocal_rank_fusion(primary, secondary, rrf_k=rrf_k)
        assert len(results) == 3
        assert results[0]["id_val"] == "common"
        assert results[0]["distance"] == pytest.approx(2.0 / rrf_k)
        # Check the next two elements, their order might vary due to tie in score
        next_ids = {results[1]["id_val"], results[2]["id_val"]}
        next_scores = {results[1]["distance"], results[2]["distance"]}
        assert next_ids == {"p_only", "s_only"}
        for score in next_scores:
            assert score == pytest.approx(1.0 / (1 + rrf_k))

    def test_fetch_top_k_rrf(self):
        primary = [get_row(f"p{i}", (10 - i) / 10.0) for i in range(5)]
        secondary = []
        rrf_k = 1
        results = reciprocal_rank_fusion(primary, secondary, rrf_k=rrf_k, fetch_top_k=2)
        assert len(results) == 2
        assert results[0]["id_val"] == "p0"
        assert results[0]["distance"] == pytest.approx(1.0 / (0 + rrf_k))
        assert results[1]["id_val"] == "p1"
        assert results[1]["distance"] == pytest.approx(1.0 / (1 + rrf_k))

    def test_rrf_content_preservation(self):
        primary = [get_row("doc1", 0.9, content="Primary Content")]
        secondary = [get_row("doc1", 0.8, content="Secondary Content")]
        # RRF processes primary then secondary. If a doc is in both,
        # the content from the secondary list will overwrite primary's.
        results = reciprocal_rank_fusion(primary, secondary, rrf_k=60)
        assert len(results) == 1
        assert results[0]["id_val"] == "doc1"
        assert results[0]["content_field"] == "Secondary Content"

        # If only in primary
        results_prim_only = reciprocal_rank_fusion(primary, [], rrf_k=60)
        assert results_prim_only[0]["content_field"] == "Primary Content"

    def test_reordering_from_inputs_rrf(self):
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
        results = reciprocal_rank_fusion(primary, secondary, rrf_k=rrf_k)
        assert len(results) == 3
        assert {results[0]["id_val"], results[1]["id_val"]} == {"docA", "docC"}
        assert results[0]["distance"] == pytest.approx(4.0 / 3.0)
        assert results[1]["distance"] == pytest.approx(4.0 / 3.0)
        assert results[2]["id_val"] == "docB"
        assert results[2]["distance"] == pytest.approx(1.0)

    def test_reordering_from_inputs_weighted_sum(self):
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

        results = weighted_sum_ranking(primary, secondary)
        assert len(results) == 2
        assert results[0]["id_val"] == "docB"
        assert results[0]["distance"] == pytest.approx(0.75)
        assert results[1]["id_val"] == "docA"
        assert results[1]["distance"] == pytest.approx(0.55)
