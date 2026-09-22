import warnings

import pytest

from langchain_postgres.v2.indexes import (
    DistanceStrategy,
    HNSWIndex,
    HNSWQueryOptions,
    IVFFlatIndex,
    IVFFlatQueryOptions,
    VectorType,
)


@pytest.mark.enable_socket
class TestPGIndex:
    def test_distance_strategy(self) -> None:
        assert DistanceStrategy.EUCLIDEAN.operator == "<->"
        assert DistanceStrategy.EUCLIDEAN.search_function == "l2_distance"
        assert DistanceStrategy.EUCLIDEAN.operator_class_suffix == "l2_ops"

        assert DistanceStrategy.COSINE_DISTANCE.operator == "<=>"
        assert DistanceStrategy.COSINE_DISTANCE.search_function == "cosine_distance"
        assert DistanceStrategy.COSINE_DISTANCE.operator_class_suffix == "cosine_ops"

        assert DistanceStrategy.INNER_PRODUCT.operator == "<#>"
        assert DistanceStrategy.INNER_PRODUCT.search_function == "inner_product"
        assert DistanceStrategy.INNER_PRODUCT.operator_class_suffix == "ip_ops"

    @pytest.mark.parametrize(
        "vector_type, distance_strategy, expected",
        [
            (
                VectorType.VECTOR,
                DistanceStrategy.COSINE_DISTANCE,
                "vector_cosine_ops",
            ),
            (VectorType.VECTOR, DistanceStrategy.EUCLIDEAN, "vector_l2_ops"),
            (VectorType.VECTOR, DistanceStrategy.INNER_PRODUCT, "vector_ip_ops"),
            (
                VectorType.HALFVEC,
                DistanceStrategy.COSINE_DISTANCE,
                "halfvec_cosine_ops",
            ),
            (VectorType.HALFVEC, DistanceStrategy.EUCLIDEAN, "halfvec_l2_ops"),
            (VectorType.HALFVEC, DistanceStrategy.INNER_PRODUCT, "halfvec_ip_ops"),
            (
                VectorType.SPARSEVEC,
                DistanceStrategy.COSINE_DISTANCE,
                "sparsevec_cosine_ops",
            ),
            (VectorType.SPARSEVEC, DistanceStrategy.EUCLIDEAN, "sparsevec_l2_ops"),
            (VectorType.SPARSEVEC, DistanceStrategy.INNER_PRODUCT, "sparsevec_ip_ops"),
        ],
    )
    def test_operator_class_by_vector_type(
        self,
        vector_type: VectorType,
        distance_strategy: DistanceStrategy,
        expected: str,
    ) -> None:
        idx = HNSWIndex(vector_type=vector_type, distance_strategy=distance_strategy)
        assert idx.operator_class() == expected

    def test_hnsw_index(self) -> None:
        index = HNSWIndex(name="test_index", m=32, ef_construction=128)
        assert index.index_type == "hnsw"
        assert index.m == 32
        assert index.ef_construction == 128
        assert index.index_options() == "(m = 32, ef_construction = 128)"
        assert index.operator_class() == "vector_cosine_ops"

    def test_hnsw_query_options(self) -> None:
        options = HNSWQueryOptions(ef_search=80)
        assert options.to_parameter() == ["hnsw.ef_search = 80"]

        with warnings.catch_warnings(record=True) as w:
            options.to_string()

            assert len(w) == 1
            assert "to_string is deprecated, use to_parameter instead." in str(
                w[-1].message
            )

    def test_ivfflat_index(self) -> None:
        index = IVFFlatIndex(name="test_index", lists=200)
        assert index.index_type == "ivfflat"
        assert index.lists == 200
        assert index.index_options() == "(lists = 200)"
        assert index.operator_class() == "vector_cosine_ops"

    def test_ivfflat_query_options(self) -> None:
        options = IVFFlatQueryOptions(probes=2)
        assert options.to_parameter() == ["ivfflat.probes = 2"]

        with warnings.catch_warnings(record=True) as w:
            options.to_string()
            assert len(w) == 1
            assert "to_string is deprecated, use to_parameter instead." in str(
                w[-1].message
            )
