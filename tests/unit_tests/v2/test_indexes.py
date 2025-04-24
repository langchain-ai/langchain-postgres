import warnings

import pytest

from langchain_postgres.v2.indexes import (
    DistanceStrategy,
    HNSWIndex,
    HNSWQueryOptions,
    IVFFlatIndex,
    IVFFlatQueryOptions,
)


@pytest.mark.enable_socket
class TestPGIndex:
    def test_distance_strategy(self) -> None:
        assert DistanceStrategy.EUCLIDEAN.operator == "<->"
        assert DistanceStrategy.EUCLIDEAN.search_function == "l2_distance"
        assert DistanceStrategy.EUCLIDEAN.index_function == "vector_l2_ops"

        assert DistanceStrategy.COSINE_DISTANCE.operator == "<=>"
        assert DistanceStrategy.COSINE_DISTANCE.search_function == "cosine_distance"
        assert DistanceStrategy.COSINE_DISTANCE.index_function == "vector_cosine_ops"

        assert DistanceStrategy.INNER_PRODUCT.operator == "<#>"
        assert DistanceStrategy.INNER_PRODUCT.search_function == "inner_product"
        assert DistanceStrategy.INNER_PRODUCT.index_function == "vector_ip_ops"

    def test_hnsw_index(self) -> None:
        index = HNSWIndex(name="test_index", m=32, ef_construction=128)
        assert index.index_type == "hnsw"
        assert index.m == 32
        assert index.ef_construction == 128
        assert index.index_options() == "(m = 32, ef_construction = 128)"

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

    def test_ivfflat_query_options(self) -> None:
        options = IVFFlatQueryOptions(probes=2)
        assert options.to_parameter() == ["ivfflat.probes = 2"]

        with warnings.catch_warnings(record=True) as w:
            options.to_string()
            assert len(w) == 1
            assert "to_string is deprecated, use to_parameter instead." in str(
                w[-1].message
            )
