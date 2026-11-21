"""Unit tests for spatial weight matrix generators and SpatialWeights container."""

import numpy as np
import pytest

from geostride.stats.weights import (
    SpatialWeights,
    distance_band_weights,
    inverse_distance_weights,
    knn_weights,
    standardize_weights,
)


def test_spatial_weights_container_basic():
    # 3x3 binary matrix with zero diagonal
    mat = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    w = SpatialWeights(matrix=mat, transform="B")
    assert w.n_observations == 3
    assert w.s0 == 4.0
    assert np.array_equal(w.cardinalities, [1, 2, 1])
    assert not w.has_islands
    assert len(w.islands) == 0

    # Row standardize
    w_r = w.row_standardize()
    assert w_r.transform == "R"
    assert np.allclose(np.sum(w_r.matrix, axis=1), [1.0, 1.0, 1.0])


def test_spatial_weights_island_detection():
    # Node 2 has 0 neighbors (island)
    mat = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    w = SpatialWeights(matrix=mat, transform="B")
    assert w.has_islands
    assert w.islands == [2]

    # Row standardization must not raise ZeroDivisionError
    w_r = w.row_standardize()
    assert np.isclose(np.sum(w_r.matrix[0]), 1.0)
    assert np.isclose(np.sum(w_r.matrix[2]), 0.0)


def test_knn_weights():
    # 4 collinear points along equator
    coords = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, 3.0],
    ])

    w_knn = knn_weights(coords, k=2, row_standardize=True)
    assert w_knn.n_observations == 4
    # Zero diagonal
    assert np.all(np.diagonal(w_knn.matrix) == 0.0)
    # Row standardized
    assert np.allclose(np.sum(w_knn.matrix, axis=1), 1.0)

    # Binary KNN
    w_bin = knn_weights(coords, k=1, row_standardize=False)
    assert np.all(w_bin.cardinalities == 1)


def test_distance_band_weights():
    # NYC Manhattan nearby vs distant points
    coords = np.array([
        [40.71, -74.00],
        [40.72, -73.99],  # ~1.5 km away
        [41.50, -74.00],  # ~88 km away
    ])

    # Threshold 10 km: point 0 & 1 are neighbors, point 2 is isolated
    w_band = distance_band_weights(coords, threshold_km=10.0, row_standardize=False)
    assert w_band.matrix[0, 1] == 1.0
    assert w_band.matrix[1, 0] == 1.0
    assert w_band.matrix[0, 2] == 0.0
    assert w_band.islands == [2]


def test_inverse_distance_weights():
    coords = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 5.0],
    ])

    w_idw = inverse_distance_weights(coords, alpha=1.0, row_standardize=False)
    assert np.all(np.diagonal(w_idw.matrix) == 0.0)
    # Closer neighbor (0 to 1) has higher raw weight than (0 to 2)
    assert w_idw.matrix[0, 1] > w_idw.matrix[0, 2]


def test_weights_validation_errors():
    # Non-square matrix
    with pytest.raises(ValueError, match="must be square"):
        SpatialWeights(matrix=np.ones((2, 3)))

    # Non-zero diagonal
    with pytest.raises(ValueError, match="zero diagonal"):
        SpatialWeights(matrix=np.eye(3))

    # k >= N
    with pytest.raises(ValueError, match="require N > k"):
        knn_weights(np.zeros((3, 2)), k=3)

    # Invalid threshold
    with pytest.raises(ValueError, match="strictly positive"):
        distance_band_weights(np.zeros((3, 2)), threshold_km=-5.0)


class TestWeightsStandardization:
    """Tests for spatial weights matrix standardization transformations."""

    @pytest.fixture
    def sample_weights(self) -> SpatialWeights:
        # Connected 4-node network
        mat = np.array([
            [0.0, 2.0, 4.0, 0.0],
            [2.0, 0.0, 2.0, 2.0],
            [4.0, 2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ])
        return SpatialWeights(matrix=mat, transform="B")

    def test_row_standardization(self, sample_weights: SpatialWeights) -> None:
        w_row = sample_weights.standardize("row")
        assert w_row.transform == "R"
        assert np.allclose(np.sum(w_row.matrix, axis=1), 1.0)

        # Short style code "r"
        w_r = sample_weights.set_transform("r")
        assert w_r.transform == "R"
        assert np.allclose(w_r.matrix, w_row.matrix)

    def test_binary_standardization(self, sample_weights: SpatialWeights) -> None:
        w_bin = sample_weights.standardize("binary")
        assert w_bin.transform == "B"
        # Non-zero entries should all be 1.0
        assert np.all((w_bin.matrix == 0.0) | (w_bin.matrix == 1.0))
        assert w_bin.matrix[0, 1] == 1.0
        assert w_bin.matrix[0, 2] == 1.0
        assert w_bin.matrix[0, 3] == 0.0

    def test_spectral_standardization(self, sample_weights: SpatialWeights) -> None:
        w_spec = sample_weights.standardize("spectral")
        assert w_spec.transform == "V"
        # Spectral radius should now equal 1.0
        eigs = np.linalg.eigvals(w_spec.matrix)
        assert np.isclose(np.max(np.real(eigs)), 1.0)

    def test_minmax_standardization(self, sample_weights: SpatialWeights) -> None:
        w_mm = sample_weights.standardize("minmax")
        assert w_mm.transform == "M"
        r_max = np.max(np.sum(w_mm.matrix, axis=1))
        c_max = np.max(np.sum(w_mm.matrix, axis=0))
        # Min of max row or col sums should be 1.0
        assert np.isclose(min(r_max, c_max), 1.0)

    def test_island_preservation(self) -> None:
        # Matrix with isolate node 2
        mat = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        w = SpatialWeights(matrix=mat)
        w_row = w.standardize("row")
        assert np.isclose(np.sum(w_row.matrix[0]), 1.0)
        assert np.isclose(np.sum(w_row.matrix[1]), 1.0)
        assert np.isclose(np.sum(w_row.matrix[2]), 0.0)

    def test_standardize_weights_function(self, sample_weights: SpatialWeights) -> None:
        # With SpatialWeights container
        res_container = standardize_weights(sample_weights, style="spectral")
        assert isinstance(res_container, SpatialWeights)
        assert res_container.transform == "V"

        # With raw NumPy matrix
        res_np = standardize_weights(sample_weights.matrix, style="row")
        assert isinstance(res_np, np.ndarray)
        assert np.allclose(np.sum(res_np, axis=1), 1.0)

    def test_errors(self, sample_weights: SpatialWeights) -> None:
        with pytest.raises(ValueError, match="Unsupported standardization style"):
            sample_weights.standardize("unknown_style")

        with pytest.raises(TypeError, match="weights must be a SpatialWeights container or a NumPy ndarray"):
            standardize_weights("not_a_matrix")  # type: ignore[arg-type]

