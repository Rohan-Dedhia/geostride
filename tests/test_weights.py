"""Unit tests for spatial weight matrix generators and SpatialWeights container."""

import numpy as np
import pytest

from geostride.stats.weights import (
    SpatialWeights,
    distance_band_weights,
    inverse_distance_weights,
    knn_weights,
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
