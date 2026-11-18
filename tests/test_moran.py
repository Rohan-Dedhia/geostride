"""Unit tests for Global Moran's I spatial autocorrelation and permutation inference."""

import numpy as np
import pytest

from geostride.stats.moran import MoranResult, global_moran
from geostride.stats.weights import knn_weights


def test_moran_positive_spatial_clustering():
    # 6x6 grid where northern half has high values and southern half has low values
    lats = np.linspace(40.0, 41.0, 6)
    lons = np.linspace(-74.0, -73.0, 6)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    coords = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])

    # Values cluster geographically by latitude
    values = np.where(coords[:, 0] >= 40.5, 100.0, 10.0)

    res = global_moran(coords, values, k_neighbors=4, permutations=199, random_state=42)

    assert isinstance(res, MoranResult)
    # Positive spatial clustering produces high Moran's I
    assert res.moran_i > 0.4
    assert res.p_value_analytical < 0.01
    assert res.p_value_sim < 0.05
    assert res.z_score > 2.0
    assert "Global Moran's I Results" in str(res)


def test_moran_negative_spatial_dispersion():
    # Regular 4x4 grid with alternating checkerboard pattern
    coords = []
    values = []
    for i in range(4):
        for j in range(4):
            coords.append([float(i), float(j)])
            # Checkerboard: (i + j) even -> 10.0, odd -> 0.0
            values.append(10.0 if (i + j) % 2 == 0 else 0.0)

    coords_arr = np.array(coords)
    values_arr = np.array(values)

    res = global_moran(coords_arr, values_arr, k_neighbors=4, permutations=199, random_state=42)
    # Checkerboard produces strong negative spatial autocorrelation
    assert res.moran_i < -0.2


def test_moran_random_spatial_noise():
    rng = np.random.default_rng(123)
    n = 50
    coords = rng.uniform(0.0, 10.0, size=(n, 2))
    values = rng.normal(50.0, 5.0, size=n)

    res = global_moran(coords, values, k_neighbors=5, permutations=199, random_state=42)
    # Expected value is -1 / (N - 1) ~ -0.02
    assert abs(res.moran_i - res.expected_i) < 0.25
    # Random noise should not be statistically significant
    assert res.p_value_sim > 0.05


def test_moran_with_custom_weights():
    coords = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, 3.0],
        [0.0, 4.0],
    ])
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    w = knn_weights(coords, k=2, row_standardize=True)
    res = global_moran(coords, values, weights=w, permutations=99)
    assert res.moran_i > 0.0


def test_moran_validation_errors():
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])

    # Zero variance (constant values)
    with pytest.raises(ValueError, match="zero variance"):
        global_moran(coords, [5.0, 5.0, 5.0, 5.0])

    # N < 4
    with pytest.raises(ValueError, match="at least 4"):
        global_moran(coords[:3], [1.0, 2.0, 3.0])

    # Length mismatch
    with pytest.raises(ValueError, match="Mismatched lengths"):
        global_moran(coords, [1.0, 2.0])
