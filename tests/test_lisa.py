"""Unit tests for Local Indicators of Spatial Association (LISA) decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from geostride import LISAResult, local_moran
from geostride.stats.weights import knn_weights


def test_lisa_hot_and_cold_spots() -> None:
    """Test clear spatial clustering into High-High hot spots and Low-Low cold spots."""
    rng = np.random.default_rng(42)
    # Cluster 1: 15 points near (0, 0) with high values ~100
    cluster1 = rng.normal(loc=0.0, scale=0.1, size=(15, 2))
    vals1 = rng.normal(loc=100.0, scale=2.0, size=15)

    # Cluster 2: 15 points far away near (10, 10) with low values ~10
    cluster2 = rng.normal(loc=10.0, scale=0.1, size=(15, 2))
    vals2 = rng.normal(loc=10.0, scale=2.0, size=15)

    coords = np.vstack([cluster1, cluster2])
    vals = np.concatenate([vals1, vals2])

    res = local_moran(coords, vals, k_neighbors=4, n_permutations=499, seed=42)

    assert isinstance(res, LISAResult)
    assert len(res.local_moran) == 30
    assert len(res.p_values) == 30
    assert len(res.quadrants) == 30

    # Cluster 1 points should be High-High (quadrant 1)
    # Cluster 2 points should be Low-Low (quadrant 3)
    np.testing.assert_array_equal(res.raw_quadrants[:15], 1)
    np.testing.assert_array_equal(res.raw_quadrants[15:], 3)

    # Significant clusters should be identified
    assert len(res.high_high) > 0
    assert len(res.low_low) > 0
    assert set(res.high_high).issubset(set(range(15)))
    assert set(res.low_low).issubset(set(range(15, 30)))


def test_lisa_spatial_outliers() -> None:
    """Test identification of spatial outliers (High-Low and Low-High)."""
    # Grid of points with one High-Low outlier and one Low-High outlier
    coords = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0],
        [5.0, 6.0],
        [6.0, 5.0],
        [6.0, 6.0],
    ])
    # Group 1 mostly low (10.0), with point 0 as an anomalous spike (100.0) -> High-Low
    # Group 2 mostly high (100.0), with point 7 as an anomalous dip (10.0) -> Low-High
    vals = np.array([100.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 10.0])

    res = local_moran(coords, vals, k_neighbors=3, n_permutations=299, seed=123)

    # Point 0 is High surrounded by Low -> High-Low (quadrant 4)
    assert res.raw_quadrants[0] == 4
    assert res.quadrant_labels[0] in ("High-Low", "Not Significant")

    # Point 7 is Low surrounded by High -> Low-High (quadrant 2)
    assert res.raw_quadrants[7] == 2


def test_lisa_custom_weights() -> None:
    """Test using pre-constructed SpatialWeights container."""
    coords = np.array([
        [40.71, -74.00],
        [40.72, -73.99],
        [40.73, -73.98],
        [40.74, -73.97],
        [40.75, -73.96],
    ])
    vals = np.array([50.0, 55.0, 48.0, 52.0, 51.0])

    w = knn_weights(coords, k=2)
    res = local_moran(coords, vals, weights=w, n_permutations=199, seed=99)

    assert len(res.local_moran) == 5
    assert res.p_threshold == 0.05
    assert "LISA Cluster Analysis" in str(res)


def test_lisa_error_handling() -> None:
    """Test validation errors for invalid inputs."""
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    vals = np.array([5.0, 5.0, 5.0, 5.0])

    # Zero variance values
    with pytest.raises(ValueError, match="zero variance"):
        local_moran(coords, vals)

    # Shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        local_moran(coords, [1.0, 2.0])

    # Too few observations (< 4)
    with pytest.raises(ValueError, match="at least 4 observations"):
        local_moran([[0, 0], [0, 1], [1, 0]], [1.0, 2.0, 3.0])

    # Invalid p_threshold
    with pytest.raises(ValueError, match="p_threshold must be in"):
        local_moran(coords, [1.0, 2.0, 3.0, 4.0], p_threshold=1.5)

    # Invalid n_permutations
    with pytest.raises(ValueError, match="n_permutations must be at least"):
        local_moran(coords, [1.0, 2.0, 3.0, 4.0], n_permutations=3)

    # Weights dimension mismatch
    w_bad = knn_weights(np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]]), k=2)
    with pytest.raises(ValueError, match="Weights matrix dimension"):
        local_moran(coords, [1.0, 2.0, 3.0, 4.0], weights=w_bad)
