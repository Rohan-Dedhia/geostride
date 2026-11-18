"""Unit tests for Getis-Ord Gi and Gi* local spatial hot-spot statistics."""

import numpy as np
import pytest

from geostride.stats.getis_ord import GetisOrdResult, getis_ord_local
from geostride.stats.weights import knn_weights


def test_getis_ord_hot_spot_detection():
    # 6x6 spatial grid
    lats = np.linspace(40.0, 41.0, 6)
    lons = np.linspace(-74.0, -73.0, 6)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    coords = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])

    # Low background values (10.0), with top-left corner as a high-value hot spot (100.0)
    values = np.full(36, 10.0)
    # Mark first 4 points as a concentrated hot spot
    values[:4] = 100.0

    res = getis_ord_local(coords, values, star=True, k_neighbors=4)

    assert isinstance(res, GetisOrdResult)
    assert len(res.z_scores) == 36
    assert len(res.p_values) == 36
    assert len(res.labels) == 36

    # The hot spot points should have significant positive z-scores
    assert np.all(res.z_scores[:4] > 2.0)
    assert len(res.hot_spots_95) >= 2
    summary = res.summary()
    assert "Hot Spot" in str(summary)


def test_getis_ord_cold_spot_detection():
    coords = np.column_stack([np.arange(10, dtype=float), np.zeros(10)])
    # High background (100.0), with first two points as very low cold spot (0.0)
    values = np.full(10, 100.0)
    values[:2] = 0.0

    res = getis_ord_local(coords, values, star=True, k_neighbors=2)
    # The low value points should produce negative z-scores
    assert res.z_scores[0] < -1.5
    assert res.z_scores[1] < -1.5


def test_getis_ord_gi_vs_gi_star():
    coords = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, 3.0],
        [0.0, 4.0],
    ])
    values = np.array([10.0, 50.0, 10.0, 10.0, 10.0])

    res_star = getis_ord_local(coords, values, star=True, k_neighbors=2)
    res_gi = getis_ord_local(coords, values, star=False, k_neighbors=2)

    assert res_star.star is True
    assert res_gi.star is False
    # Self inclusion affects the exact z-score at point 1 (val 50)
    assert res_star.z_scores[1] != res_gi.z_scores[1]


def test_getis_ord_with_custom_weights():
    coords = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, 3.0],
    ])
    values = np.array([10.0, 20.0, 30.0, 40.0])
    w = knn_weights(coords, k=2, row_standardize=False)

    res = getis_ord_local(coords, values, weights=w)
    assert len(res.z_scores) == 4


def test_getis_ord_validation_errors():
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])

    # Zero variance (constant values)
    with pytest.raises(ValueError, match="zero variance"):
        getis_ord_local(coords, [10.0, 10.0, 10.0])

    # Sample size N < 3
    with pytest.raises(ValueError, match="at least 3"):
        getis_ord_local(coords[:2], [1.0, 2.0])

    # Length mismatch
    with pytest.raises(ValueError, match="Mismatched lengths"):
        getis_ord_local(coords, [1.0, 2.0])
