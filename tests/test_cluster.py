"""Unit tests for spatial DBSCAN clustering."""

import numpy as np
import pytest

from geostride.cluster.dbscan import SpatialDBSCANResult, spatial_dbscan


def test_spatial_dbscan_two_distinct_clusters():
    # NYC points (cluster A)
    nyc_coords = np.array([
        [40.7128, -74.0060],
        [40.7150, -74.0080],
        [40.7100, -74.0040],
        [40.7180, -74.0020],
    ])

    # Philadelphia points (cluster B) - approx 130 km away from NYC
    philly_coords = np.array([
        [39.9526, -75.1652],
        [39.9550, -75.1680],
        [39.9500, -75.1620],
        [39.9580, -75.1640],
    ])

    all_coords = np.vstack([nyc_coords, philly_coords])

    # eps = 5.0 km, min_samples = 3
    res = spatial_dbscan(all_coords, eps_km=5.0, min_samples=3)

    assert isinstance(res, SpatialDBSCANResult)
    assert res.n_clusters == 2
    assert res.noise_ratio == 0.0

    # NYC points should all share the same cluster id
    nyc_labels = res.labels[:4]
    assert len(set(nyc_labels)) == 1
    assert nyc_labels[0] >= 0

    # Philly points should all share the other cluster id
    philly_labels = res.labels[4:]
    assert len(set(philly_labels)) == 1
    assert philly_labels[0] >= 0
    assert nyc_labels[0] != philly_labels[0]

    # Verify cluster centroids
    assert len(res.cluster_centroids) == 2
    for centroid in res.cluster_centroids.values():
        assert (-90.0 <= centroid[0] <= 90.0)
        assert (-180.0 <= centroid[1] <= 180.0)


def test_spatial_dbscan_noise_detection():
    # Dense cluster of 4 points in NYC
    cluster_coords = np.array([
        [40.7128, -74.0060],
        [40.7130, -74.0062],
        [40.7125, -74.0058],
        [40.7132, -74.0065],
    ])

    # Isolated noise point 50 km away
    noise_pt = np.array([[40.3000, -74.0060]])

    all_coords = np.vstack([cluster_coords, noise_pt])
    res = spatial_dbscan(all_coords, eps_km=1.0, min_samples=3)

    assert res.n_clusters == 1
    assert res.labels[-1] == -1  # Last point is noise
    assert res.noise_ratio == 0.2  # 1 out of 5 is noise
    assert "Noise Ratio" in str(res)


def test_spatial_dbscan_all_noise():
    # 3 points thousands of km apart
    coords = np.array([
        [40.7128, -74.0060],  # NYC
        [51.5074, -0.1278],   # London
        [35.6762, 139.6503],  # Tokyo
    ])
    res = spatial_dbscan(coords, eps_km=50.0, min_samples=2)
    assert res.n_clusters == 0
    assert np.all(res.labels == -1)
    assert res.noise_ratio == 1.0


def test_spatial_dbscan_empty_input():
    empty_coords = np.empty((0, 2))
    res = spatial_dbscan(empty_coords, eps_km=1.0, min_samples=3)
    assert res.n_clusters == 0
    assert len(res.labels) == 0


def test_spatial_dbscan_validation_errors():
    coords = np.array([[40.0, -74.0], [40.1, -74.1]])

    # Invalid eps
    with pytest.raises(ValueError, match="strictly positive"):
        spatial_dbscan(coords, eps_km=-1.0)

    # Invalid min_samples
    with pytest.raises(ValueError, match="at least 1"):
        spatial_dbscan(coords, min_samples=0)
