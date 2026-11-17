"""Unit tests for Haversine and geodetic metrics."""

import numpy as np
import pytest

from geostride.metrics.haversine import haversine_matrix, haversine_pairwise


def test_haversine_same_point():
    # Distance between point and itself should be 0.0
    dist = haversine_pairwise(40.7128, -74.0060, 40.7128, -74.0060, unit="km")
    assert np.isclose(dist, 0.0, atol=1e-6)


def test_haversine_known_distance():
    # NYC (40.7128, -74.0060) to London (51.5074, -0.1278) ~ 5570 km
    nyc_lat, nyc_lon = 40.7128, -74.0060
    lon_lat, lon_lon = 51.5074, -0.1278

    dist_km = haversine_pairwise(nyc_lat, nyc_lon, lon_lat, lon_lon, unit="km")
    assert 5560 < dist_km < 5585

    dist_miles = haversine_pairwise(nyc_lat, nyc_lon, lon_lat, lon_lon, unit="miles")
    assert 3450 < dist_miles < 3475

    dist_m = haversine_pairwise(nyc_lat, nyc_lon, lon_lat, lon_lon, unit="m")
    assert np.isclose(dist_m, dist_km * 1000.0)


def test_haversine_broadcasting():
    lats1 = np.array([40.7128, 34.0522])
    lons1 = np.array([-74.0060, -118.2437])
    lats2 = np.array([40.7128, 37.7749])
    lons2 = np.array([-74.0060, -122.4194])

    dists = haversine_pairwise(lats1, lons1, lats2, lons2, unit="km")
    assert dists.shape == (2,)
    assert np.isclose(dists[0], 0.0)
    assert dists[1] > 500  # LA to SF is ~559 km


def test_haversine_matrix():
    coords_a = np.array([
        [40.7128, -74.0060],  # NYC
        [34.0522, -118.2437]  # LA
    ])
    coords_b = np.array([
        [51.5074, -0.1278],   # London
        [35.6762, 139.6503],  # Tokyo
        [40.7128, -74.0060]   # NYC
    ])

    mat = haversine_matrix(coords_a, coords_b, unit="km")
    assert mat.shape == (2, 3)

    # NYC to NYC in matrix should be 0.0
    assert np.isclose(mat[0, 2], 0.0)
    # Distance from NYC to London
    assert 5560 < mat[0, 0] < 5585


def test_invalid_unit():
    with pytest.raises(ValueError, match="Unsupported unit"):
        haversine_pairwise(0, 0, 1, 1, unit="lightyear")
