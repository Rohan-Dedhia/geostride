"""Unit tests for Haversine, Vincenty, and Manhattan geodetic metrics."""

import numpy as np
import pytest

from geostride.metrics.haversine import haversine_matrix, haversine_pairwise
from geostride.metrics.manhattan import spatial_manhattan_distance, spatial_manhattan_matrix
from geostride.metrics.vincenty import vincenty_distance, vincenty_pairwise


def test_haversine_same_point():
    dist = haversine_pairwise(40.7128, -74.0060, 40.7128, -74.0060, unit="km")
    assert np.isclose(dist, 0.0, atol=1e-6)


def test_haversine_known_distance():
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
    assert dists[1] > 500


def test_haversine_matrix():
    coords_a = np.array([
        [40.7128, -74.0060],  # NYC
        [34.0522, -118.2437],  # LA
    ])
    coords_b = np.array([
        [51.5074, -0.1278],   # London
        [35.6762, 139.6503],  # Tokyo
        [40.7128, -74.0060],   # NYC
    ])

    mat = haversine_matrix(coords_a, coords_b, unit="km")
    assert mat.shape == (2, 3)
    assert np.isclose(mat[0, 2], 0.0)
    assert 5560 < mat[0, 0] < 5585


def test_invalid_unit():
    with pytest.raises(ValueError, match="Unsupported unit"):
        haversine_pairwise(0, 0, 1, 1, unit="lightyear")


def test_vincenty_distance():
    # Coincident point
    assert vincenty_distance(40.7128, -74.0060, 40.7128, -74.0060, unit="km") == 0.0

    # NYC to London on WGS-84 ellipsoid
    v_km = vincenty_distance(40.7128, -74.0060, 51.5074, -0.1278, unit="km")
    h_km = haversine_pairwise(40.7128, -74.0060, 51.5074, -0.1278, unit="km")

    # Vincenty and Haversine should agree within ~0.5% (ellipsoid vs spherical difference)
    assert 5560 < v_km < 5595
    assert abs(v_km - h_km) / v_km < 0.005

    # Test unit conversion
    v_m = vincenty_distance(40.7128, -74.0060, 51.5074, -0.1278, unit="m")
    assert np.isclose(v_m, v_km * 1000.0)


def test_vincenty_pairwise():
    coords_a = [[40.7128, -74.0060], [34.0522, -118.2437]]
    coords_b = [[51.5074, -0.1278], [37.7749, -122.4194]]

    res = vincenty_pairwise(coords_a, coords_b, unit="km")
    assert len(res) == 2
    assert res[0] > 5000
    assert res[1] > 500

    # Mismatched length raises error
    with pytest.raises(ValueError, match="matching row counts"):
        vincenty_pairwise(coords_a, coords_b[:1])


def test_spatial_manhattan_distance():
    # Manhattan distance should be >= Haversine distance
    nyc_lat, nyc_lon = 40.7128, -74.0060
    point_lat, point_lon = 40.7589, -73.9851

    d_manhattan = spatial_manhattan_distance(nyc_lat, nyc_lon, point_lat, point_lon, unit="km")
    d_haversine = haversine_pairwise(nyc_lat, nyc_lon, point_lat, point_lon, unit="km")

    assert d_manhattan >= d_haversine
    assert d_manhattan > 0.0


def test_spatial_manhattan_matrix():
    coords_a = [[40.7128, -74.0060], [40.7589, -73.9851]]
    coords_b = [[40.7128, -74.0060], [40.7829, -73.9654]]

    mat = spatial_manhattan_matrix(coords_a, coords_b, unit="km")
    assert mat.shape == (2, 2)
    assert np.isclose(mat[0, 0], 0.0)
    assert mat[0, 1] > 0.0
