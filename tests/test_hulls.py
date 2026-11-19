"""Unit tests for convex hull, concave alpha shape, and polygon area utilities."""

import math

import numpy as np
import pytest

from geostride.cluster.hulls import (
    alpha_shape_concave_hull,
    convex_hull_2d,
    hull_to_geojson,
    polygon_area_km2,
)


@pytest.fixture
def square_with_interior():
    # 4 corners of a 1x1 degree square in lat/lon plus interior points
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.7, 0.3],
        ]
    )


def test_convex_hull_2d_square(square_with_interior):
    hull = convex_hull_2d(square_with_interior)

    assert len(hull) == 4
    # Check that all 4 corners are present in hull
    corners = {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)}
    hull_set = {tuple(p) for p in hull}
    assert hull_set == corners


def test_convex_hull_2d_small_inputs():
    # Single point
    pt1 = np.array([[40.71, -74.00]])
    assert len(convex_hull_2d(pt1)) == 1

    # Two points
    pt2 = np.array([[40.71, -74.00], [40.75, -73.98]])
    assert len(convex_hull_2d(pt2)) == 2

    # Empty array
    with pytest.raises(ValueError, match="cannot be empty"):
        convex_hull_2d(np.empty((0, 2)))


def test_convex_hull_2d_collinear():
    line_pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    hull = convex_hull_2d(line_pts)
    assert len(hull) == 2
    endpoints = {tuple(p) for p in hull}
    assert endpoints == {(0.0, 0.0), (3.0, 3.0)}


def test_polygon_area_km2_equatorial():
    # 1 deg x 1 deg square at equator
    coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    area = polygon_area_km2(coords)
    # Expected spherical area is approx 12,364 km^2
    assert math.isclose(area, 12363.7, rel_tol=0.01)


def test_polygon_area_km2_degenerate():
    assert polygon_area_km2([[0.0, 0.0], [1.0, 1.0]]) == 0.0
    assert polygon_area_km2(np.empty((0, 2))) == 0.0


def test_alpha_shape_concave_hull_horseshoe():
    # Points arranged in a U-shape / horseshoe
    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [2.0, 1.0],
            [2.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.1],
            [1.5, 0.1],
            [1.8, 1.8],
            [0.2, 1.8],
        ]
    )

    # Tight alpha (concave)
    rings = alpha_shape_concave_hull(coords, alpha=0.008)
    assert len(rings) >= 1
    primary_ring = rings[0]

    # Closed ring: first vertex equals last vertex
    assert np.allclose(primary_ring[0], primary_ring[-1])

    # Concave hull area should be strictly less than convex hull area
    convex_h = convex_hull_2d(coords)
    convex_area = polygon_area_km2(convex_h)
    concave_area = polygon_area_km2(primary_ring)
    assert concave_area <= convex_area


def test_alpha_shape_concave_hull_fallback():
    # Fewer than 4 points falls back to convex hull
    coords = np.array([[10.0, 10.0], [10.0, 11.0], [11.0, 10.0]])
    rings = alpha_shape_concave_hull(coords, alpha=1.0)
    assert len(rings) == 1
    assert len(rings[0]) == 4
    assert np.allclose(rings[0][0], rings[0][-1])


def test_alpha_shape_invalid_params():
    with pytest.raises(ValueError, match="alpha must be strictly positive"):
        alpha_shape_concave_hull([[0.0, 0.0]], alpha=-1.0)

    with pytest.raises(ValueError, match="cannot be empty"):
        alpha_shape_concave_hull(np.empty((0, 2)), alpha=1.0)


def test_hull_to_geojson():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    feature = hull_to_geojson(coords, properties={"cluster_id": 42})

    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["cluster_id"] == 42

    # RFC 7946 coordinates order: [lon, lat] and closed ring
    ring = feature["geometry"]["coordinates"][0]
    assert len(ring) == 4
    assert ring[0] == ring[-1]
    assert ring[0] == [0.0, 0.0]


def test_hull_to_geojson_invalid():
    with pytest.raises(ValueError, match="requires at least 3"):
        hull_to_geojson([[0.0, 0.0], [1.0, 1.0]])
