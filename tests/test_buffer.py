"""Unit tests for geodesic polygon buffering and boundary offset dilation/erosion."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geostride import (
    buffer_point,
    buffer_polygon,
    haversine_pairwise,
    point_in_polygon,
    polygon_area_km2,
)


def test_buffer_point_geometry_and_area() -> None:
    """Test geodesic circle generation around center point."""
    center_lat, center_lon = 40.7128, -74.0060
    radius_km = 5.0
    n_pts = 32

    circle = buffer_point(center_lat, center_lon, distance_km=radius_km, n_points=n_pts)

    assert isinstance(circle, np.ndarray)
    assert circle.shape == (n_pts + 1, 2)
    # Ring must be closed
    np.testing.assert_array_equal(circle[0], circle[-1])

    # Geodesic radius check from each boundary vertex back to center
    dists = haversine_pairwise(circle[:, 0], circle[:, 1], center_lat, center_lon, unit="km")
    for d in dists:
        assert math.isclose(d, radius_km, rel_tol=0.01)

    # Surface area check: pi * r^2 ~ 78.54 km^2
    computed_area = polygon_area_km2(circle)
    expected_area = math.pi * (radius_km**2)
    assert math.isclose(computed_area, expected_area, rel_tol=0.02)


def test_buffer_point_invalid_args() -> None:
    """Test validation errors for buffer_point."""
    with pytest.raises(ValueError, match="distance_km must be strictly positive"):
        buffer_point(0.0, 0.0, distance_km=0.0)

    with pytest.raises(ValueError, match="distance_km must be strictly positive"):
        buffer_point(0.0, 0.0, distance_km=-5.0)

    with pytest.raises(ValueError, match="n_points must be at least 4"):
        buffer_point(0.0, 0.0, distance_km=5.0, n_points=3)

    with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
        buffer_point(95.0, 0.0, distance_km=5.0)


def test_buffer_polygon_outward_dilation() -> None:
    """Test expanding a polygon boundary outward with round, bevel, and mitre joins."""
    # 0.1 deg x 0.1 deg rectangle (~11 km x ~8.4 km in NY)
    poly = np.array([
        [40.70, -74.00],
        [40.80, -74.00],
        [40.80, -73.90],
        [40.70, -73.90],
    ])
    orig_area = polygon_area_km2(poly)
    dist_km = 2.0

    buf_round = buffer_polygon(poly, distance_km=dist_km, join_style="round")
    buf_bevel = buffer_polygon(poly, distance_km=dist_km, join_style="bevel")
    buf_mitre = buffer_polygon(poly, distance_km=dist_km, join_style="mitre")

    area_round = polygon_area_km2(buf_round)
    area_bevel = polygon_area_km2(buf_bevel)
    area_mitre = polygon_area_km2(buf_mitre)

    # Area hierarchy: orig < bevel < round < mitre
    assert orig_area < area_bevel < area_round < area_mitre

    # All original polygon points must be strictly inside the dilated boundaries
    assert np.all(point_in_polygon(poly, buf_round))
    assert np.all(point_in_polygon(poly, buf_bevel))
    assert np.all(point_in_polygon(poly, buf_mitre))

    # All buffered polygons must be closed
    assert np.allclose(buf_round[0], buf_round[-1])
    assert np.allclose(buf_bevel[0], buf_bevel[-1])
    assert np.allclose(buf_mitre[0], buf_mitre[-1])


def test_buffer_polygon_inward_erosion() -> None:
    """Test shrinking a polygon boundary inward (negative distance_km)."""
    # 0.2 deg x 0.2 deg square (~22 km x ~17 km)
    poly = np.array([
        [40.60, -74.10],
        [40.80, -74.10],
        [40.80, -73.90],
        [40.60, -73.90],
    ])
    orig_area = polygon_area_km2(poly)

    # Erode by 2 km
    eroded = buffer_polygon(poly, distance_km=-2.0)
    assert len(eroded) > 0
    eroded_area = polygon_area_km2(eroded)

    assert 0 < eroded_area < orig_area

    # All points of the eroded polygon must lie inside the original polygon
    assert np.all(point_in_polygon(eroded, poly))


def test_buffer_polygon_collapse() -> None:
    """Test that excessive inward erosion collapses polygon to empty array."""
    # Small polygon ~1 km wide
    poly = np.array([
        [40.71, -74.00],
        [40.72, -74.00],
        [40.72, -73.99],
        [40.71, -73.99],
    ])

    # Inward erosion of 5 km exceeds the inradius of 1 km polygon
    collapsed = buffer_polygon(poly, distance_km=-5.0)
    assert isinstance(collapsed, np.ndarray)
    assert len(collapsed) == 0
    assert collapsed.shape == (0, 2)


def test_buffer_polygon_zero_distance() -> None:
    """Test buffer with distance_km=0.0 returns closed original boundary."""
    poly = np.array([
        [10.0, 20.0],
        [15.0, 20.0],
        [15.0, 25.0],
        [10.0, 25.0],
    ])
    res = buffer_polygon(poly, distance_km=0.0)
    assert len(res) == len(poly) + 1
    np.testing.assert_array_equal(res[0], res[-1])
    np.testing.assert_array_equal(res[:-1], poly)


def test_buffer_polygon_validation_errors() -> None:
    """Test validation and error handling for buffer_polygon."""
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # Invalid join style
    with pytest.raises(ValueError, match="Invalid join_style"):
        buffer_polygon(poly, distance_km=1.0, join_style="zigzag")

    # Too few vertices (< 3)
    with pytest.raises(ValueError, match="at least 3 distinct coordinate vertices"):
        buffer_polygon([[0, 0], [1, 1]], distance_km=1.0)

    # n_segments < 4
    with pytest.raises(ValueError, match="n_segments must be at least 4"):
        buffer_polygon(poly, distance_km=1.0, n_segments=2)

    # Invalid lat/lon
    with pytest.raises(ValueError, match="Latitude values must be within"):
        buffer_polygon([[95.0, 0.0], [0.0, 1.0], [1.0, 1.0]], distance_km=1.0)
