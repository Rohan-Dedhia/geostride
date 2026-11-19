"""Unit tests for vectorized Point-in-Polygon (PIP) spatial indexing algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from geostride import point_in_polygon, points_in_polygons
from geostride.geometry.pip import _parse_polygon_rings, _point_in_ring


def test_square_pip_interior_exterior() -> None:
    """Test standard square polygon with interior, exterior, edge, and vertex points."""
    # Unit square: [lat, lon] from (0, 0) to (2, 2)
    square = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]])

    pts = np.array([
        [1.0, 1.0],   # Center interior -> True
        [0.5, 1.5],   # Interior -> True
        [3.0, 3.0],   # Exterior top-right -> False
        [-1.0, 1.0],  # Exterior bottom -> False
        [1.0, 3.0],   # Exterior right -> False
        [1.0, -0.5],  # Exterior left -> False
    ])

    mask = point_in_polygon(pts, square)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, [True, True, False, False, False, False])


def test_square_boundary_and_vertices() -> None:
    """Test points lying exactly on boundary edges and vertices with include_boundary toggle."""
    square = [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]]

    pts = np.array([
        [0.0, 0.0],  # Vertex (bottom-left)
        [2.0, 2.0],  # Vertex (top-right)
        [0.0, 1.0],  # Edge (bottom)
        [1.0, 2.0],  # Edge (right)
        [2.0, 1.0],  # Edge (top)
        [1.0, 0.0],  # Edge (left)
        [1.0, 1.0],  # Strict interior
        [3.0, 3.0],  # Strict exterior
    ])

    mask_inc = point_in_polygon(pts, square, include_boundary=True)
    expected_inc = [True, True, True, True, True, True, True, False]
    np.testing.assert_array_equal(mask_inc, expected_inc)

    mask_exc = point_in_polygon(pts, square, include_boundary=False)
    expected_exc = [False, False, False, False, False, False, True, False]
    np.testing.assert_array_equal(mask_exc, expected_exc)


def test_closed_vs_unclosed_polygon_ring() -> None:
    """Ensure rings with explicit closing vertex behave identically to unclosed rings."""
    unclosed = np.array([[10.0, 10.0], [10.0, 20.0], [20.0, 20.0], [20.0, 10.0]])
    closed = np.vstack([unclosed, unclosed[0]])

    pts = np.array([
        [15.0, 15.0],
        [5.0, 15.0],
        [10.0, 15.0],
        [25.0, 25.0],
    ])

    res_unclosed = point_in_polygon(pts, unclosed, include_boundary=True)
    res_closed = point_in_polygon(pts, closed, include_boundary=True)

    np.testing.assert_array_equal(res_unclosed, res_closed)
    np.testing.assert_array_equal(res_closed, [True, False, True, False])


def test_concave_l_shape_polygon() -> None:
    """Test concave L-shaped polygon containment, ensuring cutout corner is exterior."""
    # L-shape:
    # (0, 0) -> (2, 0) -> (2, 1) -> (1, 1) -> (1, 2) -> (0, 2) -> (0, 0)
    l_shape = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ])

    pts = np.array([
        [0.5, 0.5],  # Bottom arm -> True
        [1.5, 0.5],  # Bottom arm -> True
        [0.5, 1.5],  # Left arm -> True
        [1.5, 1.5],  # Cutout notch -> Exterior -> False
        [2.5, 2.5],  # Exterior -> False
        [-0.5, 0.5], # Exterior -> False
    ])

    mask = point_in_polygon(pts, l_shape)
    np.testing.assert_array_equal(mask, [True, True, True, False, False, False])


def test_concave_star_polygon() -> None:
    """Test 5-pointed concave star polygon."""
    # Construct regular 5-pointed star
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    radii = np.where(np.arange(10) % 2 == 0, 2.0, 0.8)
    star_lats = radii * np.cos(angles)
    star_lons = radii * np.sin(angles)
    star = np.column_stack([star_lats, star_lons])

    pts = np.array([
        [0.0, 0.0],    # Star center -> True
        [1.5, 0.0],    # Along tip 0 -> True
        [1.5, 1.5],    # Outer indentation crevice -> False
        [3.0, 0.0],    # Beyond tip -> False
    ])

    mask = point_in_polygon(pts, star)
    np.testing.assert_array_equal(mask, [True, True, False, False])


def test_polygon_with_hole() -> None:
    """Test polygon with interior hole (donut polygon)."""
    # Outer square: [0, 0] to [4, 4]
    outer = np.array([[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]])
    # Inner hole: [1, 1] to [3, 3]
    hole = np.array([[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]])

    poly_with_hole = [outer, hole]

    pts = np.array([
        [0.5, 0.5],  # Inside outer ring, outside hole -> True
        [3.5, 3.5],  # Inside outer ring, outside hole -> True
        [2.0, 2.0],  # Strictly inside hole -> False
        [2.0, 1.0],  # On hole boundary edge
        [0.0, 2.0],  # On outer boundary edge
        [5.0, 5.0],  # Outside outer ring -> False
    ])

    # With include_boundary=True: hole boundary points and outer boundary are True
    mask_inc = point_in_polygon(pts, poly_with_hole, include_boundary=True)
    np.testing.assert_array_equal(mask_inc, [True, True, False, True, True, False])

    # With include_boundary=False: both boundaries are excluded
    mask_exc = point_in_polygon(pts, poly_with_hole, include_boundary=False)
    np.testing.assert_array_equal(mask_exc, [True, True, False, False, False, False])


def test_geojson_polygon_input() -> None:
    """Test GeoJSON Polygon dictionary input (RFC 7946 [lon, lat] ordering)."""
    geojson_poly = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [10.0, 20.0],
                    [30.0, 20.0],
                    [30.0, 40.0],
                    [10.0, 40.0],
                    [10.0, 20.0],
                ]
            ],
        },
    }

    # Points are [lat, lon]
    pts = np.array([
        [30.0, 20.0],  # lat=30 (in [20, 40]), lon=20 (in [10, 30]) -> True
        [10.0, 20.0],  # lat=10 (outside [20, 40]) -> False
        [30.0, 50.0],  # lon=50 (outside [10, 30]) -> False
    ])

    mask = point_in_polygon(pts, geojson_poly)
    np.testing.assert_array_equal(mask, [True, False, False])


def test_points_in_polygons_multi_query() -> None:
    """Test points_in_polygons returning index of enclosing polygon or -1."""
    poly0 = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]])
    poly1 = np.array([[5.0, 5.0], [5.0, 7.0], [7.0, 7.0], [7.0, 5.0]])

    pts = np.array([
        [1.0, 1.0],    # Inside poly0 -> 0
        [6.0, 6.0],    # Inside poly1 -> 1
        [10.0, 10.0],  # Outside all -> -1
        [0.0, 1.0],    # On poly0 boundary -> 0 (with include_boundary=True)
    ])

    indices = points_in_polygons(pts, [poly0, poly1], include_boundary=True)
    assert indices.dtype == np.int64
    np.testing.assert_array_equal(indices, [0, 1, -1, 0])

    indices_exc = points_in_polygons(pts, [poly0, poly1], include_boundary=False)
    np.testing.assert_array_equal(indices_exc, [0, 1, -1, -1])


def test_empty_and_degenerate_inputs() -> None:
    """Test validation and error handling for empty points or degenerate polygons."""
    poly = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]])

    # Empty points array
    empty_pts = np.empty((0, 2))
    res_empty = point_in_polygon(empty_pts, poly)
    assert len(res_empty) == 0
    assert res_empty.dtype == bool

    res_polys_empty = points_in_polygons(empty_pts, [poly])
    assert len(res_polys_empty) == 0
    assert res_polys_empty.dtype == np.int64

    # Empty polygon list in points_in_polygons
    pts = np.array([[1.0, 1.0]])
    res_no_polys = points_in_polygons(pts, [])
    np.testing.assert_array_equal(res_no_polys, [-1])

    # Degenerate polygons (< 3 distinct vertices)
    with pytest.raises(ValueError, match="at least 3 distinct vertices"):
        point_in_polygon(pts, [[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="at least 3 distinct vertices"):
        point_in_polygon(pts, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    # Invalid coordinates (lat > 90)
    with pytest.raises(ValueError, match="Latitude values must be within"):
        point_in_polygon([[95.0, 0.0]], poly)


def test_vectorized_throughput_large_n() -> None:
    """Verify performance and vectorization with 10,000 points."""
    # Complex polygon (octagon)
    t = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    octagon = np.column_stack([10.0 + 5.0 * np.cos(t), 10.0 + 5.0 * np.sin(t)])

    rng = np.random.default_rng(42)
    # 10,000 random points across [0, 20] x [0, 20]
    lats = rng.uniform(0.0, 20.0, size=10_000)
    lons = rng.uniform(0.0, 20.0, size=10_000)
    coords = np.column_stack([lats, lons])

    mask = point_in_polygon(coords, octagon)
    assert len(mask) == 10_000
    assert mask.dtype == bool

    # Octagon area is roughly 2 * sqrt(2) * r^2 ~ 2 * 1.414 * 25 ~ 70.7
    # Total bounding box area is 20 * 20 = 400
    # Expected proportion inside is ~70.7 / 400 ~ 17.7%
    fraction_inside = np.mean(mask)
    assert 0.14 <= fraction_inside <= 0.22


def test_helper_parse_polygon_rings() -> None:
    """Test helper _parse_polygon_rings directly."""
    # 3D array of shape (2, 4, 2)
    outer = np.array([[0.0, 0.0], [0.0, 3.0], [3.0, 3.0], [3.0, 0.0]])
    hole = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 1.0]])
    stacked = np.stack([outer, hole])

    ext, holes = _parse_polygon_rings(stacked)
    assert len(holes) == 1
    np.testing.assert_array_equal(ext, outer)
    np.testing.assert_array_equal(holes[0], hole)

    # Empty rings list
    with pytest.raises(ValueError, match="cannot be empty"):
        _parse_polygon_rings([])

    with pytest.raises(TypeError, match="Unsupported polygon type"):
        _parse_polygon_rings(12345)


def test_helper_point_in_ring_bbox_shortcircuit() -> None:
    """Test that points completely outside bounding box are immediately flagged False."""
    ring = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    pts_far = np.array([[100.0, 100.0], [-50.0, -50.0]])

    strict, on_bound = _point_in_ring(pts_far, ring)
    assert not np.any(strict)
    assert not np.any(on_bound)
