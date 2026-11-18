"""Unit tests for discrete hexagonal (H3-style) spatial tessellation generator."""

import math

import numpy as np
import pytest

from geostride.indexing.hexagon import (
    bin_coordinates_hex,
    coord_to_hex_cell,
    hex_cell_to_centroid,
    hex_cell_to_polygon,
    hex_neighbors,
)
from geostride.metrics.haversine import haversine_pairwise


def test_coord_to_hex_cell_origin():
    q, r = coord_to_hex_cell(0.0, 0.0, edge_km=1.0)
    assert q == 0
    assert r == 0


def test_coord_to_hex_cell_proximity():
    # Points 10 meters apart should fall into the same 5km hexagon cell
    cell1 = coord_to_hex_cell(40.7128, -74.0060, edge_km=5.0)
    cell2 = coord_to_hex_cell(40.7129, -74.0061, edge_km=5.0)
    assert cell1 == cell2


def test_coord_to_hex_cell_separation():
    # Points 100km apart should fall into different 2km hexagon cells
    cell_nyc = coord_to_hex_cell(40.7128, -74.0060, edge_km=2.0)
    cell_phl = coord_to_hex_cell(39.9526, -75.1652, edge_km=2.0)
    assert cell_nyc != cell_phl


def test_coord_to_hex_cell_invalid():
    with pytest.raises(ValueError, match="edge_km must be strictly positive"):
        coord_to_hex_cell(40.0, -74.0, edge_km=0.0)

    with pytest.raises(ValueError, match="edge_km must be strictly positive"):
        coord_to_hex_cell(40.0, -74.0, edge_km=-1.5)

    with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
        coord_to_hex_cell(95.0, 0.0)


def test_hex_cell_to_centroid_roundtrip():
    # Centroid of cell (q, r) should map right back to (q, r)
    test_cells = [(0, 0), (1, 0), (0, 1), (-2, 3), (5, -4), (10, 12)]
    edge_km = 2.5

    for q, r in test_cells:
        lat, lon = hex_cell_to_centroid(q, r, edge_km=edge_km)
        mapped_q, mapped_r = coord_to_hex_cell(lat, lon, edge_km=edge_km)
        assert (mapped_q, mapped_r) == (q, r)


def test_hex_cell_to_polygon_geometry():
    edge_km = 3.0
    poly = hex_cell_to_polygon(0, 0, edge_km=edge_km)

    assert len(poly) == 6
    lat0, lon0 = hex_cell_to_centroid(0, 0, edge_km=edge_km)

    # Each vertex should be approximately edge_km from centroid
    for v_lat, v_lon in poly:
        dist = haversine_pairwise(lat0, lon0, v_lat, v_lon, unit="km")
        assert math.isclose(dist, edge_km, rel_tol=0.05)

    # Distance between adjacent polygon vertices should be approximately edge_km
    for i in range(6):
        v1 = poly[i]
        v2 = poly[(i + 1) % 6]
        edge_dist = haversine_pairwise(v1[0], v1[1], v2[0], v2[1], unit="km")
        assert math.isclose(edge_dist, edge_km, rel_tol=0.05)


def test_hex_neighbors():
    q, r = 3, -2
    edge_km = 1.0
    neighbors = hex_neighbors(q, r)

    assert len(neighbors) == 6
    assert len(set(neighbors)) == 6  # All distinct

    # Distance between cell centroid and any neighbor centroid is sqrt(3) * edge_km
    c_lat, c_lon = hex_cell_to_centroid(q, r, edge_km=edge_km)
    expected_dist = math.sqrt(3.0) * edge_km

    for nq, nr in neighbors:
        n_lat, n_lon = hex_cell_to_centroid(nq, nr, edge_km=edge_km)
        dist = haversine_pairwise(c_lat, c_lon, n_lat, n_lon, unit="km")
        assert math.isclose(dist, expected_dist, rel_tol=0.05)


def test_bin_coordinates_hex_basic():
    # 3 points close together in NYC, 2 points in London
    coords = [
        [40.7128, -74.0060],
        [40.7130, -74.0058],
        [40.7125, -74.0062],
        [51.5074, -0.1278],
        [51.5075, -0.1279],
    ]
    df_binned = bin_coordinates_hex(coords, edge_km=5.0)

    assert len(df_binned) == 2
    assert list(df_binned.columns) == ["q", "r", "count", "centroid_lat", "centroid_lon"]
    assert df_binned.iloc[0]["count"] == 3
    assert df_binned.iloc[1]["count"] == 2
    assert df_binned["count"].sum() == 5


def test_bin_coordinates_hex_empty():
    df_empty = bin_coordinates_hex(np.empty((0, 2)), edge_km=1.0)
    assert len(df_empty) == 0
    assert list(df_empty.columns) == ["q", "r", "count", "centroid_lat", "centroid_lon"]


def test_bin_coordinates_hex_vectorized_scale():
    rng = np.random.default_rng(42)
    # Generate 2,000 random coordinates around Paris
    lats = rng.uniform(48.80, 48.90, size=2000)
    lons = rng.uniform(2.25, 2.45, size=2000)
    coords = np.column_stack([lats, lons])

    df_binned = bin_coordinates_hex(coords, edge_km=1.0)

    assert len(df_binned) > 0
    assert df_binned["count"].sum() == 2000
    assert (df_binned["count"] > 0).all()
    # Centroids should be valid coordinates
    assert (df_binned["centroid_lat"] >= -90).all() and (df_binned["centroid_lat"] <= 90).all()
    assert (df_binned["centroid_lon"] >= -180).all() and (df_binned["centroid_lon"] <= 180).all()
