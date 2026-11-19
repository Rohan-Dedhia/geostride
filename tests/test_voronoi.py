"""Unit tests for spherical Voronoi diagram generator and GeoJSON exporter."""

import math

import numpy as np
import pytest

from geostride.indexing.voronoi import (
    SphericalVoronoiResult,
    spherical_voronoi,
    voronoi_to_geojson,
)


@pytest.fixture
def octahedron_coords():
    # 6 orthogonal points forming a regular octahedron on the sphere
    return np.array(
        [
            [90.0, 0.0],  # North Pole
            [-90.0, 0.0],  # South Pole
            [0.0, 0.0],  # Prime Meridian Equator
            [0.0, 90.0],  # 90E Equator
            [0.0, 180.0],  # Antimeridian Equator
            [0.0, -90.0],  # 90W Equator
        ]
    )


def test_spherical_voronoi_area_conservation(octahedron_coords):
    radius_km = 6371.0088
    res = spherical_voronoi(octahedron_coords, radius_km=radius_km)

    assert isinstance(res, SphericalVoronoiResult)
    assert len(res.generators) == 6
    assert len(res.cell_polygons) == 6
    assert len(res.cell_areas_km2) == 6

    # Expected surface area of a sphere: 4 * pi * R^2
    expected_total_area = 4.0 * math.pi * (radius_km**2)
    assert math.isclose(res.total_area_km2, expected_total_area, rel_tol=1e-5)

    # For a symmetric regular octahedron, all 6 cells should have equal surface areas
    expected_cell_area = expected_total_area / 6.0
    for area in res.cell_areas_km2:
        assert math.isclose(area, expected_cell_area, rel_tol=1e-4)


def test_spherical_voronoi_polygon_closure(octahedron_coords):
    res = spherical_voronoi(octahedron_coords)

    for poly in res.cell_polygons:
        assert len(poly) >= 4  # At least triangle + closing vertex
        # First point equals last point
        assert np.allclose(poly[0], poly[-1])
        # Latitude within [-90, 90] and longitude within [-180, 180]
        assert (poly[:, 0] >= -90.0).all() and (poly[:, 0] <= 90.0).all()
        assert (poly[:, 1] >= -180.0).all() and (poly[:, 1] <= 180.0).all()


def test_spherical_voronoi_repr(octahedron_coords):
    res = spherical_voronoi(octahedron_coords)
    r = repr(res)
    assert "SphericalVoronoiResult" in r
    assert "n_generators=6" in r
    assert "total_area_km2=" in r


def test_spherical_voronoi_invalid_inputs():
    # Fewer than 4 points
    with pytest.raises(ValueError, match="at least 4 generator points"):
        spherical_voronoi([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])

    # Duplicate coordinates
    coords_with_dups = [
        [0.0, 0.0],
        [0.0, 0.0],
        [45.0, 45.0],
        [-45.0, -45.0],
    ]
    with pytest.raises(ValueError, match="duplicate points detected"):
        spherical_voronoi(coords_with_dups)

    # Coplanar points (all 4 points on the equator plane z=0)
    equator_points = [
        [0.0, 0.0],
        [0.0, 90.0],
        [0.0, 180.0],
        [0.0, -90.0],
    ]
    with pytest.raises(ValueError, match="must not be coplanar"):
        spherical_voronoi(equator_points)

    # Invalid radius
    octahedron = [
        [90.0, 0.0],
        [-90.0, 0.0],
        [0.0, 0.0],
        [0.0, 90.0],
    ]
    with pytest.raises(ValueError, match="radius_km must be strictly positive"):
        spherical_voronoi(octahedron, radius_km=-10.0)


def test_voronoi_to_geojson(octahedron_coords):
    res = spherical_voronoi(octahedron_coords)
    props = [{"label": f"Station_{i}"} for i in range(len(octahedron_coords))]

    fc = voronoi_to_geojson(res, properties=props)

    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 6

    f0 = fc["features"][0]
    assert f0["type"] == "Feature"
    assert f0["geometry"]["type"] == "Polygon"
    assert f0["properties"]["label"] == "Station_0"
    assert "area_km2" in f0["properties"]
    assert "generator_lat" in f0["properties"]
    assert "generator_lon" in f0["properties"]

    # Coordinates format: [lon, lat] and closed
    coords = f0["geometry"]["coordinates"][0]
    assert coords[0] == coords[-1]


def test_voronoi_to_geojson_mismatched_properties(octahedron_coords):
    res = spherical_voronoi(octahedron_coords)
    with pytest.raises(ValueError, match="does not match number of cells"):
        voronoi_to_geojson(res, properties=[{"a": 1}])
