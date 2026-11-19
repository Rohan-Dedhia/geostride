"""Spherical Voronoi diagram generator for continuous geographic space partitioning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import SphericalVoronoi

from geostride.core.coordinates import validate_coordinates

_DEFAULT_EARTH_RADIUS_KM = 6371.0088


def _coords_to_unit_sphere(coords: np.ndarray) -> np.ndarray:
    """Convert (latitude, longitude) coordinates to 3D Cartesian points on the unit sphere."""
    phi = np.radians(coords[:, 0])
    lam = np.radians(coords[:, 1])
    x = np.cos(phi) * np.cos(lam)
    y = np.cos(phi) * np.sin(lam)
    z = np.sin(phi)
    return np.column_stack([x, y, z])


def _unit_sphere_to_coords(pts_3d: np.ndarray) -> np.ndarray:
    """Convert 3D Cartesian points on the unit sphere back to (latitude, longitude)."""
    x = pts_3d[:, 0]
    y = pts_3d[:, 1]
    z = pts_3d[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    z_norm = np.clip(z / r, -1.0, 1.0)
    lat = np.degrees(np.arcsin(z_norm))
    lon = np.degrees(np.arctan2(y, x))
    return np.column_stack([lat, lon])


@dataclass
class SphericalVoronoiResult:
    """Container holding spherical Voronoi tessellation results.

    Attributes
    ----------
    generators : np.ndarray of shape (N, 2)
        Array of generator site coordinates [latitude, longitude].
    regions : list of list of int
        List of vertex index lists defining the spherical polygon for each generator.
    vertices : np.ndarray of shape (M, 2)
        Array of Voronoi junction vertices [latitude, longitude].
    cell_polygons : list of np.ndarray
        List of closed boundary polygon vertex arrays of shape (K, 2) in [lat, lon] order.
    cell_areas_km2 : np.ndarray of shape (N,)
        Geodesic spherical surface area of each Voronoi cell in square kilometers.
    radius_km : float
        Spherical Earth radius used for area computations.
    """

    generators: np.ndarray
    regions: list[list[int]]
    vertices: np.ndarray
    cell_polygons: list[np.ndarray]
    cell_areas_km2: np.ndarray
    radius_km: float = _DEFAULT_EARTH_RADIUS_KM

    @property
    def total_area_km2(self) -> float:
        """Total covered surface area across all Voronoi cells in km^2."""
        return float(np.sum(self.cell_areas_km2))

    def __repr__(self) -> str:
        return (
            f"SphericalVoronoiResult(n_generators={len(self.generators)}, "
            f"n_vertices={len(self.vertices)}, total_area_km2={self.total_area_km2:,.1f})"
        )


def spherical_voronoi(
    coords: np.ndarray | Sequence[Sequence[float]],
    radius_km: float = _DEFAULT_EARTH_RADIUS_KM,
) -> SphericalVoronoiResult:
    """Compute exact Voronoi spatial partitions on the surface of a spherical Earth.

    Each generator point is assigned a polygonal spherical territory encompassing
    all locations on the sphere closer to that generator than to any other.

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Generator site coordinates in decimal degrees [latitude, longitude].
        At least 4 non-coplanar points are required.
    radius_km : float, default=6371.0088
        Radius of the spherical Earth in kilometers.

    Returns
    -------
    SphericalVoronoiResult
        Structured container with regions, vertices, cell polygons, and geodesic areas.

    Raises
    ------
    ValueError
        If fewer than 4 points are supplied, coordinates contain duplicates,
        points are coplanar in 3D, or `radius_km <= 0`.
    """
    if radius_km <= 0:
        raise ValueError(f"radius_km must be strictly positive; got {radius_km}")

    arr = validate_coordinates(coords)
    n = len(arr)

    if n < 4:
        raise ValueError(
            f"Spherical Voronoi diagram requires at least 4 generator points; got {n}."
        )

    # Check for duplicate coordinates
    unique_arr = np.unique(arr, axis=0)
    if len(unique_arr) < n:
        raise ValueError("Generator coordinates must be unique; duplicate points detected.")

    pts_3d = _coords_to_unit_sphere(arr)

    # Check 3D coplanarity (rank must be 3)
    if np.linalg.matrix_rank(pts_3d) < 3:
        raise ValueError("Generator points must not be coplanar in 3D unit space.")

    # Compute Voronoi diagram on unit sphere
    sv = SphericalVoronoi(pts_3d, radius=1.0, center=[0.0, 0.0, 0.0])
    sv.sort_vertices_of_regions()

    # Convert 3D Voronoi vertices back to geographic coordinates
    vertices_2d = _unit_sphere_to_coords(sv.vertices)

    # Extract cell polygons in counter-clockwise order and close the ring
    cell_polygons: list[np.ndarray] = []
    for region in sv.regions:
        poly_pts = vertices_2d[region]
        if len(poly_pts) >= 3 and not np.allclose(poly_pts[0], poly_pts[-1]):
            poly_pts = np.vstack([poly_pts, poly_pts[0]])
        cell_polygons.append(poly_pts)

    # Compute exact spherical areas in km^2
    areas_unit_sphere = sv.calculate_areas()
    cell_areas_km2 = areas_unit_sphere * (radius_km**2)

    return SphericalVoronoiResult(
        generators=arr,
        regions=sv.regions,
        vertices=vertices_2d,
        cell_polygons=cell_polygons,
        cell_areas_km2=cell_areas_km2,
        radius_km=radius_km,
    )


def voronoi_to_geojson(
    result: SphericalVoronoiResult,
    properties: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Export a SphericalVoronoiResult into an RFC 7946 GeoJSON FeatureCollection.

    Parameters
    ----------
    result : SphericalVoronoiResult
        Computed Voronoi diagram result.
    properties : list of dict, optional
        Optional per-site attribute dictionaries to serialize into feature properties.

    Returns
    -------
    dict
        GeoJSON FeatureCollection object.

    Raises
    ------
    ValueError
        If length of properties does not match number of Voronoi cells.
    """
    n_cells = len(result.cell_polygons)
    if properties is not None and len(properties) != n_cells:
        raise ValueError(
            f"Length of properties ({len(properties)}) does not match number of cells ({n_cells})."
        )

    features: list[dict[str, Any]] = []

    for i, poly in enumerate(result.cell_polygons):
        # RFC 7946: [longitude, latitude]
        ring = [[float(pt[1]), float(pt[0])] for pt in poly]
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        props: dict[str, Any] = {
            "site_id": i,
            "generator_lat": float(result.generators[i, 0]),
            "generator_lon": float(result.generators[i, 1]),
            "area_km2": float(result.cell_areas_km2[i]),
        }
        if properties is not None:
            props.update(properties[i])

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring],
                },
                "properties": props,
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
    }
