"""Convex and concave alpha-shape cluster boundary hull generators."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial import Delaunay, QhullError

from geostride.core.coordinates import validate_coordinates

_EARTH_RADIUS_KM = 6371.0088


def _project_planar_km(coords: np.ndarray) -> np.ndarray:
    """Project geographic coordinates [lat, lon] to planar (x, y) coordinates in km."""
    phi = np.radians(coords[:, 0])
    lam = np.radians(coords[:, 1])
    x = _EARTH_RADIUS_KM * lam * np.cos(phi)
    y = _EARTH_RADIUS_KM * phi
    return np.column_stack([x, y])


def _unproject_planar_km(pts: np.ndarray) -> np.ndarray:
    """Inverse project planar (x, y) in km back to geographic coordinates [lat, lon]."""
    x = pts[:, 0]
    y = pts[:, 1]
    phi = y / _EARTH_RADIUS_KM
    lat = np.clip(np.degrees(phi), -90.0, 90.0)

    cos_phi = np.cos(phi)
    safe = np.abs(cos_phi) > 1e-12
    lon = np.zeros_like(x)
    lon[safe] = np.degrees(x[safe] / (_EARTH_RADIUS_KM * cos_phi[safe]))
    lon = ((lon + 180.0) % 360.0) - 180.0
    return np.column_stack([lat, lon])


def _as_closed_ring(pts: np.ndarray) -> np.ndarray:
    """Ensure coordinate polygon ring is explicitly closed."""
    if len(pts) >= 3 and not np.allclose(pts[0], pts[-1]):
        return np.vstack([pts, pts[0]])
    return pts


def convex_hull_2d(
    coords: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Compute the 2D convex hull of geographic coordinates using Andrew's Monotone Chain.

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Array of (latitude, longitude) coordinate pairs.

    Returns
    -------
    np.ndarray of shape (K, 2)
        Counter-clockwise boundary vertices of the minimum convex bounding polygon.

    Raises
    ------
    ValueError
        If coordinates array is empty.
    """
    arr = validate_coordinates(coords)
    n = len(arr)
    if n == 0:
        raise ValueError("Coordinate array cannot be empty.")
    if n <= 2:
        return arr

    # Deduplicate and sort points lexicographically by (lon, lat)
    unique_pts = np.unique(arr, axis=0)
    if len(unique_pts) <= 2:
        return unique_pts

    sorted_idx = np.lexsort((unique_pts[:, 0], unique_pts[:, 1]))
    pts = unique_pts[sorted_idx]

    # Lower hull
    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2:
            cp = (lower[-1][1] - lower[-2][1]) * (p[0] - lower[-2][0]) - (
                lower[-1][0] - lower[-2][0]
            ) * (p[1] - lower[-2][1])
            if cp <= 1e-12:
                lower.pop()
            else:
                break
        lower.append(p)

    # Upper hull
    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2:
            cp = (upper[-1][1] - upper[-2][1]) * (p[0] - upper[-2][0]) - (
                upper[-1][0] - upper[-2][0]
            ) * (p[1] - upper[-2][1])
            if cp <= 1e-12:
                upper.pop()
            else:
                break
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float64)


def alpha_shape_concave_hull(
    coords: np.ndarray | Sequence[Sequence[float]],
    alpha: float = 1.0,
) -> list[np.ndarray]:
    """Extract non-convex boundary polygons enclosing spatial clusters using alpha shapes.

    Uses Delaunay triangulation circumradius filtering in local planar coordinates.
    As `alpha -> 0`, the boundary converges to the convex hull. As `alpha` increases,
    finer concave details, walking corridors, and irregular shapes are carved out.

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Array of (latitude, longitude) coordinate pairs.
    alpha : float, default=1.0
        Alpha parameter controls boundary tightness (in 1/km). Must be strictly positive.
        Smaller values create looser, convex-like envelopes; larger values create tighter
        concave boundaries.

    Returns
    -------
    list of np.ndarray
        List of closed polygon boundary vertex arrays, each of shape (K, 2) in [lat, lon] order.
        The primary exterior boundary is typically the first element.

    Raises
    ------
    ValueError
        If `alpha <= 0` or coordinates array is empty.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be strictly positive; got {alpha}")

    arr = validate_coordinates(coords)
    n = len(arr)
    if n == 0:
        raise ValueError("Coordinate array cannot be empty.")
    if n < 4:
        return [_as_closed_ring(convex_hull_2d(arr))]

    # Project to planar coordinates (km)
    planar = _project_planar_km(arr)

    # Check for collinearity
    rank = np.linalg.matrix_rank(planar - planar.mean(axis=0))
    if rank < 2:
        return [_as_closed_ring(convex_hull_2d(arr))]

    try:
        tri = Delaunay(planar)
    except (QhullError, ValueError, RuntimeError):
        return [_as_closed_ring(convex_hull_2d(arr))]

    max_circumradius = 1.0 / alpha

    # Filter simplices by circumradius
    kept_edges: defaultdict[tuple[int, int], int] = defaultdict(int)

    for simplex in tri.simplices:
        ia, ib, ic = simplex
        pa, pb, pc = planar[ia], planar[ib], planar[ic]

        a = float(np.linalg.norm(pb - pc))
        b = float(np.linalg.norm(pa - pc))
        c = float(np.linalg.norm(pa - pb))
        s = (a + b + c) / 2.0
        area_sq = max(0.0, s * (s - a) * (s - b) * (s - c))
        area = math.sqrt(area_sq)

        if area > 1e-12:
            circumradius = (a * b * c) / (4.0 * area)
        else:
            circumradius = float("inf")

        if circumradius <= max_circumradius:
            for u, v in [(ia, ib), (ib, ic), (ic, ia)]:
                edge = (min(u, v), max(u, v))
                kept_edges[edge] += 1

    # Boundary edges appear exactly once
    boundary = [edge for edge, count in kept_edges.items() if count == 1]
    if not boundary:
        return [_as_closed_ring(convex_hull_2d(arr))]

    # Build adjacency graph
    adj: defaultdict[int, list[int]] = defaultdict(list)
    for u, v in boundary:
        adj[u].append(v)
        adj[v].append(u)

    # Extract polygon loops
    visited: set[int] = set()
    rings: list[np.ndarray] = []

    for start_node in list(adj.keys()):
        if start_node in visited:
            continue

        curr = start_node
        ring = [curr]
        visited.add(curr)
        prev: int | None = None

        while True:
            neighbors = [n for n in adj[curr] if n != prev]
            next_node: int | None = None

            for cand in neighbors:
                if cand == start_node and len(ring) >= 3:
                    next_node = cand
                    break
                if cand not in visited:
                    next_node = cand
                    break

            if next_node is None:
                # If can close back to start_node
                if start_node in neighbors and len(ring) >= 3:
                    ring.append(start_node)
                break

            if next_node == start_node:
                ring.append(start_node)
                break

            ring.append(next_node)
            visited.add(next_node)
            prev = curr
            curr = next_node

        if len(ring) >= 4 and ring[0] == ring[-1]:
            ring_coords = _as_closed_ring(arr[ring])
            rings.append(ring_coords)

    if not rings:
        return [_as_closed_ring(convex_hull_2d(arr))]

    # Sort rings by area descending so primary envelope is first
    rings.sort(key=polygon_area_km2, reverse=True)
    return rings


def polygon_area_km2(coords: np.ndarray | Sequence[Sequence[float]]) -> float:
    """Compute the geodesic surface area of a polygon on Earth in square kilometers.

    Uses spherical excess polygon integration.

    Parameters
    ----------
    coords : array-like of shape (K, 2)
        Array of (latitude, longitude) coordinate vertices.

    Returns
    -------
    float
        Geodesic surface area in square kilometers.
    """
    arr = validate_coordinates(coords)
    n = len(arr)
    if n < 3:
        return 0.0

    phi = np.radians(arr[:, 0])
    lam = np.radians(arr[:, 1])

    prev_lam = np.roll(lam, 1)
    next_lam = np.roll(lam, -1)

    total = np.sum((next_lam - prev_lam) * np.sin(phi))
    area = abs(total) * (_EARTH_RADIUS_KM**2) / 2.0
    return float(area)


def hull_to_geojson(
    hull_coords: np.ndarray | Sequence[Sequence[float]],
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert boundary hull coordinates into a valid RFC 7946 GeoJSON Polygon feature.

    Parameters
    ----------
    hull_coords : array-like of shape (K, 2)
        Array of (latitude, longitude) polygon vertices.
    properties : dict, optional
        Metadata attributes to serialize into the GeoJSON feature properties.

    Returns
    -------
    dict
        GeoJSON Feature dictionary with geometry type 'Polygon'.

    Raises
    ------
    ValueError
        If fewer than 3 coordinate points are provided.
    """
    arr = validate_coordinates(hull_coords)
    if len(arr) < 3:
        raise ValueError("A polygon boundary requires at least 3 coordinate vertices.")

    # RFC 7946 specifies coordinates in [lon, lat] order and closed rings
    ring = [[float(pt[1]), float(pt[0])] for pt in arr]
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [ring],
        },
        "properties": properties or {},
    }
