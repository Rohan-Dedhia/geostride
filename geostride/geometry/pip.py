"""Vectorized Point-in-Polygon (Ray Casting) spatial indexing algorithms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from geostride.core.coordinates import validate_coordinates


def _parse_polygon_rings(polygon: Any) -> tuple[np.ndarray, list[np.ndarray]]:
    """Parse and validate polygon input into an exterior ring and optional interior hole rings.

    Parameters
    ----------
    polygon : array-like, sequence of rings, or GeoJSON dict
        Polygon specification. Supported formats:
        - 2D array-like of shape (M, 2): coordinates of exterior ring [lat, lon].
        - Sequence of rings: first element is exterior ring, subsequent are hole rings.
        - GeoJSON geometry dictionary of type 'Polygon'.

    Returns
    -------
    tuple of (np.ndarray, list of np.ndarray)
        exterior ring coordinates array (M, 2) and list of hole ring arrays.

    Raises
    ------
    ValueError
        If polygon structure is invalid or any ring has fewer than 3 distinct vertices.
    """
    if isinstance(polygon, dict):
        # GeoJSON Feature or Polygon geometry
        geom = polygon.get("geometry", polygon) if polygon.get("type") == "Feature" else polygon
        if not isinstance(geom, dict) or geom.get("type") != "Polygon":
            raise ValueError(
                f"Expected GeoJSON geometry of type 'Polygon'; got {type(geom)}: {geom.get('type')}"
            )
        raw_rings = geom.get("coordinates")
        if not raw_rings or not isinstance(raw_rings, (list, tuple)):
            raise ValueError("GeoJSON Polygon coordinates must be a non-empty sequence of rings.")
        # GeoJSON RFC 7946 coordinates are [lon, lat]; convert to GeoStride [lat, lon]
        rings = [
            np.array([[float(pt[1]), float(pt[0])] for pt in r], dtype=np.float64)
            for r in raw_rings
        ]
        exterior = rings[0]
        holes = rings[1:]
    elif isinstance(polygon, (list, tuple, np.ndarray)):
        arr = np.asarray(polygon)
        if arr.ndim == 2:
            exterior = arr
            holes = []
        elif arr.ndim == 3:
            if len(arr) == 0:
                raise ValueError("Polygon sequence cannot be empty.")
            exterior = arr[0]
            holes = [r for r in arr[1:]]
        elif isinstance(polygon, (list, tuple)):
            if len(polygon) == 0:
                raise ValueError("Polygon sequence cannot be empty.")
            first = np.asarray(polygon[0])
            if first.ndim == 2 and first.shape[1] == 2:
                # Sequence of rings: [exterior, hole1, ...]
                exterior = first
                holes = [np.asarray(h, dtype=np.float64) for h in polygon[1:]]
            elif first.ndim == 1 and first.shape[0] == 2:
                # Sequence of coordinate points representing a single ring
                exterior = np.asarray(polygon, dtype=np.float64)
                holes = []
            else:
                raise ValueError(
                    f"Unsupported polygon ring elements with shape {first.shape}."
                )
        else:
            raise ValueError(f"Unsupported polygon format: {type(polygon)}")
    else:
        raise TypeError(f"Unsupported polygon type: {type(polygon)}")

    # Validate exterior ring
    ext_val = validate_coordinates(exterior)
    unique_ext = np.unique(ext_val, axis=0)
    if len(unique_ext) < 3:
        raise ValueError(
            f"Polygon exterior ring must have at least 3 distinct vertices; got {len(unique_ext)}."
        )

    # Validate holes
    validated_holes: list[np.ndarray] = []
    for idx, h in enumerate(holes):
        h_val = validate_coordinates(h)
        unique_h = np.unique(h_val, axis=0)
        if len(unique_h) < 3:
            raise ValueError(
                f"Polygon hole ring {idx} must have at least 3 distinct vertices; "
                f"got {len(unique_h)}."
            )
        validated_holes.append(h_val)

    return ext_val, validated_holes


def _point_in_ring(
    pts: np.ndarray,
    ring: np.ndarray,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Determine strict interior containment and boundary presence for points within a single ring.

    Parameters
    ----------
    pts : np.ndarray of shape (N, 2)
        Input coordinate points [lat, lon].
    ring : np.ndarray of shape (M, 2)
        Ring coordinate vertices [lat, lon].
    tol : float, default=1e-9
        Tolerance for collinear edge / boundary detection.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - `strict_inside`: boolean mask of points strictly inside ring.
        - `on_boundary`: boolean mask of points lying on boundary edges/vertices.
    """
    n = len(pts)
    strict_inside = np.zeros(n, dtype=bool)
    on_boundary = np.zeros(n, dtype=bool)

    if n == 0:
        return strict_inside, on_boundary

    # Bounding box pre-filter for the ring
    ymin = float(np.min(ring[:, 0])) - tol
    ymax = float(np.max(ring[:, 0])) + tol
    xmin = float(np.min(ring[:, 1])) - tol
    xmax = float(np.max(ring[:, 1])) + tol

    py_all = pts[:, 0]
    px_all = pts[:, 1]

    in_bbox = (py_all >= ymin) & (py_all <= ymax) & (px_all >= xmin) & (px_all <= xmax)
    if not np.any(in_bbox):
        return strict_inside, on_boundary

    active_indices = np.flatnonzero(in_bbox)
    py = py_all[active_indices]
    px = px_all[active_indices]
    active_len = len(active_indices)

    active_inside = np.zeros(active_len, dtype=bool)
    active_boundary = np.zeros(active_len, dtype=bool)

    # Prepare ring edge segments
    if np.allclose(ring[0], ring[-1]):
        v_from = ring[:-1]
        v_to = ring[1:]
    else:
        v_from = ring
        v_to = np.roll(ring, -1, axis=0)

    num_edges = len(v_from)

    for j in range(num_edges):
        y1 = float(v_from[j, 0])
        x1 = float(v_from[j, 1])
        y2 = float(v_to[j, 0])
        x2 = float(v_to[j, 1])

        dy = y2 - y1
        dx = x2 - x1
        l2 = dy * dy + dx * dx

        # Collinear boundary test
        if l2 > 1e-14:
            t = ((py - y1) * dy + (px - x1) * dx) / l2
            t_clamped = np.clip(t, 0.0, 1.0)
            proj_y = y1 + t_clamped * dy
            proj_x = x1 + t_clamped * dx
            dist_sq = (py - proj_y) ** 2 + (px - proj_x) ** 2
            active_boundary |= dist_sq <= (tol * tol)
        else:
            dist_sq = (py - y1) ** 2 + (px - x1) ** 2
            active_boundary |= dist_sq <= (tol * tol)

        # Ray-casting crossing test: horizontal ray along +x direction
        # Half-open vertical interval [min(y1, y2), max(y1, y2))
        cond_y = ((y1 <= py) & (py < y2)) | ((y2 <= py) & (py < y1))
        if np.any(cond_y) and abs(dy) > 1e-14:
            x_int = x1 + (py[cond_y] - y1) * (dx / dy)
            cross = px[cond_y] < x_int
            active_inside[cond_y] ^= cross

    # Strict interior requires odd crossings and not on edge boundary
    strict = active_inside & (~active_boundary)
    strict_inside[active_indices] = strict
    on_boundary[active_indices] = active_boundary

    return strict_inside, on_boundary


def point_in_polygon(
    coords: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
    polygon: np.ndarray | Sequence[Any] | dict[str, Any],
    include_boundary: bool = True,
    tol: float = 1e-9,
) -> np.ndarray:
    """Test whether geographic coordinate points lie inside a polygon boundary.

    Uses a fully vectorized Ray Casting (crossing number) algorithm in pure NumPy.
    Supports convex and concave polygons, polygons with holes (interior rings), and
    bounding-box pre-filtering.

    Parameters
    ----------
    coords : array-like of shape (N, 2) or (2,)
        Input coordinate points [latitude, longitude].
    polygon : array-like, sequence of rings, or GeoJSON dict
        Polygon boundary vertices. Can be:
        - 2D array-like of shape (M, 2) [latitude, longitude].
        - Sequence of rings `[exterior_ring, hole_1, hole_2, ...]`.
        - GeoJSON geometry dictionary of type 'Polygon'.
    include_boundary : bool, default=True
        Whether points lying exactly on polygon boundaries or vertices are considered inside.
        If False, only strictly interior points evaluate to True.
    tol : float, default=1e-9
        Tolerance in decimal degrees for boundary and vertex edge detection.

    Returns
    -------
    np.ndarray of shape (N,), dtype=bool
        Boolean mask where True indicates the coordinate point lies inside the polygon.

    Raises
    ------
    ValueError
        If coordinates or polygon structure are invalid, or if any ring has fewer
        than 3 distinct vertices.

    Examples
    --------
    >>> import numpy as np
    >>> from geostride.geometry import point_in_polygon
    >>> square = [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]]
    >>> pts = [[1.0, 1.0], [3.0, 3.0], [0.0, 1.0]]
    >>> point_in_polygon(pts, square)
    array([ True, False,  True])
    """
    pts = validate_coordinates(coords)
    n = len(pts)
    if n == 0:
        return np.empty(0, dtype=bool)

    exterior, holes = _parse_polygon_rings(polygon)

    ext_inside, ext_boundary = _point_in_ring(pts, exterior, tol=tol)

    if not holes:
        if include_boundary:
            return ext_inside | ext_boundary
        return ext_inside

    # Evaluate interior hole rings
    hole_inside_any = np.zeros(n, dtype=bool)
    hole_boundary_any = np.zeros(n, dtype=bool)

    for hole in holes:
        h_inside, h_boundary = _point_in_ring(pts, hole, tol=tol)
        hole_inside_any |= h_inside
        hole_boundary_any |= h_boundary

    # Points strictly inside polygon with holes
    strict_inside = ext_inside & (~hole_inside_any) & (~hole_boundary_any)

    if include_boundary:
        # Boundary of polygon with holes:
        # - Points on exterior boundary not inside any hole
        # - Points on hole boundaries that fall within the exterior polygon
        poly_boundary = (ext_boundary & (~hole_inside_any)) | (
            (ext_inside | ext_boundary) & hole_boundary_any
        )
        return strict_inside | poly_boundary

    return strict_inside


def points_in_polygons(
    coords: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
    polygons: Sequence[Any],
    include_boundary: bool = True,
    tol: float = 1e-9,
) -> np.ndarray:
    """Find the index of the enclosing polygon for each coordinate point.

    Accelerated by bounding-box pre-filtering so that distant polygons are never
    evaluated against non-candidate points.

    Parameters
    ----------
    coords : array-like of shape (N, 2) or (2,)
        Input coordinate points [latitude, longitude].
    polygons : sequence of polygons
        List of polygon boundaries. Each item can be a 2D array, sequence of rings,
        or GeoJSON geometry dictionary.
    include_boundary : bool, default=True
        Whether points on polygon edges or vertices are considered inside.
    tol : float, default=1e-9
        Tolerance for boundary edge containment.

    Returns
    -------
    np.ndarray of shape (N,), dtype=int64
        Array of integer indices corresponding to the first matching polygon in `polygons`,
        or -1 if the point lies outside all polygons.

    Raises
    ------
    ValueError
        If coordinates or any polygon is malformed.

    Examples
    --------
    >>> poly1 = [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]]
    >>> poly2 = [[3.0, 3.0], [3.0, 5.0], [5.0, 5.0], [5.0, 3.0]]
    >>> pts = [[1.0, 1.0], [4.0, 4.0], [10.0, 10.0]]
    >>> points_in_polygons(pts, [poly1, poly2])
    array([ 0,  1, -1])
    """
    pts = validate_coordinates(coords)
    n = len(pts)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    if len(polygons) == 0:
        return np.full(n, -1, dtype=np.int64)

    # Pre-parse all polygons and pre-compute exterior bounding boxes
    parsed_polys: list[tuple[Any, float, float, float, float]] = []
    for poly in polygons:
        ext, _ = _parse_polygon_rings(poly)
        ymin = float(np.min(ext[:, 0])) - tol
        ymax = float(np.max(ext[:, 0])) + tol
        xmin = float(np.min(ext[:, 1])) - tol
        xmax = float(np.max(ext[:, 1])) + tol
        parsed_polys.append((poly, ymin, ymax, xmin, xmax))

    result = np.full(n, -1, dtype=np.int64)
    py = pts[:, 0]
    px = pts[:, 1]

    for poly_idx, (poly, ymin, ymax, xmin, xmax) in enumerate(parsed_polys):
        unassigned = result == -1
        if not np.any(unassigned):
            break

        cand_mask = unassigned & (py >= ymin) & (py <= ymax) & (px >= xmin) & (px <= xmax)
        if not np.any(cand_mask):
            continue

        cand_indices = np.flatnonzero(cand_mask)
        in_poly = point_in_polygon(
            pts[cand_indices],
            poly,
            include_boundary=include_boundary,
            tol=tol,
        )

        matched = cand_indices[in_poly]
        result[matched] = poly_idx

    return result
