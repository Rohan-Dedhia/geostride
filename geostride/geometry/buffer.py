"""Geodesic polygon buffering and equidistant boundary dilation/erosion."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from geostride.core.coordinates import Coordinate, validate_coordinates

_EARTH_RADIUS_KM = 6371.0088


def buffer_point(
    lat: float,
    lon: float,
    distance_km: float,
    n_points: int = 32,
) -> np.ndarray:
    """Compute an exact geodesic circle buffer boundary around a center coordinate.

    Uses great-circle direct forward geodesic equations to produce an equidistant
    circular polygon boundary on the spherical Earth.

    Parameters
    ----------
    lat : float
        Latitude of the center point in decimal degrees [-90.0, 90.0].
    lon : float
        Longitude of the center point in decimal degrees [-180.0, 180.0].
    distance_km : float
        Geodesic buffer radius in kilometers. Must be strictly positive.
    n_points : int, default=32
        Number of boundary vertices along the circle circumference (minimum 4).

    Returns
    -------
    np.ndarray of shape (n_points + 1, 2)
        Closed polygon boundary vertices `[latitude, longitude]`.

    Raises
    ------
    ValueError
        If coordinates are out of bounds, distance_km <= 0, or n_points < 4.

    Examples
    --------
    >>> from geostride.geometry import buffer_point
    >>> circle = buffer_point(40.7128, -74.0060, distance_km=5.0, n_points=16)
    >>> circle.shape
    (17, 2)
    """
    coord = Coordinate(latitude=float(lat), longitude=float(lon))
    if distance_km <= 0.0:
        raise ValueError(f"distance_km must be strictly positive; got {distance_km}")
    if n_points < 4:
        raise ValueError(f"n_points must be at least 4; got {n_points}")

    phi1, lam1 = coord.radians
    delta = distance_km / _EARTH_RADIUS_KM

    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)

    ring: list[list[float]] = []
    for alpha in angles:
        sin_alpha = math.sin(alpha)
        cos_alpha = math.cos(alpha)

        sin_phi2 = sin_phi1 * cos_delta + cos_phi1 * sin_delta * cos_alpha
        phi2 = math.asin(max(-1.0, min(1.0, sin_phi2)))

        y = sin_alpha * sin_delta * cos_phi1
        x = cos_delta - sin_phi1 * sin_phi2
        lam2 = lam1 + math.atan2(y, x)

        deg_lat = math.degrees(phi2)
        deg_lon = ((math.degrees(lam2) + 180.0) % 360.0) - 180.0
        ring.append([deg_lat, deg_lon])

    # Explicitly close the ring
    ring.append(ring[0])
    return np.array(ring, dtype=np.float64)


def buffer_polygon(
    coords: np.ndarray | Sequence[Sequence[float]],
    distance_km: float,
    n_segments: int = 16,
    join_style: str = "round",
) -> np.ndarray:
    """Buffer a geographic polygon boundary outward (dilation) or inward (erosion).

    Projects coordinates to a local equidistant metric plane, computes outward edge offsets
    with exact corner joins, and inverse-projects back to spherical coordinates.

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Polygon boundary vertices `[latitude, longitude]`.
    distance_km : float
        Buffer offset distance in kilometers.
        - Positive: outward dilation / expansion.
        - Negative: inward erosion / contraction.
        - Zero: returns the original polygon as a closed ring.
    n_segments : int, default=16
        Number of curve segments per quarter-circle for 'round' corner joins (minimum 4).
    join_style : {"round", "mitre", "miter", "bevel"}, default="round"
        Style of offset corner joins:
        - "round": smooth circular arc offset joins.
        - "mitre" or "miter": sharp extended miter joins with cutoff threshold.
        - "bevel": flat chamfered corner joins.

    Returns
    -------
    np.ndarray of shape (K, 2)
        Closed polygon coordinates `[latitude, longitude]` of the buffered boundary.
        If an inward erosion collapses the polygon, an empty array of shape (0, 2) is returned.

    Raises
    ------
    ValueError
        If coordinates are invalid, polygon has fewer than 3 distinct vertices,
        or join_style is unsupported.

    Examples
    --------
    >>> import numpy as np
    >>> from geostride.geometry import buffer_polygon
    >>> square = [[40.0, -74.0], [40.1, -74.0], [40.1, -73.9], [40.0, -73.9]]
    >>> buffered = buffer_polygon(square, distance_km=2.0, join_style="round")
    >>> len(buffered) > len(square)
    True
    """
    valid_joins = {"round", "mitre", "miter", "bevel"}
    norm_join = join_style.lower()
    if norm_join not in valid_joins:
        raise ValueError(
            f"Invalid join_style '{join_style}'. Supported styles are: {sorted(valid_joins)}"
        )
    if norm_join == "miter":
        norm_join = "mitre"

    if n_segments < 4:
        raise ValueError(f"n_segments must be at least 4; got {n_segments}")

    arr = validate_coordinates(coords)
    # Deduplicate closing point if present
    if len(arr) >= 4 and np.allclose(arr[0], arr[-1]):
        pts = arr[:-1]
    else:
        pts = arr

    unique_pts = np.unique(pts, axis=0)
    if len(unique_pts) < 3:
        raise ValueError(
            f"A polygon boundary requires at least 3 distinct coordinate vertices; "
            f"got {len(unique_pts)}."
        )

    # If zero distance, return closed ring
    if math.isclose(distance_km, 0.0):
        if not np.allclose(pts[0], pts[-1]):
            return np.vstack([pts, pts[0]])
        return pts

    # Local metric planar projection centered at centroid
    mean_lat = float(np.mean(pts[:, 0]))
    mean_lon = float(np.mean(pts[:, 1]))
    phi0 = math.radians(mean_lat)
    lam0 = math.radians(mean_lon)
    cos_phi0 = math.cos(phi0)

    # Transform [lat, lon] to local (x, y) in kilometers
    dphi = np.radians(pts[:, 0]) - phi0
    dlam = np.radians(pts[:, 1]) - lam0
    # Wrap longitude delta to [-pi, pi]
    dlam = ((dlam + np.pi) % (2.0 * np.pi)) - np.pi

    px = _EARTH_RADIUS_KM * dlam * cos_phi0
    py = _EARTH_RADIUS_KM * dphi
    planar_pts = np.column_stack([px, py])

    # Ensure counter-clockwise (CCW) orientation
    n = len(planar_pts)
    signed_area = 0.5 * sum(
        planar_pts[i, 0] * planar_pts[(i + 1) % n, 1]
        - planar_pts[(i + 1) % n, 0] * planar_pts[i, 1]
        for i in range(n)
    )
    if signed_area < 0:
        planar_pts = planar_pts[::-1]

    d = float(distance_km)

    # Compute offset edges and corner joins
    out_pts: list[np.ndarray] = []

    # First precompute vertices, edge vectors, unit tangents and outward normals
    tangents = []
    normals = []
    for i in range(n):
        p_curr = planar_pts[i]
        p_next = planar_pts[(i + 1) % n]
        v = p_next - p_curr
        length = math.hypot(v[0], v[1])
        if length < 1e-12:
            t = np.array([1.0, 0.0])
            n_vec = np.array([0.0, -1.0])
        else:
            t = v / length
            n_vec = np.array([t[1], -t[0]])
        tangents.append(t)
        normals.append(n_vec)

    # Inward erosion collapse check
    if d < 0.0:
        for i in range(n):
            p_curr = planar_pts[i]
            p_next = planar_pts[(i + 1) % n]

            t1 = tangents[(i - 1) % n]
            t2 = tangents[i]
            n1 = normals[(i - 1) % n]
            n2 = normals[i]

            d1 = 1.0 + float(t1[0] * t2[0] + t1[1] * t2[1])
            d2 = 1.0 + float(tangents[i][0] * tangents[(i + 1) % n][0] + tangents[i][1] * tangents[(i + 1) % n][1])

            if abs(d1) > 1e-6 and abs(d2) > 1e-6:
                m1 = p_curr + d * (n1 + n2) / d1
                m2 = p_next + d * (normals[i] + normals[(i + 1) % n]) / d2
                v_offset = m2 - m1
                v_orig = p_next - p_curr
                # If offset edge flips direction against original edge, polygon collapsed
                if np.dot(v_offset, v_orig) <= 0.0:
                    return np.empty((0, 2), dtype=np.float64)

    for i in range(n):
        p_curr = planar_pts[i]
        t1 = tangents[(i - 1) % n]
        t2 = tangents[i]
        n1 = normals[(i - 1) % n]
        n2 = normals[i]

        cross_val = float(t1[0] * t2[1] - t1[1] * t2[0])
        dot_val = float(t1[0] * t2[0] + t1[1] * t2[1])

        p1_end = p_curr + d * n1
        p2_start = p_curr + d * n2

        if d > 0.0:
            if cross_val > 1e-9:  # Convex corner (left turn)
                if norm_join == "round":
                    th1 = math.atan2(n1[1], n1[0])
                    th2 = math.atan2(n2[1], n2[0])
                    diff = (th2 - th1) % (2.0 * math.pi)
                    if diff < 0.0:
                        diff += 2.0 * math.pi
                    steps = max(2, int(n_segments * (diff / (math.pi / 2.0))))
                    angles = np.linspace(th1, th1 + diff, steps)
                    for a in angles:
                        out_pts.append(p_curr + d * np.array([math.cos(a), math.sin(a)]))
                elif norm_join == "bevel":
                    out_pts.append(p1_end)
                    out_pts.append(p2_start)
                elif norm_join == "mitre":
                    denom = 1.0 + dot_val
                    if abs(denom) > 1e-6:
                        m_pt = p_curr + d * (n1 + n2) / denom
                        miter_dist = math.hypot(m_pt[0] - p_curr[0], m_pt[1] - p_curr[1])
                        if miter_dist <= 3.0 * d:
                            out_pts.append(m_pt)
                        else:
                            out_pts.append(p1_end)
                            out_pts.append(p2_start)
                    else:
                        out_pts.append(p1_end)
                        out_pts.append(p2_start)
            else:  # Concave or collinear corner
                denom = 1.0 + dot_val
                if abs(denom) > 1e-6:
                    out_pts.append(p_curr + d * (n1 + n2) / denom)
                else:
                    out_pts.append(p1_end)
        else:
            # Inward erosion d < 0
            denom = 1.0 + dot_val
            if abs(denom) > 1e-6:
                out_pts.append(p_curr + d * (n1 + n2) / denom)
            else:
                out_pts.append(p1_end)

    if not out_pts:
        return np.empty((0, 2), dtype=np.float64)

    buf_arr = np.array(out_pts, dtype=np.float64)
    # Check if collapsed or self-inverted
    k = len(buf_arr)
    area = 0.5 * sum(
        buf_arr[i, 0] * buf_arr[(i + 1) % k, 1] - buf_arr[(i + 1) % k, 0] * buf_arr[i, 1]
        for i in range(k)
    )
    if area <= 1e-6:
        return np.empty((0, 2), dtype=np.float64)

    # Inverse-project planar coordinates (x, y) back to [latitude, longitude]
    phi_out = phi0 + (buf_arr[:, 1] / _EARTH_RADIUS_KM)
    lam_out = lam0 + (buf_arr[:, 0] / (_EARTH_RADIUS_KM * cos_phi0))

    lat_deg = np.clip(np.degrees(phi_out), -90.0, 90.0)
    lon_deg = ((np.degrees(lam_out) + 180.0) % 360.0) - 180.0

    res_coords = np.column_stack([lat_deg, lon_deg])

    # Ensure closed ring
    if not np.allclose(res_coords[0], res_coords[-1]):
        res_coords = np.vstack([res_coords, res_coords[0]])

    return res_coords
