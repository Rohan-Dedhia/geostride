"""Spherical trajectory simplification using geodesic Douglas-Peucker algorithm."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from geostride.core.coordinates import validate_coordinates
from geostride.metrics.haversine import EARTH_RADIUS


def _haversine_scalar(lat1: float, lon1: float, lat2: float, lon2: float, radius: float) -> float:
    """Compute great-circle distance between two scalar points in meters."""
    phi1, lam1 = np.radians(lat1), np.radians(lon1)
    phi2, lam2 = np.radians(lat2), np.radians(lon2)
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def _haversine_vector(
    lats: np.ndarray, lons: np.ndarray, lat_ref: float, lon_ref: float, radius: float
) -> np.ndarray:
    """Compute great-circle distances between a 1D vector of coordinates and a reference point."""
    phi1, lam1 = np.radians(lat_ref), np.radians(lon_ref)
    phi2 = np.radians(lats)
    lam2 = np.radians(lons)
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _bearing_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute initial forward azimuth in radians from point 1 to point 2."""
    phi1, lam1 = np.radians(lat1), np.radians(lon1)
    phi2, lam2 = np.radians(lat2), np.radians(lon2)
    y = np.sin(lam2 - lam1) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(lam2 - lam1)
    return float(np.arctan2(y, x))


def _bearing_vector(lat_ref: float, lon_ref: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Compute initial forward azimuths in radians from a reference point to an array of points."""
    phi1, lam1 = np.radians(lat_ref), np.radians(lon_ref)
    phi2 = np.radians(lats)
    lam2 = np.radians(lons)
    y = np.sin(lam2 - lam1) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(lam2 - lam1)
    return np.arctan2(y, x)


def cross_track_distance_meters(
    lat: float,
    lon: float,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    radius: float = EARTH_RADIUS["m"],
) -> float:
    """Compute the geodesic cross-track distance in meters from a point to a great-circle arc segment.

    Accounts for along-track clamping so that points situated beyond the endpoints
    measure distance directly to the closest segment terminus.

    Parameters
    ----------
    lat : float
        Latitude of the query point in decimal degrees.
    lon : float
        Longitude of the query point in decimal degrees.
    start_lat : float
        Latitude of segment starting point in decimal degrees.
    start_lon : float
        Longitude of segment starting point in decimal degrees.
    end_lat : float
        Latitude of segment endpoint in decimal degrees.
    end_lon : float
        Longitude of segment endpoint in decimal degrees.
    radius : float, default=EARTH_RADIUS["m"]
        Earth radius in meters.

    Returns
    -------
    float
        Geodesic distance in meters from the query point to the segment.
    """
    d_se = _haversine_scalar(start_lat, start_lon, end_lat, end_lon, radius)
    if d_se < 1e-6:
        return _haversine_scalar(lat, lon, start_lat, start_lon, radius)

    d_sp = _haversine_scalar(start_lat, start_lon, lat, lon, radius)
    if d_sp < 1e-6:
        return 0.0

    d_ep = _haversine_scalar(end_lat, end_lon, lat, lon, radius)
    if d_ep < 1e-6:
        return 0.0

    theta_se = _bearing_scalar(start_lat, start_lon, end_lat, end_lon)
    theta_sp = _bearing_scalar(start_lat, start_lon, lat, lon)
    delta_start = theta_sp - theta_se

    theta_es = _bearing_scalar(end_lat, end_lon, start_lat, start_lon)
    theta_ep = _bearing_scalar(end_lat, end_lon, lat, lon)
    delta_end = theta_ep - theta_es

    if np.cos(delta_start) <= 0.0:
        return d_sp
    if np.cos(delta_end) <= 0.0:
        return d_ep

    delta13 = d_sp / radius
    delta_xt = np.arcsin(np.clip(np.sin(delta13) * np.sin(delta_start), -1.0, 1.0))
    return float(abs(delta_xt) * radius)


def _vectorized_segment_distances(
    lats: np.ndarray,
    lons: np.ndarray,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float,
) -> np.ndarray:
    """Compute vectorized geodesic distances from an array of coordinates to a segment."""
    d_se = _haversine_scalar(lat1, lon1, lat2, lon2, radius)
    if d_se < 1e-6:
        return _haversine_vector(lats, lons, lat1, lon1, radius)

    d_sp = _haversine_vector(lats, lons, lat1, lon1, radius)
    d_ep = _haversine_vector(lats, lons, lat2, lon2, radius)

    theta_se = _bearing_scalar(lat1, lon1, lat2, lon2)
    theta_sp = _bearing_vector(lat1, lon1, lats, lons)
    delta_start = theta_sp - theta_se

    theta_es = _bearing_scalar(lat2, lon2, lat1, lon1)
    theta_ep = _bearing_vector(lat2, lon2, lats, lons)
    delta_end = theta_ep - theta_es

    delta13 = d_sp / radius
    delta_xt = np.arcsin(np.clip(np.sin(delta13) * np.sin(delta_start), -1.0, 1.0))
    xt_dist = np.abs(delta_xt) * radius

    mask_start = np.cos(delta_start) <= 0.0
    mask_end = np.cos(delta_end) <= 0.0

    return np.where(mask_start, d_sp, np.where(mask_end, d_ep, xt_dist))


def simplify_trajectory(
    coords: np.ndarray | Sequence[Sequence[float]],
    tolerance_meters: float,
    radius: float = EARTH_RADIUS["m"],
) -> np.ndarray:
    """Simplify a GPS trajectory using the geodesic spherical Douglas-Peucker algorithm.

    Iteratively prunes intermediate vertices whose cross-track error to the chord segment
    falls below `tolerance_meters`.

    Parameters
    ----------
    coords : np.ndarray or Sequence[Sequence[float]]
        Input coordinates of shape (N, 2) with columns [latitude, longitude].
    tolerance_meters : float
        Maximum allowable perpendicular cross-track error in meters. Must be > 0.
    radius : float, default=EARTH_RADIUS["m"]
        Earth radius in meters.

    Returns
    -------
    np.ndarray
        Simplified coordinate matrix of shape (M, 2) where M <= N.

    Raises
    ------
    ValueError
        If tolerance_meters <= 0 or coordinates fail validation.
    """
    if tolerance_meters <= 0.0:
        raise ValueError(
            f"tolerance_meters must be strictly positive; got {tolerance_meters}"
        )

    arr = validate_coordinates(coords)
    n = len(arr)
    if n < 3:
        return arr.copy()

    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    # Use explicit stack to avoid Python recursion limits
    stack: list[tuple[int, int]] = [(0, n - 1)]

    while stack:
        start_idx, end_idx = stack.pop()
        if end_idx - start_idx <= 1:
            continue

        lat1, lon1 = arr[start_idx, 0], arr[start_idx, 1]
        lat2, lon2 = arr[end_idx, 0], arr[end_idx, 1]

        sub_lats = arr[start_idx + 1 : end_idx, 0]
        sub_lons = arr[start_idx + 1 : end_idx, 1]

        dists = _vectorized_segment_distances(sub_lats, sub_lons, lat1, lon1, lat2, lon2, radius)
        max_idx_offset = int(np.argmax(dists))
        max_dist = float(dists[max_idx_offset])
        max_idx = start_idx + 1 + max_idx_offset

        if max_dist > tolerance_meters:
            keep[max_idx] = True
            stack.append((start_idx, max_idx))
            stack.append((max_idx, end_idx))

    return arr[keep]
