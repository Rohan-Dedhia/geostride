"""Vectorized Haversine distance computations on spherical Earth."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from geostride.core.coordinates import validate_coordinates

# Earth mean radii in various units (WGS84 spherical approximation)
EARTH_RADIUS = {
    "km": 6371.0088,
    "m": 6371008.8,
    "miles": 3958.7613,
    "nm": 3440.0695,
    "ft": 20902260.0,
}


def haversine_pairwise(
    lat1: float | Sequence[float] | np.ndarray,
    lon1: float | Sequence[float] | np.ndarray,
    lat2: float | Sequence[float] | np.ndarray,
    lon2: float | Sequence[float] | np.ndarray,
    unit: str = "km",
) -> float | np.ndarray:
    """Compute the great-circle distance between pairs of points using the Haversine formula.

    Supports scalar inputs as well as 1D/2D numpy arrays with NumPy broadcasting.

    Parameters
    ----------
    lat1 : float or array-like
        Latitude of origin point(s) in decimal degrees.
    lon1 : float or array-like
        Longitude of origin point(s) in decimal degrees.
    lat2 : float or array-like
        Latitude of destination point(s) in decimal degrees.
    lon2 : float or array-like
        Longitude of destination point(s) in decimal degrees.
    unit : {"km", "m", "miles", "nm", "ft"}, default="km"
        Distance measurement unit.

    Returns
    -------
    float or np.ndarray
        Great-circle distance(s) in the specified unit.

    Raises
    ------
    ValueError
        If the unit is not supported or inputs contain NaN values.
    """
    if unit not in EARTH_RADIUS:
        raise ValueError(f"Unsupported unit '{unit}'. Supported units: {list(EARTH_RADIUS.keys())}")

    r = EARTH_RADIUS[unit]

    lat1_rad = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1_rad = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2_rad = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2_rad = np.radians(np.asarray(lon2, dtype=np.float64))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    # Clip numerical floating-point inaccuracies
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))

    dist = r * c
    if isinstance(lat1, (int, float)) and isinstance(lat2, (int, float)):
        return float(dist.item()) if hasattr(dist, "item") else float(dist)
    return dist


def haversine_matrix(
    coords_a: np.ndarray | Sequence[Sequence[float]],
    coords_b: np.ndarray | Sequence[Sequence[float]],
    unit: str = "km",
) -> np.ndarray:
    """Compute the full pairwise Haversine distance matrix between two sets of coordinates.

    Parameters
    ----------
    coords_a : array-like of shape (N, 2)
        First coordinate collection where column 0 is latitude and column 1 is longitude.
    coords_b : array-like of shape (M, 2)
        Second coordinate collection where column 0 is latitude and column 1 is longitude.
    unit : {"km", "m", "miles", "nm", "ft"}, default="km"
        Distance measurement unit.

    Returns
    -------
    np.ndarray of shape (N, M)
        Matrix of pairwise distances where entry (i, j) is the distance from
        `coords_a[i]` to `coords_b[j]`.
    """
    arr_a = validate_coordinates(coords_a)
    arr_b = validate_coordinates(coords_b)

    lat_a = arr_a[:, 0, np.newaxis]  # shape (N, 1)
    lon_a = arr_a[:, 1, np.newaxis]  # shape (N, 1)

    lat_b = arr_b[np.newaxis, :, 0]  # shape (1, M)
    lon_b = arr_b[np.newaxis, :, 1]  # shape (1, M)

    return haversine_pairwise(lat_a, lon_a, lat_b, lon_b, unit=unit)
