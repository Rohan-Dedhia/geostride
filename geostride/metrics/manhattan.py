"""Spatial Manhattan (taxicab) distance metrics for urban grid analytics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from geostride.core.coordinates import validate_coordinates
from geostride.metrics.haversine import EARTH_RADIUS, haversine_pairwise


def spatial_manhattan_distance(
    lat1: float | Sequence[float] | np.ndarray,
    lon1: float | Sequence[float] | np.ndarray,
    lat2: float | Sequence[float] | np.ndarray,
    lon2: float | Sequence[float] | np.ndarray,
    unit: str = "km",
) -> float | np.ndarray:
    """Compute the spatial Manhattan (L1 / taxicab) distance between coordinates.

    In urban street networks aligned roughly north-south and east-west, Euclidean or
    great-circle distances underestimate actual route travel distances. Spatial Manhattan
    distance evaluates:
        dist_total = dist_latitudinal + dist_longitudinal

    Parameters
    ----------
    lat1, lon1 : float or array-like
        Origin coordinate point(s).
    lat2, lon2 : float or array-like
        Destination coordinate point(s).
    unit : {"km", "m", "miles", "nm", "ft"}, default="km"
        Distance measurement unit.

    Returns
    -------
    float or np.ndarray
        Manhattan distance along orthogonal north-south and east-west components.
    """
    if unit not in EARTH_RADIUS:
        raise ValueError(f"Unsupported unit '{unit}'. Supported units: {list(EARTH_RADIUS.keys())}")

    # Latitudinal travel (moving strictly north-south along meridian from lat1 to lat2)
    dy = haversine_pairwise(lat1, lon1, lat2, lon1, unit=unit)

    # Longitudinal travel (moving strictly east-west along parallel at lat2 from lon1 to lon2)
    dx = haversine_pairwise(lat2, lon1, lat2, lon2, unit=unit)

    return dy + dx


def spatial_manhattan_matrix(
    coords_a: np.ndarray | Sequence[Sequence[float]],
    coords_b: np.ndarray | Sequence[Sequence[float]],
    unit: str = "km",
) -> np.ndarray:
    """Compute the pairwise Manhattan distance matrix between two sets of coordinates.

    Parameters
    ----------
    coords_a : array-like of shape (N, 2)
        First coordinate collection.
    coords_b : array-like of shape (M, 2)
        Second coordinate collection.
    unit : {"km", "m", "miles", "nm", "ft"}, default="km"
        Distance measurement unit.

    Returns
    -------
    np.ndarray of shape (N, M)
        Pairwise Manhattan distance matrix.
    """
    arr_a = validate_coordinates(coords_a)
    arr_b = validate_coordinates(coords_b)

    lat_a = arr_a[:, 0, np.newaxis]  # (N, 1)
    lon_a = arr_a[:, 1, np.newaxis]  # (N, 1)

    lat_b = arr_b[np.newaxis, :, 0]  # (1, M)
    lon_b = arr_b[np.newaxis, :, 1]  # (1, M)

    return spatial_manhattan_distance(lat_a, lon_a, lat_b, lon_b, unit=unit)
