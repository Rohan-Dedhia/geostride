"""High-precision Vincenty inverse geodesic calculation on the WGS-84 ellipsoid."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# WGS-84 ellipsoid parameters
WGS84_A = 6378137.0  # Semi-major axis in meters
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_B = (1.0 - WGS84_F) * WGS84_A  # Semi-minor axis in meters (6356752.314245)

UNIT_FACTORS = {
    "m": 1.0,
    "km": 0.001,
    "miles": 0.000621371192237334,
    "nm": 0.0005399568034557235,
    "ft": 3.280839895013123,
}


def vincenty_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    unit: str = "km",
    max_iter: int = 200,
    tol: float = 1e-12,
) -> float:
    """Compute the geodesic distance between two points on the WGS-84 ellipsoid using Vincenty's inverse method.

    Parameters
    ----------
    lat1 : float
        Latitude of origin in decimal degrees [-90.0, 90.0].
    lon1 : float
        Longitude of origin in decimal degrees [-180.0, 180.0].
    lat2 : float
        Latitude of destination in decimal degrees [-90.0, 90.0].
    lon2 : float
        Longitude of destination in decimal degrees [-180.0, 180.0].
    unit : {"m", "km", "miles", "nm", "ft"}, default="km"
        Measurement unit for the returned distance.
    max_iter : int, default=200
        Maximum number of iterations for the solver.
    tol : float, default=1e-12
        Convergence tolerance on lambda.

    Returns
    -------
    float
        Geodesic distance in the requested unit.
    """
    if unit not in UNIT_FACTORS:
        raise ValueError(f"Unsupported unit '{unit}'. Supported: {list(UNIT_FACTORS.keys())}")

    # Check for identical points
    if math.isclose(lat1, lat2, abs_tol=1e-11) and math.isclose(lon1, lon2, abs_tol=1e-11):
        return 0.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    L = math.radians(lon2 - lon1)

    a = WGS84_A
    b = WGS84_B
    f = WGS84_F

    # Reduced latitudes
    U1 = math.atan((1.0 - f) * math.tan(phi1))
    U2 = math.atan((1.0 - f) * math.tan(phi2))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)

    lamb = L
    cos_sq_alpha = 0.0
    sin_sigma = 0.0
    cos_sigma = 0.0
    cos2_sigma_m = 0.0
    sigma = 0.0

    for _ in range(max_iter):
        sin_lamb = math.sin(lamb)
        cos_lamb = math.cos(lamb)

        sin_sigma = math.sqrt(
            (cosU2 * sin_lamb) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cos_lamb) ** 2
        )
        if math.isclose(sin_sigma, 0.0, abs_tol=1e-15):
            return 0.0  # Coincident points

        cos_sigma = sinU1 * sinU2 + cosU1 * cosU2 * cos_lamb
        sigma = math.atan2(sin_sigma, cos_sigma)

        sin_alpha = (cosU1 * cosU2 * sin_lamb) / sin_sigma
        cos_sq_alpha = 1.0 - sin_alpha**2

        if math.isclose(cos_sq_alpha, 0.0, abs_tol=1e-15):
            cos2_sigma_m = 0.0  # Equatorial line
        else:
            cos2_sigma_m = cos_sigma - (2.0 * sinU1 * sinU2) / cos_sq_alpha

        C = (f / 16.0) * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha))
        lamb_prev = lamb
        lamb = L + (1.0 - C) * f * sin_alpha * (
            sigma + C * sin_sigma * (cos2_sigma_m + C * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m**2))
        )

        if abs(lamb - lamb_prev) < tol:
            break
    else:
        # Fallback for near-antipodal non-convergence
        from geostride.metrics.haversine import haversine_pairwise

        return float(haversine_pairwise(lat1, lon1, lat2, lon2, unit=unit))

    u_sq = cos_sq_alpha * ((a**2 - b**2) / (b**2))
    A = 1.0 + (u_sq / 16384.0) * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)))
    B = (u_sq / 1024.0) * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)))

    delta_sigma = (
        B
        * sin_sigma
        * (
            cos2_sigma_m
            + 0.25
            * B
            * (
                cos_sigma * (-1.0 + 2.0 * cos2_sigma_m**2)
                - (B / 6.0)
                * cos2_sigma_m
                * (-3.0 + 4.0 * sin_sigma**2)
                * (-3.0 + 4.0 * cos2_sigma_m**2)
            )
        )
    )

    s_meters = b * A * (sigma - delta_sigma)
    return float(s_meters * UNIT_FACTORS[unit])


def vincenty_pairwise(
    coords_a: Sequence[Sequence[float]] | np.ndarray,
    coords_b: Sequence[Sequence[float]] | np.ndarray,
    unit: str = "km",
) -> np.ndarray:
    """Compute pairwise Vincenty distances between corresponding elements of two coordinate arrays.

    Parameters
    ----------
    coords_a : array-like of shape (N, 2)
        Origin coordinates (lat, lon).
    coords_b : array-like of shape (N, 2)
        Destination coordinates (lat, lon).
    unit : {"m", "km", "miles", "nm", "ft"}, default="km"
        Distance unit.

    Returns
    -------
    np.ndarray of shape (N,)
        Computed geodesic distances.
    """
    from geostride.core.coordinates import validate_coordinates

    arr_a = validate_coordinates(coords_a)
    arr_b = validate_coordinates(coords_b)

    if arr_a.shape[0] != arr_b.shape[0]:
        raise ValueError(
            f"Arrays must have matching row counts; got {arr_a.shape[0]} and {arr_b.shape[0]}"
        )

    n = arr_a.shape[0]
    results = np.empty(n, dtype=np.float64)
    for i in range(n):
        results[i] = vincenty_distance(
            arr_a[i, 0], arr_a[i, 1], arr_b[i, 0], arr_b[i, 1], unit=unit
        )
    return results
