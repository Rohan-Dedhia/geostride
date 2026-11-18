"""Discrete hexagonal (H3-style) spatial tessellation generator and binning."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from geostride.core.coordinates import Coordinate, validate_coordinates

# Mean Earth radius in kilometers (IUGG R1)
_EARTH_RADIUS_KM = 6371.0088


def _project_point(lat: float, lon: float) -> tuple[float, float]:
    """Project (latitude, longitude) to local planar coordinates (x, y) in kilometers."""
    phi = math.radians(lat)
    lam = math.radians(lon)
    x = _EARTH_RADIUS_KM * lam * math.cos(phi)
    y = _EARTH_RADIUS_KM * phi
    return x, y


def _unproject_point(x: float, y: float) -> tuple[float, float]:
    """Inverse-project planar coordinates (x, y) in kilometers back to (latitude, longitude)."""
    phi = y / _EARTH_RADIUS_KM
    lat = math.degrees(phi)
    lat = max(-90.0, min(90.0, lat))

    cos_phi = math.cos(phi)
    if abs(cos_phi) > 1e-12:
        lon = math.degrees(x / (_EARTH_RADIUS_KM * cos_phi))
        lon = ((lon + 180.0) % 360.0) - 180.0
    else:
        lon = 0.0

    return lat, lon


def coord_to_hex_cell(
    latitude: float,
    longitude: float,
    edge_km: float = 1.0,
) -> tuple[int, int]:
    """Map a geographic coordinate pair to discrete axial hexagonal grid indices (q, r).

    Uses a regular pointy-topped hexagon lattice with cube-coordinate nearest-center
    rounding.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees within [-90, 90].
    longitude : float
        Longitude in decimal degrees within [-180, 180].
    edge_km : float, default=1.0
        Length of a hexagon edge in kilometers (must be > 0).

    Returns
    -------
    tuple of int
        Axial hexagonal grid coordinate pair `(q, r)`.

    Raises
    ------
    ValueError
        If `edge_km <= 0` or coordinates fall outside valid bounds.
    """
    if edge_km <= 0:
        raise ValueError(f"edge_km must be strictly positive; got {edge_km}")

    # Validate coordinate bounds
    coord = Coordinate(latitude=latitude, longitude=longitude)
    x, y = _project_point(coord.latitude, coord.longitude)

    # Fractional axial coordinates for pointy-topped hexagon
    r_f = (2.0 / 3.0) * (y / edge_km)
    q_f = (math.sqrt(3.0) / 3.0) * (x / edge_km) - (1.0 / 3.0) * (y / edge_km)

    # Cube coordinate rounding
    x_c = q_f
    z_c = r_f
    y_c = -x_c - z_c

    rx = round(x_c)
    ry = round(y_c)
    rz = round(z_c)

    dx = abs(rx - x_c)
    dy = abs(ry - y_c)
    dz = abs(rz - z_c)

    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return int(rx), int(rz)


def hex_cell_to_centroid(
    q: int,
    r: int,
    edge_km: float = 1.0,
) -> tuple[float, float]:
    """Compute the geographic centroid (latitude, longitude) of a hexagonal cell.

    Parameters
    ----------
    q : int
        Axial column index.
    r : int
        Axial row index.
    edge_km : float, default=1.0
        Length of a hexagon edge in kilometers (must be > 0).

    Returns
    -------
    tuple of float
        `(latitude, longitude)` of the cell center in decimal degrees.

    Raises
    ------
    ValueError
        If `edge_km <= 0`.
    """
    if edge_km <= 0:
        raise ValueError(f"edge_km must be strictly positive; got {edge_km}")

    x0 = edge_km * (math.sqrt(3.0) * q + 0.5 * math.sqrt(3.0) * r)
    y0 = edge_km * 1.5 * r
    return _unproject_point(x0, y0)


def hex_cell_to_polygon(
    q: int,
    r: int,
    edge_km: float = 1.0,
) -> list[tuple[float, float]]:
    """Compute the 6 boundary vertex coordinates of a hexagonal cell.

    Vertices are returned in counter-clockwise order around the cell boundary.

    Parameters
    ----------
    q : int
        Axial column index.
    r : int
        Axial row index.
    edge_km : float, default=1.0
        Length of a hexagon edge in kilometers (must be > 0).

    Returns
    -------
    list of tuple of float
        List of 6 `(latitude, longitude)` coordinate tuples.

    Raises
    ------
    ValueError
        If `edge_km <= 0`.
    """
    if edge_km <= 0:
        raise ValueError(f"edge_km must be strictly positive; got {edge_km}")

    x0 = edge_km * (math.sqrt(3.0) * q + 0.5 * math.sqrt(3.0) * r)
    y0 = edge_km * 1.5 * r

    vertices: list[tuple[float, float]] = []
    for k in range(6):
        theta = (math.pi / 6.0) + (k * math.pi / 3.0)
        xk = x0 + edge_km * math.cos(theta)
        yk = y0 + edge_km * math.sin(theta)
        vertices.append(_unproject_point(xk, yk))

    return vertices


def hex_neighbors(q: int, r: int) -> list[tuple[int, int]]:
    """Return the axial coordinates of the 6 immediate neighbor cells.

    Parameters
    ----------
    q : int
        Axial column index.
    r : int
        Axial row index.

    Returns
    -------
    list of tuple of int
        6 adjacent `(q, r)` cell tuples.
    """
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    return [(q + dq, r + dr) for dq, dr in directions]


def bin_coordinates_hex(
    coords: np.ndarray | Sequence[Sequence[float]],
    edge_km: float = 1.0,
) -> pd.DataFrame:
    """Aggregate a batch of geographic coordinates into discrete hexagonal spatial bins.

    Parameters
    ----------
    coords : np.ndarray or sequence of shape (N, 2)
        Array of (latitude, longitude) coordinate pairs.
    edge_km : float, default=1.0
        Length of a hexagon edge in kilometers (must be > 0).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - `q`: int axial column index
        - `r`: int axial row index
        - `count`: int number of points within cell
        - `centroid_lat`: float latitude of cell centroid
        - `centroid_lon`: float longitude of cell centroid
        Sorted by `count` in descending order.

    Raises
    ------
    ValueError
        If `edge_km <= 0` or coordinates are invalid.
    """
    if edge_km <= 0:
        raise ValueError(f"edge_km must be strictly positive; got {edge_km}")

    arr = validate_coordinates(coords)
    cols = ["q", "r", "count", "centroid_lat", "centroid_lon"]

    if len(arr) == 0:
        return pd.DataFrame(columns=cols)

    lats = arr[:, 0]
    lons = arr[:, 1]

    # Vectorized equirectangular projection
    phi = np.radians(lats)
    lam = np.radians(lons)
    x = _EARTH_RADIUS_KM * lam * np.cos(phi)
    y = _EARTH_RADIUS_KM * phi

    # Vectorized axial coordinate calculation
    r_f = (2.0 / 3.0) * (y / edge_km)
    q_f = (np.sqrt(3.0) / 3.0) * (x / edge_km) - (1.0 / 3.0) * (y / edge_km)

    # Vectorized cube rounding
    x_c = q_f
    z_c = r_f
    y_c = -x_c - z_c

    rx = np.round(x_c)
    ry = np.round(y_c)
    rz = np.round(z_c)

    dx = np.abs(rx - x_c)
    dy = np.abs(ry - y_c)
    dz = np.abs(rz - z_c)

    mask_x = (dx > dy) & (dx > dz)
    mask_y = (~mask_x) & (dy > dz)
    mask_z = (~mask_x) & (~mask_y)

    rx[mask_x] = -ry[mask_x] - rz[mask_x]
    ry[mask_y] = -rx[mask_y] - rz[mask_y]
    rz[mask_z] = -rx[mask_z] - ry[mask_z]

    q_arr = rx.astype(np.int64)
    r_arr = rz.astype(np.int64)

    df_cells = pd.DataFrame({"q": q_arr, "r": r_arr})
    binned = df_cells.groupby(["q", "r"]).size().reset_index(name="count")
    binned = binned.sort_values(by=["count", "q", "r"], ascending=[False, True, True]).reset_index(
        drop=True
    )

    # Vectorized centroids for unique bins
    q_u = binned["q"].values
    r_u = binned["r"].values

    x0 = edge_km * (np.sqrt(3.0) * q_u + 0.5 * np.sqrt(3.0) * r_u)
    y0 = edge_km * 1.5 * r_u

    phi0 = y0 / _EARTH_RADIUS_KM
    lat0 = np.clip(np.degrees(phi0), -90.0, 90.0)

    cos_phi0 = np.cos(phi0)
    safe_mask = np.abs(cos_phi0) > 1e-12
    lon0 = np.zeros_like(x0)
    lon0[safe_mask] = np.degrees(x0[safe_mask] / (_EARTH_RADIUS_KM * cos_phi0[safe_mask]))
    lon0 = ((lon0 + 180.0) % 360.0) - 180.0

    binned["centroid_lat"] = lat0
    binned["centroid_lon"] = lon0

    return binned[cols]
