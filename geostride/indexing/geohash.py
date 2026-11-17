"""Pure-Python high-performance bitwise Geohash encoding, decoding, and neighbor search."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from geostride.core.bbox import BoundingBox
from geostride.core.coordinates import validate_coordinates

BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
BASE32_MAP = {c: i for i, c in enumerate(BASE32)}

# Precision error bounds in kilometers (approximate equatorial cell dimensions)
GEOHASH_PRECISION_KM = {
    1: (5000.0, 5000.0),
    2: (1250.0, 625.0),
    3: (156.0, 156.0),
    4: (39.1, 19.5),
    5: (4.89, 4.89),
    6: (1.22, 0.61),
    7: (0.153, 0.153),
    8: (0.0382, 0.0191),
    9: (0.00477, 0.00477),
    10: (0.00119, 0.000596),
    11: (0.000149, 0.000149),
    12: (0.0000372, 0.0000186),
}


def encode_geohash(latitude: float, longitude: float, precision: int = 9) -> str:
    """Encode latitude and longitude into a standard base32 Geohash string.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees [-90.0, 90.0].
    longitude : float
        Longitude in decimal degrees [-180.0, 180.0].
    precision : int, default=9
        Number of characters in the generated Geohash (1 to 12).

    Returns
    -------
    str
        Base32 encoded Geohash string of length `precision`.

    Raises
    ------
    ValueError
        If coordinates are out of bounds or precision is outside [1, 12].
    """
    if not (1 <= precision <= 12):
        raise ValueError(f"Precision must be between 1 and 12 characters; got {precision}")
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError(f"Latitude must be within [-90.0, 90.0]; got {latitude}")
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError(f"Longitude must be within [-180.0, 180.0]; got {longitude}")

    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]

    geohash: list[str] = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    is_even = True

    while len(geohash) < precision:
        if is_even:
            mid = (lon_interval[0] + lon_interval[1]) / 2.0
            if longitude >= mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2.0
            if latitude >= mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid

        is_even = not is_even
        if bit < 4:
            bit += 1
        else:
            geohash.append(BASE32[ch])
            bit = 0
            ch = 0

    return "".join(geohash)


def decode_geohash(geohash: str) -> tuple[float, float, float, float]:
    """Decode a Geohash string into its center coordinate and lat/lon error margins.

    Parameters
    ----------
    geohash : str
        Base32 encoded Geohash string.

    Returns
    -------
    tuple of (latitude, longitude, lat_error, lon_error)
        Center point coordinates and precision error radius.

    Raises
    ------
    ValueError
        If the geohash contains invalid base32 characters.
    """
    if not geohash:
        raise ValueError("Cannot decode empty geohash string.")

    cleaned = geohash.lower().strip()
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    is_even = True

    for c in cleaned:
        if c not in BASE32_MAP:
            raise ValueError(f"Invalid character '{c}' in geohash '{geohash}'")
        cd = BASE32_MAP[c]
        for mask in [16, 8, 4, 2, 1]:
            if is_even:
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if cd & mask:
                    lon_interval[0] = mid
                else:
                    lon_interval[1] = mid
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if cd & mask:
                    lat_interval[0] = mid
                else:
                    lat_interval[1] = mid
            is_even = not is_even

    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0
    lat_err = (lat_interval[1] - lat_interval[0]) / 2.0
    lon_err = (lon_interval[1] - lon_interval[0]) / 2.0

    return lat, lon, lat_err, lon_err


def geohash_bbox(geohash: str) -> BoundingBox:
    """Return the BoundingBox envelope representing the spatial extent of a Geohash cell."""
    lat, lon, lat_err, lon_err = decode_geohash(geohash)
    return BoundingBox(
        min_lat=lat - lat_err,
        max_lat=lat + lat_err,
        min_lon=lon - lon_err,
        max_lon=lon + lon_err,
    )


def geohash_neighbors(geohash: str) -> dict[str, str]:
    """Calculate the 8 adjacent neighboring Geohash cells around a given Geohash.

    Parameters
    ----------
    geohash : str
        Target Geohash cell.

    Returns
    -------
    dict of str to str
        Mapping containing directional keys: 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'.
    """
    lat, lon, lat_err, lon_err = decode_geohash(geohash)
    precision = len(geohash)

    # Step distances in degrees
    d_lat = lat_err * 2.0
    d_lon = lon_err * 2.0

    def _safe_encode(adj_lat: float, adj_lon: float) -> str:
        # Wrap longitude across -180/180 antimeridian
        if adj_lon > 180.0:
            adj_lon -= 360.0
        elif adj_lon < -180.0:
            adj_lon += 360.0
        # Clamp latitude to poles
        adj_lat = max(-90.0, min(90.0, adj_lat))
        return encode_geohash(adj_lat, adj_lon, precision=precision)

    return {
        "n": _safe_encode(lat + d_lat, lon),
        "s": _safe_encode(lat - d_lat, lon),
        "e": _safe_encode(lat, lon + d_lon),
        "w": _safe_encode(lat, lon - d_lon),
        "ne": _safe_encode(lat + d_lat, lon + d_lon),
        "nw": _safe_encode(lat + d_lat, lon - d_lon),
        "se": _safe_encode(lat - d_lat, lon + d_lon),
        "sw": _safe_encode(lat - d_lat, lon - d_lon),
    }


def encode_coordinates_geohash(
    coords: np.ndarray | Sequence[Sequence[float]],
    precision: int = 7,
) -> np.ndarray:
    """Vectorized batch encoding of coordinates to Geohash strings for Pandas/NumPy pipelines.

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Input coordinate array where column 0 is latitude and column 1 is longitude.
    precision : int, default=7
        Geohash string character length.

    Returns
    -------
    np.ndarray of shape (N,), dtype=object
        Array of Geohash string codes.
    """
    arr = validate_coordinates(coords)
    n = arr.shape[0]
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = encode_geohash(arr[i, 0], arr[i, 1], precision=precision)
    return out
