"""Two-dimensional Morton space-filling curve (Z-order) encoder and decoder."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from geostride.core.bbox import BoundingBox
from geostride.core.coordinates import validate_coordinates

# Global default geographic envelope: lat [-90, 90], lon [-180, 180]
GLOBAL_BBOX = BoundingBox(min_lat=-90.0, max_lat=90.0, min_lon=-180.0, max_lon=180.0)


def _part1by1_32(val: int) -> int:
    """Interleave 16-bit integer with zeros to create 32-bit spaced representation."""
    val &= 0x0000FFFF
    val = (val | (val << 8)) & 0x00FF00FF
    val = (val | (val << 4)) & 0x0F0F0F0F
    val = (val | (val << 2)) & 0x33333333
    val = (val | (val << 1)) & 0x55555555
    return val


def _unpart1by1_32(val: int) -> int:
    """Extract even bits from a 32-bit interleaved integer back to a 16-bit value."""
    val &= 0x55555555
    val = (val | (val >> 1)) & 0x33333333
    val = (val | (val >> 2)) & 0x0F0F0F0F
    val = (val | (val >> 4)) & 0x00FF00FF
    val = (val | (val >> 8)) & 0x0000FFFF
    return val


def encode_morton_2d(
    latitude: float,
    longitude: float,
    bits: int = 16,
    bbox: BoundingBox | None = None,
) -> int:
    """Encode latitude and longitude into a 2D Morton (Z-order) space-filling curve integer code.

    Parameters
    ----------
    latitude : float
        Latitude coordinate in decimal degrees.
    longitude : float
        Longitude coordinate in decimal degrees.
    bits : int, default=16
        Number of bits per axis (1 to 16). The resulting Morton code occupies up to 2 * bits.
    bbox : BoundingBox, optional
        Spatial normalization envelope. Defaults to global world bounds [-90, 90], [-180, 180].

    Returns
    -------
    int
        Unsigned integer Morton code.

    Raises
    ------
    ValueError
        If bits is outside [1, 16] or coordinate falls outside the bounding box.
    """
    if not (1 <= bits <= 16):
        raise ValueError(f"Bits per axis must be between 1 and 16; got {bits}")

    box = bbox if bbox is not None else GLOBAL_BBOX
    if not box.contains((latitude, longitude)):
        raise ValueError(
            f"Coordinate ({latitude}, {longitude}) is outside the bounding box: "
            f"lat [{box.min_lat}, {box.max_lat}], lon [{box.min_lon}, {box.max_lon}]"
        )

    max_val = (1 << bits) - 1

    # Normalize to [0, 1]
    norm_lat = (latitude - box.min_lat) / box.height_deg if box.height_deg > 0 else 0.0
    norm_lon = (longitude - box.min_lon) / box.width_deg if box.width_deg > 0 else 0.0

    # Scale to integer grid [0, 2^bits - 1]
    grid_y = min(max_val, max(0, int(norm_lat * max_val)))
    grid_x = min(max_val, max(0, int(norm_lon * max_val)))

    return int((_part1by1_32(grid_y) << 1) | _part1by1_32(grid_x))


def decode_morton_2d(
    code: int,
    bits: int = 16,
    bbox: BoundingBox | None = None,
) -> tuple[float, float]:
    """Decode a 2D Morton (Z-order) integer code back into (latitude, longitude) center coordinates.

    Parameters
    ----------
    code : int
        Morton code integer.
    bits : int, default=16
        Number of bits per axis used during encoding.
    bbox : BoundingBox, optional
        Spatial normalization envelope used during encoding.

    Returns
    -------
    tuple of (latitude, longitude)
        Estimated coordinate in decimal degrees.

    Raises
    ------
    ValueError
        If bits is outside [1, 16] or code is negative.
    """
    if not (1 <= bits <= 16):
        raise ValueError(f"Bits per axis must be between 1 and 16; got {bits}")
    if code < 0:
        raise ValueError(f"Morton code must be non-negative; got {code}")

    box = bbox if bbox is not None else GLOBAL_BBOX
    max_val = (1 << bits) - 1

    grid_x = _unpart1by1_32(code)
    grid_y = _unpart1by1_32(code >> 1)

    norm_lon = (grid_x + 0.5) / (max_val + 1)
    norm_lat = (grid_y + 0.5) / (max_val + 1)

    lat = box.min_lat + norm_lat * box.height_deg
    lon = box.min_lon + norm_lon * box.width_deg

    return lat, lon


def encode_coordinates_morton(
    coords: np.ndarray | Sequence[Sequence[float]],
    bits: int = 16,
    bbox: BoundingBox | None = None,
) -> np.ndarray:
    """Vectorized batch encoding of coordinates into Morton (Z-order) codes.

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Coordinate array where column 0 is latitude and column 1 is longitude.
    bits : int, default=16
        Number of bits per axis.
    bbox : BoundingBox, optional
        Spatial normalization bounding box.

    Returns
    -------
    np.ndarray of shape (N,), dtype=np.uint64
        Array of Morton integer codes.
    """
    arr = validate_coordinates(coords)
    n = arr.shape[0]
    out = np.empty(n, dtype=np.uint64)
    for i in range(n):
        out[i] = encode_morton_2d(arr[i, 0], arr[i, 1], bits=bits, bbox=bbox)
    return out
