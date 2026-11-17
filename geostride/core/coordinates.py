"""Coordinate data models and validation routines."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Coordinate:
    """Represents a geographic point defined by latitude and longitude in decimal degrees.

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees, must be in [-90.0, 90.0].
    longitude : float
        Longitude in decimal degrees, must be in [-180.0, 180.0].
    """

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90 degrees; got {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180 degrees; got {self.longitude}")

    @property
    def radians(self) -> tuple[float, float]:
        """Return (latitude_rad, longitude_rad)."""
        return math.radians(self.latitude), math.radians(self.longitude)

    def to_tuple(self) -> tuple[float, float]:
        """Return (latitude, longitude) as a tuple of floats."""
        return self.latitude, self.longitude

    def to_numpy(self) -> np.ndarray:
        """Return coordinates as a 1D NumPy array [lat, lon]."""
        return np.array([self.latitude, self.longitude], dtype=np.float64)

    @classmethod
    def from_sequence(cls, seq: Sequence[float]) -> Coordinate:
        """Create a Coordinate instance from a 2-element sequence [lat, lon]."""
        if len(seq) != 2:
            raise ValueError(f"Sequence must have length 2 (lat, lon); got length {len(seq)}")
        return cls(latitude=float(seq[0]), longitude=float(seq[1]))


def validate_coordinates(
    coords: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
) -> np.ndarray:
    """Validate and convert an input array-like of coordinates to a 2D float64 NumPy array.

    Parameters
    ----------
    coords : array-like of shape (N, 2) or (2,)
        Input coordinates with column 0 as latitude and column 1 as longitude.

    Returns
    -------
    np.ndarray of shape (N, 2), dtype=float64
        Validated coordinate matrix.

    Raises
    ------
    ValueError
        If coordinates cannot be reshaped to (N, 2) or values fall outside bounds.
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != 2:
            raise ValueError(f"Expected 1D coordinate array of length 2 [lat, lon]; got shape {arr.shape}")
        arr = arr.reshape(1, 2)
    elif arr.ndim == 2:
        if arr.shape[1] != 2:
            raise ValueError(f"Expected 2D coordinate array of shape (N, 2); got shape {arr.shape}")
    else:
        raise ValueError(f"Coordinates array must be 1D or 2D; got {arr.ndim} dimensions")

    lats = arr[:, 0]
    lons = arr[:, 1]

    if np.any(np.isnan(lats)) or np.any(np.isnan(lons)):
        raise ValueError("Coordinates array contains NaN values.")

    if np.any((lats < -90.0) | (lats > 90.0)):
        raise ValueError("Latitude values must be within [-90.0, 90.0].")

    if np.any((lons < -180.0) | (lons > 180.0)):
        raise ValueError("Longitude values must be within [-180.0, 180.0].")

    return arr

