"""Spatial bounding box data model and operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from geostride.core.coordinates import Coordinate, validate_coordinates


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Represents a 2D geographic bounding box defined by min/max latitudes and longitudes.

    Parameters
    ----------
    min_lat : float
        Southern boundary latitude [-90.0, 90.0].
    max_lat : float
        Northern boundary latitude [-90.0, 90.0]. Must be >= min_lat.
    min_lon : float
        Western boundary longitude [-180.0, 180.0].
    max_lon : float
        Eastern boundary longitude [-180.0, 180.0]. Must be >= min_lon.
    """

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.min_lat <= 90.0) or not (-90.0 <= self.max_lat <= 90.0):
            raise ValueError(f"Latitudes must be in [-90, 90]; got ({self.min_lat}, {self.max_lat})")
        if not (-180.0 <= self.min_lon <= 180.0) or not (-180.0 <= self.max_lon <= 180.0):
            raise ValueError(f"Longitudes must be in [-180, 180]; got ({self.min_lon}, {self.max_lon})")
        if self.min_lat > self.max_lat:
            raise ValueError(f"min_lat ({self.min_lat}) cannot exceed max_lat ({self.max_lat})")
        if self.min_lon > self.max_lon:
            raise ValueError(f"min_lon ({self.min_lon}) cannot exceed max_lon ({self.max_lon})")

    @property
    def center(self) -> Coordinate:
        """Calculate the geometric centroid coordinate of the bounding box."""
        return Coordinate(
            latitude=(self.min_lat + self.max_lat) / 2.0,
            longitude=(self.min_lon + self.max_lon) / 2.0,
        )

    @property
    def height_deg(self) -> float:
        """Latitudinal span in degrees."""
        return self.max_lat - self.min_lat

    @property
    def width_deg(self) -> float:
        """Longitudinal span in degrees."""
        return self.max_lon - self.min_lon

    def contains(self, point: Coordinate | tuple[float, float] | Sequence[float]) -> bool:
        """Check whether a coordinate point lies inside this bounding box."""
        if isinstance(point, Coordinate):
            lat, lon = point.latitude, point.longitude
        else:
            lat, lon = float(point[0]), float(point[1])
        return (self.min_lat <= lat <= self.max_lat) and (self.min_lon <= lon <= self.max_lon)

    def intersects(self, other: BoundingBox) -> bool:
        """Determine whether this bounding box overlaps with another bounding box."""
        return not (
            self.max_lat < other.min_lat
            or self.min_lat > other.max_lat
            or self.max_lon < other.min_lon
            or self.min_lon > other.max_lon
        )

    def expand(self, delta_lat: float, delta_lon: float) -> BoundingBox:
        """Return an expanded bounding box padded by delta_lat and delta_lon."""
        new_min_lat = max(-90.0, self.min_lat - delta_lat)
        new_max_lat = min(90.0, self.max_lat + delta_lat)
        new_min_lon = max(-180.0, self.min_lon - delta_lon)
        new_max_lon = min(180.0, self.max_lon + delta_lon)
        return BoundingBox(
            min_lat=new_min_lat,
            max_lat=new_max_lat,
            min_lon=new_min_lon,
            max_lon=new_max_lon,
        )

    @classmethod
    def from_coordinates(cls, coords: np.ndarray | Sequence[Sequence[float]]) -> BoundingBox:
        """Construct the minimum bounding box enclosing an array-like of coordinates."""
        arr = validate_coordinates(coords)
        return cls(
            min_lat=float(np.min(arr[:, 0])),
            max_lat=float(np.max(arr[:, 0])),
            min_lon=float(np.min(arr[:, 1])),
            max_lon=float(np.max(arr[:, 1])),
        )
