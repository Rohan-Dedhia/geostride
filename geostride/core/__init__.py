"""Core abstractions and coordinate utilities for GeoStride."""

from geostride.core.bbox import BoundingBox
from geostride.core.coordinates import Coordinate, validate_coordinates

__all__ = ["BoundingBox", "Coordinate", "validate_coordinates"]
