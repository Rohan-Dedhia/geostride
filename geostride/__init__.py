"""GeoStride: Lightweight geospatial feature engineering and analytics for Python."""

from geostride._version import __version__
from geostride.core.bbox import BoundingBox
from geostride.core.coordinates import Coordinate, validate_coordinates
from geostride.indexing.geohash import (
    decode_geohash,
    encode_coordinates_geohash,
    encode_geohash,
    geohash_bbox,
    geohash_neighbors,
)
from geostride.metrics.haversine import haversine_matrix, haversine_pairwise
from geostride.metrics.manhattan import spatial_manhattan_distance, spatial_manhattan_matrix
from geostride.metrics.vincenty import vincenty_distance, vincenty_pairwise

__all__ = [
    "BoundingBox",
    "Coordinate",
    "__version__",
    "decode_geohash",
    "encode_coordinates_geohash",
    "encode_geohash",
    "geohash_bbox",
    "geohash_neighbors",
    "haversine_matrix",
    "haversine_pairwise",
    "spatial_manhattan_distance",
    "spatial_manhattan_matrix",
    "validate_coordinates",
    "vincenty_distance",
    "vincenty_pairwise",
]
