"""GeoStride: Lightweight geospatial feature engineering and analytics for Python."""

from geostride._version import __version__
from geostride.core.bbox import BoundingBox
from geostride.core.coordinates import Coordinate, validate_coordinates
from geostride.features.lags import SpatialLagTransformer, spatial_lag_knn
from geostride.indexing.geohash import (
    decode_geohash,
    encode_coordinates_geohash,
    encode_geohash,
    geohash_bbox,
    geohash_neighbors,
)
from geostride.indexing.morton import (
    decode_morton_2d,
    encode_coordinates_morton,
    encode_morton_2d,
)
from geostride.metrics.haversine import haversine_matrix, haversine_pairwise
from geostride.metrics.manhattan import spatial_manhattan_distance, spatial_manhattan_matrix
from geostride.metrics.vincenty import vincenty_distance, vincenty_pairwise
from geostride.stats.moran import MoranResult, global_moran
from geostride.stats.weights import (
    SpatialWeights,
    distance_band_weights,
    inverse_distance_weights,
    knn_weights,
)

__all__ = [
    "BoundingBox",
    "Coordinate",
    "MoranResult",
    "SpatialLagTransformer",
    "SpatialWeights",
    "__version__",
    "decode_geohash",
    "decode_morton_2d",
    "distance_band_weights",
    "encode_coordinates_geohash",
    "encode_coordinates_morton",
    "encode_geohash",
    "encode_morton_2d",
    "geohash_bbox",
    "geohash_neighbors",
    "global_moran",
    "haversine_matrix",
    "haversine_pairwise",
    "inverse_distance_weights",
    "knn_weights",
    "spatial_lag_knn",
    "spatial_manhattan_distance",
    "spatial_manhattan_matrix",
    "validate_coordinates",
    "vincenty_distance",
    "vincenty_pairwise",
]
