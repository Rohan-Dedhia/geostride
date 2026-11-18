"""GeoStride: Lightweight geospatial feature engineering and analytics for Python."""

from geostride._version import __version__
from geostride.cluster.dbscan import SpatialDBSCANResult, spatial_dbscan
from geostride.core.bbox import BoundingBox
from geostride.core.coordinates import Coordinate, validate_coordinates
from geostride.features.lags import SpatialLagTransformer, spatial_lag_knn
from geostride.features.splitters import SpatialKFold
from geostride.indexing.geohash import (
    decode_geohash,
    encode_coordinates_geohash,
    encode_geohash,
    geohash_bbox,
    geohash_neighbors,
)
from geostride.indexing.hexagon import (
    bin_coordinates_hex,
    coord_to_hex_cell,
    hex_cell_to_centroid,
    hex_cell_to_polygon,
    hex_neighbors,
)
from geostride.indexing.morton import (
    decode_morton_2d,
    encode_coordinates_morton,
    encode_morton_2d,
)
from geostride.io.geojson import (
    dataframe_to_geojson,
    dump_geojson,
    geojson_to_dataframe,
    load_geojson,
)
from geostride.metrics.haversine import haversine_matrix, haversine_pairwise
from geostride.metrics.manhattan import spatial_manhattan_distance, spatial_manhattan_matrix
from geostride.metrics.vincenty import vincenty_distance, vincenty_pairwise
from geostride.stats.getis_ord import GetisOrdResult, getis_ord_local
from geostride.stats.moran import MoranResult, global_moran
from geostride.stats.weights import (
    SpatialWeights,
    distance_band_weights,
    inverse_distance_weights,
    knn_weights,
)
from geostride.viz.map_export import export_html_map, render_html_map

__all__ = [
    "BoundingBox",
    "Coordinate",
    "GetisOrdResult",
    "MoranResult",
    "SpatialDBSCANResult",
    "SpatialKFold",
    "SpatialLagTransformer",
    "SpatialWeights",
    "__version__",
    "bin_coordinates_hex",
    "coord_to_hex_cell",
    "dataframe_to_geojson",
    "decode_geohash",
    "decode_morton_2d",
    "distance_band_weights",
    "dump_geojson",
    "encode_coordinates_geohash",
    "encode_coordinates_morton",
    "encode_geohash",
    "encode_morton_2d",
    "export_html_map",
    "geohash_bbox",
    "geohash_neighbors",
    "geojson_to_dataframe",
    "getis_ord_local",
    "global_moran",
    "haversine_matrix",
    "haversine_pairwise",
    "hex_cell_to_centroid",
    "hex_cell_to_polygon",
    "hex_neighbors",
    "inverse_distance_weights",
    "knn_weights",
    "load_geojson",
    "render_html_map",
    "spatial_dbscan",
    "spatial_lag_knn",
    "spatial_manhattan_distance",
    "spatial_manhattan_matrix",
    "validate_coordinates",
    "vincenty_distance",
    "vincenty_pairwise",
]
