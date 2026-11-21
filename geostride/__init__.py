"""GeoStride: Lightweight geospatial feature engineering and analytics for Python."""

from geostride._version import __version__
from geostride.cluster.dbscan import SpatialDBSCANResult, spatial_dbscan
from geostride.cluster.hulls import (
    alpha_shape_concave_hull,
    convex_hull_2d,
    hull_to_geojson,
    polygon_area_km2,
)
from geostride.core.bbox import BoundingBox
from geostride.core.coordinates import Coordinate, validate_coordinates
from geostride.features.kde import SphericalKDE
from geostride.features.lags import SpatialLagTransformer, spatial_lag_knn
from geostride.features.splitters import SpatialKFold
from geostride.geometry.buffer import buffer_point, buffer_polygon
from geostride.geometry.pip import point_in_polygon, points_in_polygons
from geostride.geometry.sjoin import spatial_join
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
from geostride.indexing.rtree import SpatialRTree
from geostride.indexing.voronoi import (
    SphericalVoronoiResult,
    spherical_voronoi,
    voronoi_to_geojson,
)
from geostride.io.geojson import (
    dataframe_to_geojson,
    dump_geojson,
    geojson_to_dataframe,
    load_geojson,
)
from geostride.io.wkt import (
    dataframe_to_wkt,
    from_wkt,
    to_wkt,
    wkt_to_dataframe,
)
from geostride.metrics.haversine import haversine_matrix, haversine_pairwise
from geostride.metrics.manhattan import spatial_manhattan_distance, spatial_manhattan_matrix
from geostride.metrics.vincenty import vincenty_distance, vincenty_pairwise
from geostride.stats.getis_ord import GetisOrdResult, getis_ord_local
from geostride.stats.lisa import LISAResult, local_moran
from geostride.stats.moran import MoranResult, global_moran
from geostride.stats.weights import (
    SpatialWeights,
    distance_band_weights,
    inverse_distance_weights,
    knn_weights,
    standardize_weights,
)
from geostride.trajectory.metrics import (
    TrajectoryMetrics,
    resample_trajectory,
    trajectory_kinematics,
)
from geostride.trajectory.simplify import (
    cross_track_distance_meters,
    simplify_trajectory,
)
from geostride.viz.map_export import export_html_map, render_html_map

__all__ = [
    "BoundingBox",
    "Coordinate",
    "GetisOrdResult",
    "LISAResult",
    "MoranResult",
    "SpatialDBSCANResult",
    "SpatialKFold",
    "SpatialLagTransformer",
    "SpatialRTree",
    "SpatialWeights",
    "SphericalKDE",
    "SphericalVoronoiResult",
    "TrajectoryMetrics",
    "__version__",
    "alpha_shape_concave_hull",
    "bin_coordinates_hex",
    "buffer_point",
    "buffer_polygon",
    "convex_hull_2d",
    "coord_to_hex_cell",
    "cross_track_distance_meters",
    "dataframe_to_geojson",
    "dataframe_to_wkt",
    "decode_geohash",
    "decode_morton_2d",
    "distance_band_weights",
    "dump_geojson",
    "encode_coordinates_geohash",
    "encode_coordinates_morton",
    "encode_geohash",
    "encode_morton_2d",
    "export_html_map",
    "from_wkt",
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
    "hull_to_geojson",
    "inverse_distance_weights",
    "knn_weights",
    "load_geojson",
    "local_moran",
    "point_in_polygon",
    "points_in_polygons",
    "polygon_area_km2",
    "render_html_map",
    "resample_trajectory",
    "simplify_trajectory",
    "spatial_dbscan",
    "spatial_join",
    "spatial_lag_knn",
    "spatial_manhattan_distance",
    "spatial_manhattan_matrix",
    "spherical_voronoi",
    "standardize_weights",
    "to_wkt",
    "trajectory_kinematics",
    "validate_coordinates",
    "vincenty_distance",
    "vincenty_pairwise",
    "voronoi_to_geojson",
    "wkt_to_dataframe",
]
