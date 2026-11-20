# GeoStride

[![CI](https://github.com/Rohan-Dedhia/geostride/actions/workflows/ci.yml/badge.svg)](https://github.com/Rohan-Dedhia/geostride/actions/workflows/ci.yml)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://github.com/Rohan-Dedhia/geostride)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**GeoStride** is a lightweight, pure-Python / NumPy & Pandas library for geospatial feature engineering, spatial autocorrelation, spatial clustering, and discrete spatial indexing.

Comprehensive geospatial stacks (such as `geopandas`, `shapely`, and `pyproj`) frequently depend on external compiled C/C++ libraries (`GEOS`, `PROJ`, `GDAL`), which can present installation and dependency challenges in resource-constrained environments (e.g., lightweight serverless functions, minimal Docker containers, or edge deployments). **GeoStride** provides a focused, pure-Python suite of vectorized geospatial utilities built directly on `numpy`, `pandas`, and `scipy`.

---

## Key Features

- **Geodetic & Planar Metrics:**
  - Vectorized Haversine distance matrix computation.
  - High-precision Vincenty inverse ellipsoidal geodesic formula.
  - Manhattan spatial grid distance, cross-track, and along-track calculations.
- **Discrete Spatial Indexing & Hashing:**
  - Fast bitwise Geohash encoding/decoding and neighbor lookups.
  - Morton (Z-order) space-filling curve indexer.
  - Discrete hexagonal and rectangular spatial grid binning.
- **Spatial Feature Engineering:**
  - Spatial lag calculation using K-Nearest Neighbors ($k$-NN) and Inverse Distance Weighting (IDW).
  - Point-of-Interest (POI) distance surfaces and radius density count estimators.
  - Scikit-Learn compatible transformer pipeline (`SpatialLagTransformer`).
- **Spatial Autocorrelation & Statistics:**
  - Spatial weight matrix generators ($k$-NN, distance-band, contiguous).
  - Global Moran's $I$ and Local Moran's $I$ (LISA) with Monte Carlo permutation inference.
  - Getis-Ord $G_i$ and $G_i^*$ hot-spot and cold-spot statistics.
- **Spatial Clustering & Outliers:**
  - BallTree-accelerated spatial DBSCAN with geographic radian conversion.
  - Spatial outlier diagnostics and cluster boundary summarizers.
- **I/O & Standalone Visualization:**
  - Lightweight GeoJSON serialization/deserialization for Pandas DataFrames.
  - Self-contained interactive Leaflet HTML map exports without requiring a server.

---

## Installation

```bash
pip install geostride
```

For development and testing:

```bash
git clone https://github.com/Rohan-Dedhia/geostride.git
cd geostride
pip install -e ".[dev]"
```

---

## Quick Tour

### 1. Vectorized Distance Computation

```python
import numpy as np
from geostride.metrics import haversine_pairwise, haversine_matrix

# New York to London & Tokyo coordinates (lat, lon)
coords_a = np.array([[40.7128, -74.0060]])
coords_b = np.array([
    [51.5074, -0.1278],   # London
    [35.6762, 139.6503]   # Tokyo
])

# Compute distance matrix in kilometers
distances = haversine_matrix(coords_a, coords_b, unit="km")
print(distances)
```

### 2. Spatial Lag Feature Engineering

```python
import pandas as pd
from geostride.features import SpatialLagTransformer

df = pd.DataFrame({
    "latitude": [40.71, 40.72, 40.75, 40.78],
    "longitude": [-74.00, -73.99, -73.98, -73.96],
    "house_price": [1200000, 1350000, 980000, 1100000]
})

transformer = SpatialLagTransformer(
    lat_col="latitude",
    lon_col="longitude",
    value_col="house_price",
    method="idw",
    k_neighbors=2
)
df["price_spatial_lag"] = transformer.fit_transform(df)
```

### 3. Discrete Spatial Indexing (Geohash)

```python
from geostride.indexing import encode_geohash, geohash_neighbors, geohash_bbox

# Encode coordinate into a 7-character Geohash (~150m cell)
gh = encode_geohash(latitude=40.7128, longitude=-74.0060, precision=7)
print(f"Geohash: {gh}")

# Retrieve all 8 adjacent neighbor cells
neighbors = geohash_neighbors(gh)
print(f"North neighbor: {neighbors['n']}")
```

### 4. Spatial Autocorrelation (Moran's I)

```python
from geostride.stats import global_moran

# Check whether values cluster spatially
res = global_moran(coords=df[["latitude", "longitude"]].values, values=df["house_price"].values, k_neighbors=3)
print(f"Moran's I: {res.moran_i:.3f}, p-value: {res.p_value:.4f}")
```

### 5. Spatial DBSCAN Clustering

```python
from geostride.cluster import spatial_dbscan

# Cluster coordinates within 2.0 km with at least 5 points
clusters = spatial_dbscan(df[["latitude", "longitude"]].values, eps_km=2.0, min_samples=5)
print(f"Clusters discovered: {clusters.n_clusters}")
print(f"Noise ratio: {clusters.noise_ratio:.1%}")
```

### 6. Zero-Dependency GeoJSON Serialization

```python
from geostride.io import dataframe_to_geojson, geojson_to_dataframe

# Convert Pandas DataFrame directly to RFC 7946 FeatureCollection
geojson_data = dataframe_to_geojson(df, lat_col="latitude", lon_col="longitude")

# Parse GeoJSON back into a flat DataFrame
recovered_df = geojson_to_dataframe(geojson_data)
```

### 7. Interactive Leaflet HTML Map Export

```python
from geostride.viz import export_html_map

# Export interactive Leaflet HTML map viewable in any browser without a web server
export_html_map(
    "nyc_clusters.html",
    coords=df[["latitude", "longitude"]].values,
    labels=df["cluster_id"].values,
    tooltips=[f"Price: ${p:,.0f}" for p in df["house_price"]],
    title="NYC Housing Spatial Clusters"
)
```

### 8. Discrete Hexagonal Binning (H3-Style)

```python
from geostride.indexing import bin_coordinates_hex, coord_to_hex_cell

# Map individual point to axial hexagonal grid cell (q, r)
q, r = coord_to_hex_cell(40.7128, -74.0060, edge_km=2.0)

# Aggregate thousands of coordinates into discrete equal-area hexagonal bins
hex_bins = bin_coordinates_hex(df[["latitude", "longitude"]].values, edge_km=2.0)
print(hex_bins.head())
```

### 9. Spatial Cross-Validation (Scikit-Learn Integration)

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from geostride.features import SpatialKFold

# Prevent optimistic spatial autocorrelation leakage across folds
cv = SpatialKFold(n_splits=5, method="morton", lat_col="latitude", lon_col="longitude")
scores = cross_val_score(Ridge(), X=df[["latitude", "longitude", "feature1"]], y=df["price"], cv=cv, scoring="r2")
print(f"Spatial CV R^2: {scores.mean():.3f} +/- {scores.std():.3f}")
```

### 10. Cluster Boundary Envelopes (Convex & Concave Hulls)

```python
from geostride.cluster import alpha_shape_concave_hull, convex_hull_2d, polygon_area_km2

# Compute minimum convex bounding polygon
convex_hull = convex_hull_2d(df[["latitude", "longitude"]].values)
print(f"Convex Area: {polygon_area_km2(convex_hull):.1f} km^2")

# Extract tight non-convex alpha-shape boundary around irregular urban clusters
concave_rings = alpha_shape_concave_hull(df[["latitude", "longitude"]].values, alpha=0.5)
print(f"Concave Area: {polygon_area_km2(concave_rings[0]):.1f} km^2")
```

### 11. Spherical Voronoi Tessellation

```python
from geostride.indexing import spherical_voronoi, voronoi_to_geojson

# Partition the Earth into exact spherical Voronoi service territories
voronoi_res = spherical_voronoi(df[["latitude", "longitude"]].values)
print(f"Total partitioned area: {voronoi_res.total_area_km2:,.0f} km^2")

# Export Voronoi polygons directly to GeoJSON
geojson_voronoi = voronoi_to_geojson(voronoi_res)
```

### 12. Vectorized Point-in-Polygon (Ray Casting)

```python
from geostride.geometry import point_in_polygon, points_in_polygons

# Define administrative district polygon [latitude, longitude]
district_boundary = [
    [40.70, -74.02],
    [40.70, -73.97],
    [40.76, -73.97],
    [40.76, -74.02],
]

# High-throughput containment test across thousands of points
coords = df[["latitude", "longitude"]].values
in_district_mask = point_in_polygon(coords, district_boundary)
print(f"Points inside district: {in_district_mask.sum()} / {len(coords)}")

# Multi-polygon spatial indexing with bounding-box pre-filtering
district_ids = points_in_polygons(coords, [district_boundary])
```

### 13. Pure-Python Spatial R-Tree Indexing

```python
from geostride.indexing import SpatialRTree

# Initialize hierarchical R-Tree indexer (zero external C dependencies)
rtree = SpatialRTree(max_entries=16, min_entries=4)

# Bulk load spatial features via Sort-Tile-Recursive (STR) packing
feature_ids = list(df.index)
bboxes = [
    (lat - 0.01, lat + 0.01, lon - 0.01, lon + 0.01)
    for lat, lon in df[["latitude", "longitude"]].values
]
rtree.bulk_load(feature_ids, bboxes)

# Fast spatial range intersection query
nearby_ids = rtree.query_range((40.71, 40.75, -74.01, -73.97))

# Best-first branch-and-bound k-nearest neighbor retrieval
nearest_3 = rtree.query_nearest(latitude=40.7128, longitude=-74.0060, k=3)
```

### 14. Local Indicators of Spatial Association (LISA)

```python
from geostride.stats import local_moran

# Decompose spatial autocorrelation into local hot spots and spatial outliers
lisa_res = local_moran(
    coords=df[["latitude", "longitude"]].values,
    values=df["house_price"].values,
    k_neighbors=5,
    n_permutations=999,
    p_threshold=0.05
)

print(f"High-High Hot Spots: {len(lisa_res.high_high)}")
print(f"Low-Low Cold Spots: {len(lisa_res.low_low)}")
print(f"High-Low Outliers: {len(lisa_res.high_low)}")
print(f"Low-High Outliers: {len(lisa_res.low_high)}")
```

### 15. Geodesic Polygon Buffering & Boundary Offsets

```python
from geostride.geometry import buffer_point, buffer_polygon

# Generate exact geodesic circular buffer (e.g., 5 km emergency radius)
circle_buffer = buffer_point(latitude=40.7128, longitude=-74.0060, distance_km=5.0, n_points=32)

# Dilate delivery zone polygon outward by 2.0 km with smooth rounded corner joins
zone = [[40.70, -74.00], [40.75, -74.00], [40.75, -73.95], [40.70, -73.95]]
dilated_zone = buffer_polygon(zone, distance_km=2.0, join_style="round")

# Inward erosion by 1.0 km (contracting hazard perimeter)
eroded_zone = buffer_polygon(zone, distance_km=-1.0)
```

### 16. Vectorized Spatial Join (sjoin)

```python
from geostride.geometry import spatial_join

# Point-in-polygon spatial join matching customer coordinates to delivery zones
joined_df = spatial_join(
    left_df=df,
    polygons=[dilated_zone],
    how="inner",
    predicate="within",
    poly_id_col="zone_id",
)

# Nearest neighbor spatial join computing distance to closest facility
nearest_df = spatial_join(
    left_df=df,
    polygons={"HQ": dilated_zone},
    how="left",
    predicate="nearest",
    distance_col="dist_to_hq_km",
)
```

### 17. Spherical Kernel Density Estimation (KDE)

```python
from geostride.features import SphericalKDE

# Fit continuous spatial intensity surface with automated Silverman bandwidth
kde = SphericalKDE(bandwidth_km="silverman", kernel="gaussian")
kde.fit(df[["latitude", "longitude"]].values)
print(f"Optimal bandwidth: {kde.bandwidth_km_:.2f} km")

# Evaluate continuous probability density at specific coordinates
densities = kde.evaluate([[40.7128, -74.0060], [40.7580, -73.9855]])

# Rasterize density surface into a 2D regular grid for heatmap visualization
lats, lons, density_grid = kde.rasterize(resolution=(100, 100))
```

### 18. Well-Known Text (WKT) Serialization & DataFrame Parsing

```python
from geostride.io import dataframe_to_wkt, from_wkt, to_wkt, wkt_to_dataframe

# Convert coordinate array or polygon rings into standard OGC WKT strings
point_wkt = to_wkt([40.7128, -74.0060])  # "POINT (-74.006 40.7128)"
line_wkt = to_wkt([[40.71, -74.00], [34.05, -118.24]])  # "LINESTRING (-74 40.71, -118.24 34.05)"

# Parse WKT strings back into geometry type and GeoStride coordinate matrices
geom_type, rings = from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")

# Seamless integration with Pandas DataFrames
wkt_df = dataframe_to_wkt(df, lat_col="latitude", lon_col="longitude", wkt_col="wkt")
restored_df = wkt_to_dataframe(wkt_df, wkt_col="wkt")
```

### 19. Spatio-Temporal Trajectory Analytics & Resampling

```python
from geostride.trajectory import resample_trajectory, trajectory_kinematics

# Analyze GPS trajectory kinematics: speed, forward heading, turning angles, and acceleration
metrics = trajectory_kinematics(
    coords=[[40.7128, -74.0060], [40.7580, -73.9855], [40.7829, -73.9654]],
    timestamps=["2026-01-01 10:00:00", "2026-01-01 10:05:00", "2026-01-01 10:12:00"],
)

print(f"Total distance: {metrics.total_distance_km:.2f} km")
print(f"Mean speed: {metrics.mean_speed_kmh:.1f} km/h, Max: {metrics.max_speed_kmh:.1f} km/h")
print(f"Segment headings: {metrics.headings_deg}")
print(f"Turning angles: {metrics.turning_angles_deg}")

# Resample trajectory to uniform 30-second temporal steps via Spherical Slerp
resampled_coords, resampled_ts = resample_trajectory(
    coords=[[40.7128, -74.0060], [40.7829, -73.9654]],
    timestamps=[0.0, 300.0],
    step_seconds=30.0,
)
```

---

## Benchmarks

GeoStride achieves substantial throughput through vectorized NumPy computations compared to naive Python loops:

| Benchmark Operation | Dataset Size | Execution Time | Throughput | Vectorized Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Haversine Pairwise** | $N = 50,000$ pairs | ~5.2 ms | **9,500,000 ops/s** | **~24x faster** |
| **Haversine Cross-Matrix** | $500 \times 500$ (250,000 pairs) | ~21.3 ms | **11,700,000 ops/s** | Optimized BLAS/broadcasting |
| **Morton Z-Order Encoding** | $N = 30,000$ points | ~107 ms | **280,000 ops/s** | Bit-interleaved |
| **Geohash Batch Encoding** | $N = 15,000$ points | ~121 ms | **123,000 ops/s** | Pure-Python base32 |
| **SpatialKFold Partitioning** | $N = 20,000$ points | ~54 ms | **370,000 ops/s** | Morton-ordered blocks |

Run the automated benchmark suite:
```bash
python -m benchmarks.run_benchmarks
```

---

## Architecture

```
geostride/
├── core/         # Coordinate abstractions, validation & bounding boxes
├── metrics/      # Geodesic, Haversine, Vincenty & Manhattan metrics
├── indexing/     # Geohash, Morton curve, spatial grids
├── features/     # Spatial lags, POI density & scikit-learn transformers
├── stats/        # Moran's I, Getis-Ord Gi*, spatial weights
├── cluster/      # Spatial DBSCAN & boundary hulls
├── geometry/     # Point-in-polygon & spatial boundary operations
├── trajectory/   # Trajectory kinematics & spherical resampling
├── io/           # GeoJSON/WKT parsers
└── viz/          # Interactive map exporters
```

---

## Roadmap

- [x] Core geodetic distance algorithms (Haversine, Vincenty, Manhattan)
- [x] Bitwise Geohash indexing and 8-direction neighbor search
- [x] Morton space-filling curve (Z-order) indexing
- [x] Inverse Distance Weighting (IDW) & KNN Spatial Lag Transformers
- [x] Automated performance comparison benchmarking suite
- [x] Spatial weight matrix generators (KNN, distance band, inverse distance)
- [x] Global & Local Moran's I with permutation tests
- [x] Getis-Ord $G_i^*$ spatial hot-spot analysis
- [x] BallTree-accelerated spatial DBSCAN clustering
- [x] Zero-dependency GeoJSON and WKT parser & serializer
- [x] Self-contained interactive Leaflet visualization generator
- [x] Discrete hexagonal (H3-style) spatial tessellation generator
- [x] Full Scikit-Learn pipeline compatibility benchmarks
- [x] Convex and concave alpha-shape cluster boundary hull generator
- [x] GitHub Actions multi-OS test matrix and automated code quality verification
- [x] Spherical Voronoi diagram generator (`geostride.indexing.voronoi`)
- [x] Vectorized Point-in-Polygon indexer (`geostride.geometry.pip`)
- [x] Spatial R-Tree nearest neighbor & range indexer (`geostride.indexing.rtree`)
- [x] Local Indicators of Spatial Association (LISA) quadrant mapping (`geostride.stats.lisa`)
- [x] Geodesic polygon buffering and dilation (`geostride.geometry.buffer`)
- [x] Vectorized spatial join (`sjoin`) for polygons and point proximity (`geostride.geometry.sjoin`)
- [x] Spherical Kernel Density Estimation (KDE) and heatmap rasterizer (`geostride.features.kde`)
- [x] Well-Known Text (WKT) parser and serializer (`geostride.io.wkt`)
- [x] Spatio-temporal trajectory analytics and kinematics (`geostride.trajectory.metrics`)
- [ ] Spherical trajectory simplification using geodesic Douglas-Peucker (`geostride.trajectory.simplify`)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
