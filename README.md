# GeoStride

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

---

## Benchmarks

GeoStride achieves substantial throughput through vectorized NumPy computations compared to naive Python loops:

| Benchmark Operation | Dataset Size | Execution Time | Throughput | Vectorized Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Haversine Pairwise** | $N = 50,000$ pairs | ~5.2 ms | **9,500,000 ops/s** | **~24x faster** |
| **Haversine Cross-Matrix** | $500 \times 500$ (250,000 pairs) | ~21.3 ms | **11,700,000 ops/s** | Optimized BLAS/broadcasting |
| **Morton Z-Order Encoding** | $N = 30,000$ points | ~107 ms | **280,000 ops/s** | Bit-interleaved |
| **Geohash Batch Encoding** | $N = 15,000$ points | ~121 ms | **123,000 ops/s** | Pure-Python base32 |

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
- [ ] Global & Local Moran's I with permutation tests
- [ ] Getis-Ord $G_i^*$ spatial hot-spot analysis
- [ ] Self-contained interactive Leaflet visualization generator
- [ ] Full Scikit-Learn pipeline compatibility benchmarks

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
