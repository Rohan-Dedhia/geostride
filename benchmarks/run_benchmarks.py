"""Automated performance benchmarking suite for GeoStride operations."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from geostride.features.lags import spatial_lag_knn
from geostride.features.splitters import SpatialKFold
from geostride.indexing.geohash import encode_coordinates_geohash
from geostride.indexing.morton import encode_coordinates_morton
from geostride.metrics.haversine import haversine_matrix, haversine_pairwise


@dataclass(frozen=True)
class BenchmarkResult:
    """Summary record for a single benchmark operation."""

    operation: str
    dataset_size: int
    duration_ms: float
    throughput_ops_sec: float
    speedup: float | None = None

    def __str__(self) -> str:
        speedup_str = f"{self.speedup:.1f}x" if self.speedup is not None else "N/A"
        return (
            f"{self.operation:<35} | N={self.dataset_size:<7} | "
            f"{self.duration_ms:>8.2f} ms | {self.throughput_ops_sec:>12,.0f} ops/s | "
            f"Speedup: {speedup_str}"
        )


def _time_callable(fn: Callable[[], None], repeats: int = 3, warmups: int = 1) -> float:
    """Run a callable multiple times and return minimum execution duration in milliseconds."""
    for _ in range(warmups):
        fn()
    durations: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000.0)
    return min(durations)


def _naive_haversine_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Reference naive pure-Python loop scalar Haversine implementation."""
    r = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * r * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def benchmark_haversine_pairwise(n_samples: int = 50_000) -> list[BenchmarkResult]:
    """Benchmark vectorized haversine_pairwise against a standard Python loop."""
    rng = np.random.default_rng(42)
    lats1 = rng.uniform(-80.0, 80.0, size=n_samples)
    lons1 = rng.uniform(-170.0, 170.0, size=n_samples)
    lats2 = rng.uniform(-80.0, 80.0, size=n_samples)
    lons2 = rng.uniform(-170.0, 170.0, size=n_samples)

    # 1. Naive Python iterative loop (capped at 5,000 to keep runtimes fast)
    n_loop = min(n_samples, 5_000)
    loop_ms = _time_callable(
        lambda: [_naive_haversine_scalar(lats1[i], lons1[i], lats2[i], lons2[i]) for i in range(n_loop)],
        repeats=2,
    )
    # Extrapolate loop time to full n_samples
    est_loop_ms = loop_ms * (n_samples / n_loop)

    # 2. Vectorized NumPy Haversine
    vec_ms = _time_callable(
        lambda: haversine_pairwise(lats1, lons1, lats2, lons2, unit="km"),
        repeats=5,
    )

    speedup = est_loop_ms / vec_ms if vec_ms > 0 else 1.0

    return [
        BenchmarkResult(
            operation="Haversine Pairwise (Pure Python Loop)",
            dataset_size=n_samples,
            duration_ms=est_loop_ms,
            throughput_ops_sec=(n_samples / (est_loop_ms / 1000.0)),
            speedup=1.0,
        ),
        BenchmarkResult(
            operation="Haversine Pairwise (Vectorized NumPy)",
            dataset_size=n_samples,
            duration_ms=vec_ms,
            throughput_ops_sec=(n_samples / (vec_ms / 1000.0)),
            speedup=speedup,
        ),
    ]


def benchmark_haversine_matrix(n: int = 500, m: int = 500) -> BenchmarkResult:
    """Benchmark pairwise cross-distance matrix computation."""
    rng = np.random.default_rng(42)
    coords_a = np.column_stack([rng.uniform(-80.0, 80.0, n), rng.uniform(-170.0, 170.0, n)])
    coords_b = np.column_stack([rng.uniform(-80.0, 80.0, m), rng.uniform(-170.0, 170.0, m)])

    total_pairs = n * m
    dur_ms = _time_callable(
        lambda: haversine_matrix(coords_a, coords_b, unit="km"),
        repeats=3,
    )
    return BenchmarkResult(
        operation=f"Haversine Matrix ({n}x{m})",
        dataset_size=total_pairs,
        duration_ms=dur_ms,
        throughput_ops_sec=(total_pairs / (dur_ms / 1000.0)),
    )


def benchmark_geohash_indexing(n_samples: int = 15_000) -> BenchmarkResult:
    """Benchmark Geohash batch encoding throughput."""
    rng = np.random.default_rng(42)
    coords = np.column_stack([
        rng.uniform(-80.0, 80.0, n_samples),
        rng.uniform(-170.0, 170.0, n_samples),
    ])

    dur_ms = _time_callable(
        lambda: encode_coordinates_geohash(coords, precision=7),
        repeats=3,
    )
    return BenchmarkResult(
        operation="Geohash Batch Encode (Precision 7)",
        dataset_size=n_samples,
        duration_ms=dur_ms,
        throughput_ops_sec=(n_samples / (dur_ms / 1000.0)),
    )


def benchmark_morton_indexing(n_samples: int = 30_000) -> BenchmarkResult:
    """Benchmark Morton (Z-order) space-filling curve batch encoding throughput."""
    rng = np.random.default_rng(42)
    coords = np.column_stack([
        rng.uniform(-80.0, 80.0, n_samples),
        rng.uniform(-170.0, 170.0, n_samples),
    ])

    dur_ms = _time_callable(
        lambda: encode_coordinates_morton(coords, bits=16),
        repeats=3,
    )
    return BenchmarkResult(
        operation="Morton Z-Order Batch Encode (16-bit)",
        dataset_size=n_samples,
        duration_ms=dur_ms,
        throughput_ops_sec=(n_samples / (dur_ms / 1000.0)),
    )


def benchmark_spatial_lag(n_samples: int = 1_000, k: int = 5) -> BenchmarkResult:
    """Benchmark Inverse Distance Weighting spatial lag calculation."""
    rng = np.random.default_rng(42)
    coords = np.column_stack([
        rng.uniform(40.0, 42.0, n_samples),
        rng.uniform(-74.5, -73.5, n_samples),
    ])
    values = rng.normal(100.0, 15.0, size=n_samples)

    dur_ms = _time_callable(
        lambda: spatial_lag_knn(coords, values, k=k, method="idw"),
        repeats=3,
    )
    return BenchmarkResult(
        operation=f"Spatial Lag IDW (k={k})",
        dataset_size=n_samples,
        duration_ms=dur_ms,
        throughput_ops_sec=(n_samples / (dur_ms / 1000.0)),
    )


def benchmark_spatial_kfold(n_samples: int = 20_000, n_splits: int = 5) -> BenchmarkResult:
    """Benchmark SpatialKFold cross-validation partitioning."""
    rng = np.random.default_rng(42)
    coords = np.column_stack([
        rng.uniform(40.0, 42.0, n_samples),
        rng.uniform(-74.5, -73.5, n_samples),
    ])
    cv = SpatialKFold(n_splits=n_splits, method="morton")

    dur_ms = _time_callable(
        lambda: list(cv.split(coords)),
        repeats=3,
    )
    return BenchmarkResult(
        operation=f"SpatialKFold Partitioning (k={n_splits})",
        dataset_size=n_samples,
        duration_ms=dur_ms,
        throughput_ops_sec=(n_samples / (dur_ms / 1000.0)),
    )


def run_all_benchmarks() -> list[BenchmarkResult]:
    """Execute all benchmark suites and return aggregate results."""
    results: list[BenchmarkResult] = []
    results.extend(benchmark_haversine_pairwise())
    results.append(benchmark_haversine_matrix())
    results.append(benchmark_geohash_indexing())
    results.append(benchmark_morton_indexing())
    results.append(benchmark_spatial_lag())
    results.append(benchmark_spatial_kfold())
    return results


def main() -> None:
    """CLI entrypoint for running benchmarks and displaying formatted output."""
    print("=" * 85)
    print("GeoStride Performance Benchmarks")
    print("=" * 85)
    results = run_all_benchmarks()
    for res in results:
        print(res)
    print("=" * 85)


if __name__ == "__main__":
    main()
