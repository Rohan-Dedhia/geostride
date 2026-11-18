"""Unit tests verifying the benchmarking suite runs without errors."""

from benchmarks.run_benchmarks import (
    BenchmarkResult,
    benchmark_geohash_indexing,
    benchmark_haversine_matrix,
    benchmark_haversine_pairwise,
    benchmark_morton_indexing,
    benchmark_spatial_kfold,
    benchmark_spatial_lag,
    run_all_benchmarks,
)


def test_benchmark_result_formatting():
    res = BenchmarkResult(
        operation="Test Op",
        dataset_size=1000,
        duration_ms=10.0,
        throughput_ops_sec=100_000.0,
        speedup=5.2,
    )
    s = str(res)
    assert "Test Op" in s
    assert "100,000" in s
    assert "5.2x" in s


def test_benchmark_haversine_pairwise_smoke():
    results = benchmark_haversine_pairwise(n_samples=200)
    assert len(results) == 2
    assert results[0].duration_ms > 0
    assert results[1].duration_ms > 0
    assert results[1].speedup is not None
    assert results[1].speedup > 1.0  # Vectorized should be faster than python loop


def test_benchmark_haversine_matrix_smoke():
    res = benchmark_haversine_matrix(n=20, m=20)
    assert res.dataset_size == 400
    assert res.duration_ms > 0


def test_benchmark_geohash_smoke():
    res = benchmark_geohash_indexing(n_samples=100)
    assert res.dataset_size == 100
    assert res.duration_ms > 0


def test_benchmark_morton_smoke():
    res = benchmark_morton_indexing(n_samples=100)
    assert res.dataset_size == 100
    assert res.duration_ms > 0


def test_benchmark_spatial_lag_smoke():
    res = benchmark_spatial_lag(n_samples=50, k=3)
    assert res.dataset_size == 50
    assert res.duration_ms > 0


def test_benchmark_spatial_kfold_smoke():
    res = benchmark_spatial_kfold(n_samples=100, n_splits=3)
    assert res.dataset_size == 100
    assert res.duration_ms > 0


def test_run_all_benchmarks():
    # Verify the full aggregator returns all benchmark results
    results = run_all_benchmarks()
    assert len(results) >= 7
