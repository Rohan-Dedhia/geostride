"""Unit tests for Spherical Kernel Density Estimation (KDE) and rasterization."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geostride import BoundingBox, SphericalKDE


def test_spherical_kde_gaussian_evaluation() -> None:
    """Test Gaussian spherical KDE fit and evaluation."""
    # Cluster of 5 points near NYC center
    pts = np.array([
        [40.71, -74.00],
        [40.72, -74.00],
        [40.71, -73.99],
        [40.72, -73.99],
        [40.715, -73.995],
    ])

    kde = SphericalKDE(bandwidth_km=5.0, kernel="gaussian")
    kde.fit(pts)

    assert kde.bandwidth_km_ == 5.0
    assert "SphericalKDE" in repr(kde)

    # Center evaluation should have high density
    center_dens = kde.evaluate([[40.715, -73.995]])[0]
    assert center_dens > 0.0

    # Far away point (e.g. 100 km away) should have near-zero density
    far_dens = kde.evaluate([[41.50, -73.00]])[0]
    assert 0.0 <= far_dens < center_dens * 1e-4


def test_spherical_kde_compact_kernels() -> None:
    """Test Epanechnikov and Quartic compact support kernels."""
    pts = np.array([[0.0, 0.0]])
    h = 10.0  # 10 km bandwidth

    kde_epan = SphericalKDE(bandwidth_km=h, kernel="epanechnikov").fit(pts)
    kde_quart = SphericalKDE(bandwidth_km=h, kernel="quartic").fit(pts)

    # Point at center (distance 0 km)
    assert kde_epan.evaluate([[0.0, 0.0]])[0] > 0.0
    assert kde_quart.evaluate([[0.0, 0.0]])[0] > 0.0

    # Point within support (distance ~5.5 km)
    near_pt = [[0.05, 0.0]]  # ~5.56 km
    assert kde_epan.evaluate(near_pt)[0] > 0.0
    assert kde_quart.evaluate(near_pt)[0] > 0.0

    # Point strictly beyond compact support (distance ~22 km > 10 km)
    far_pt = [[0.2, 0.0]]  # ~22.2 km
    assert kde_epan.evaluate(far_pt)[0] == 0.0
    assert kde_quart.evaluate(far_pt)[0] == 0.0


def test_spherical_kde_automated_bandwidth() -> None:
    """Test automated Silverman and Scott bandwidth estimation rules."""
    rng = np.random.default_rng(42)
    # 50 points scattered over ~20 km
    lats = rng.normal(40.7, 0.1, size=50)
    lons = rng.normal(-74.0, 0.1, size=50)
    coords = np.column_stack([lats, lons])

    kde_silverman = SphericalKDE(bandwidth_km="silverman").fit(coords)
    assert kde_silverman.bandwidth_km_ is not None
    assert 1.0 <= kde_silverman.bandwidth_km_ <= 30.0

    kde_scott = SphericalKDE(bandwidth_km="scott").fit(coords)
    assert kde_scott.bandwidth_km_ is not None
    assert 1.0 <= kde_scott.bandwidth_km_ <= 30.0


def test_spherical_kde_2d_integration() -> None:
    """Verify that a single point's 2D density integrates to approximately 1.0."""
    pts = np.array([[0.0, 0.0]])
    h = 5.0  # 5 km

    kde = SphericalKDE(bandwidth_km=h, kernel="epanechnikov").fit(pts)

    # Numerical integration in polar coordinates: integral of f(r) * 2*pi*r dr from 0 to h
    r_vals = np.linspace(0.01, h, 500)
    dr = r_vals[1] - r_vals[0]
    # Evaluate at each radius along meridian
    # delta_lat = r / 111.19 km
    eval_lats = r_vals / 111.1949
    eval_coords = np.column_stack([eval_lats, np.zeros_like(eval_lats)])

    densities = kde.evaluate(eval_coords)
    # Area element in 2D is 2 * pi * r * dr
    total_integral = np.sum(densities * 2.0 * math.pi * r_vals * dr)

    assert math.isclose(total_integral, 1.0, rel_tol=0.03)


def test_spherical_kde_sample_weights() -> None:
    """Test effect of sample weights on density estimation."""
    pt1 = [0.0, 0.0]
    pt2 = [1.0, 1.0]
    coords = np.array([pt1, pt2])

    # Assign 90% weight to pt1 and 10% to pt2
    kde_weighted = SphericalKDE(bandwidth_km=10.0).fit(coords, weights=[0.9, 0.1])
    d1 = kde_weighted.evaluate([pt1])[0]
    d2 = kde_weighted.evaluate([pt2])[0]

    assert d1 > d2 * 5.0


def test_spherical_kde_rasterize() -> None:
    """Test 2D grid rasterization."""
    pts = np.array([[10.0, 10.0], [10.05, 10.05]])
    kde = SphericalKDE(bandwidth_km=5.0).fit(pts)

    bbox = BoundingBox(min_lat=9.9, max_lat=10.2, min_lon=9.9, max_lon=10.2)
    lats, lons, grid = kde.rasterize(bbox=bbox, resolution=(30, 40))

    assert len(lats) == 30
    assert len(lons) == 40
    assert grid.shape == (30, 40)
    assert np.all(grid >= 0.0)

    # Maximum density should be near (10.0, 10.0)
    max_idx = np.unravel_index(np.argmax(grid), grid.shape)
    assert abs(lats[max_idx[0]] - 10.0) < 0.1
    assert abs(lons[max_idx[1]] - 10.0) < 0.1

    # Automatic bbox rasterization
    lats_auto, lons_auto, grid_auto = kde.rasterize(resolution=(20, 20))
    assert len(lats_auto) == 20
    assert len(lons_auto) == 20
    assert grid_auto.shape == (20, 20)


def test_spherical_kde_validation_errors() -> None:
    """Test input validation and error raising in SphericalKDE."""
    # Invalid kernel
    with pytest.raises(ValueError, match="Invalid kernel"):
        SphericalKDE(kernel="triangular")

    # Invalid bandwidth
    with pytest.raises(ValueError, match="bandwidth_km must be strictly positive"):
        SphericalKDE(bandwidth_km=-2.0)

    with pytest.raises(ValueError, match="Invalid bandwidth rule"):
        SphericalKDE(bandwidth_km="unknown_rule")

    kde = SphericalKDE()

    # Evaluate before fit
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        kde.evaluate([[0, 0]])

    # Rasterize before fit
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        kde.rasterize()

    # Fit on empty coordinates
    with pytest.raises(ValueError, match="Cannot fit KDE on empty"):
        kde.fit(np.empty((0, 2)))

    # Weights mismatch
    with pytest.raises(ValueError, match="Weights length"):
        kde.fit([[0, 0], [1, 1]], weights=[1.0])

    # Negative weights
    with pytest.raises(ValueError, match="must be non-negative"):
        kde.fit([[0, 0], [1, 1]], weights=[-1.0, 2.0])

    # Zero sum weights
    with pytest.raises(ValueError, match="must be strictly positive"):
        kde.fit([[0, 0], [1, 1]], weights=[0.0, 0.0])

    # Invalid resolution
    kde.fit([[0, 0], [1, 1]])
    with pytest.raises(ValueError, match="resolution must be at least"):
        kde.rasterize(resolution=(1, 5))
