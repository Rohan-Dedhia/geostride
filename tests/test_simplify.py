"""Unit tests for spherical trajectory simplification (geodesic Douglas-Peucker)."""

from __future__ import annotations

import numpy as np
import pytest

from geostride.trajectory.simplify import (
    cross_track_distance_meters,
    simplify_trajectory,
)


class TestCrossTrackDistance:
    """Tests for cross_track_distance_meters."""

    def test_point_on_segment(self) -> None:
        # Segment along equator from lon 0 to lon 10
        dist = cross_track_distance_meters(
            lat=0.0,
            lon=5.0,
            start_lat=0.0,
            start_lon=0.0,
            end_lat=0.0,
            end_lon=10.0,
        )
        assert np.isclose(dist, 0.0, atol=1e-3)

    def test_perpendicular_cross_track(self) -> None:
        # Point 1 deg North of equator segment (should be ~111,195 m)
        dist = cross_track_distance_meters(
            lat=1.0,
            lon=5.0,
            start_lat=0.0,
            start_lon=0.0,
            end_lat=0.0,
            end_lon=10.0,
        )
        assert np.isclose(dist, 111195.08, atol=1.0)

    def test_along_track_clamping_beyond_endpoints(self) -> None:
        # Point beyond end (lon 15.0 on equator): distance to (0, 10) is 5 degrees (~555,975 m)
        dist_beyond = cross_track_distance_meters(
            lat=0.0,
            lon=15.0,
            start_lat=0.0,
            start_lon=0.0,
            end_lat=0.0,
            end_lon=10.0,
        )
        assert np.isclose(dist_beyond, 555975.40, atol=2.0)

        # Point behind start (lon -5.0 on equator): distance to (0, 0) is 5 degrees (~555,975 m)
        dist_behind = cross_track_distance_meters(
            lat=0.0,
            lon=-5.0,
            start_lat=0.0,
            start_lon=0.0,
            end_lat=0.0,
            end_lon=10.0,
        )
        assert np.isclose(dist_behind, 555975.40, atol=2.0)

    def test_coincident_segment_endpoints(self) -> None:
        # Zero-length segment: distance is direct point-to-point distance
        dist = cross_track_distance_meters(
            lat=1.0,
            lon=0.0,
            start_lat=0.0,
            start_lon=0.0,
            end_lat=0.0,
            end_lon=0.0,
        )
        assert np.isclose(dist, 111195.08, atol=1.0)


class TestSimplifyTrajectory:
    """Tests for simplify_trajectory."""

    def test_collinear_trajectory_reduction(self) -> None:
        # 10 collinear points along the equator
        lats = np.zeros(10)
        lons = np.linspace(0.0, 5.0, 10)
        coords = np.column_stack([lats, lons])

        simplified = simplify_trajectory(coords, tolerance_meters=10.0)
        # Should reduce to exactly 2 endpoints
        assert len(simplified) == 2
        assert np.allclose(simplified[0], [0.0, 0.0])
        assert np.allclose(simplified[-1], [0.0, 5.0])

    def test_threshold_preservation_and_pruning(self) -> None:
        # Path with 3 points: start, middle point with 500m deviation (~0.004496 deg lat), end
        dev_deg = 500.0 / 111195.08
        coords = np.array([
            [0.0, 0.0],
            [dev_deg, 2.5],
            [0.0, 5.0],
        ])

        # Large tolerance (> 500m) should prune the middle point
        simplified_loose = simplify_trajectory(coords, tolerance_meters=1000.0)
        assert len(simplified_loose) == 2

        # Strict tolerance (< 500m) should preserve the middle point
        simplified_strict = simplify_trajectory(coords, tolerance_meters=100.0)
        assert len(simplified_strict) == 3
        assert np.allclose(simplified_strict, coords)

    def test_circular_arc_simplification(self) -> None:
        # 50 points along a circular curve
        angles = np.linspace(0, np.pi, 50)
        radius_deg = 0.5  # ~55 km radius
        lats = radius_deg * np.sin(angles)
        lons = radius_deg * np.cos(angles)
        coords = np.column_stack([lats, lons])

        # Moderate tolerance (500m) preserves curve fidelity while reducing vertices
        sim_fine = simplify_trajectory(coords, tolerance_meters=500.0)
        assert 10 <= len(sim_fine) < len(coords)

        # Large tolerance (20km = 20,000m) compresses drastically
        sim_coarse = simplify_trajectory(coords, tolerance_meters=20000.0)
        assert len(sim_coarse) < len(sim_fine)
        assert len(sim_coarse) >= 2

        # Endpoints preserved
        assert np.allclose(sim_fine[0], coords[0])
        assert np.allclose(sim_fine[-1], coords[-1])
        assert np.allclose(sim_coarse[0], coords[0])
        assert np.allclose(sim_coarse[-1], coords[-1])

    def test_degenerate_and_short_trajectories(self) -> None:
        # Empty trajectory
        empty = np.empty((0, 2))
        assert len(simplify_trajectory(empty, tolerance_meters=50.0)) == 0

        # 1-point trajectory
        single = np.array([[10.0, 20.0]])
        assert len(simplify_trajectory(single, tolerance_meters=50.0)) == 1

        # 2-point trajectory
        two_pts = np.array([[10.0, 20.0], [10.5, 20.5]])
        res_two = simplify_trajectory(two_pts, tolerance_meters=50.0)
        assert len(res_two) == 2
        assert np.allclose(res_two, two_pts)

    def test_invalid_parameters(self) -> None:
        coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        with pytest.raises(ValueError, match="tolerance_meters must be strictly positive"):
            simplify_trajectory(coords, tolerance_meters=0.0)

        with pytest.raises(ValueError, match="tolerance_meters must be strictly positive"):
            simplify_trajectory(coords, tolerance_meters=-10.0)

        with pytest.raises(ValueError, match="Latitude values must be within"):
            simplify_trajectory([[95.0, 0.0], [0.0, 0.0], [1.0, 1.0]], tolerance_meters=10.0)
