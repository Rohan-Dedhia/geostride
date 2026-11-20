"""Unit tests for spatio-temporal trajectory kinematics and resampling."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from geostride.trajectory.metrics import (
    TrajectoryMetrics,
    resample_trajectory,
    trajectory_kinematics,
)


class TestTrajectoryKinematics:
    """Tests for trajectory_kinematics."""

    def test_northbound_constant_speed(self) -> None:
        # 3 points moving directly North along prime meridian at 1-degree steps (~111.19 km)
        coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ])
        # 1 hour (3600s) per step
        timestamps = np.array([0.0, 3600.0, 7200.0])

        metrics = trajectory_kinematics(coords, timestamps)
        assert isinstance(metrics, TrajectoryMetrics)
        assert len(metrics.distances_km) == 2
        assert len(metrics.durations_sec) == 2
        assert len(metrics.speeds_kmh) == 2
        assert len(metrics.headings_deg) == 2
        assert len(metrics.turning_angles_deg) == 1
        assert len(metrics.accelerations_m_s2) == 1
        assert len(metrics.cumulative_distance_km) == 3

        # Heading should be ~0.0 degrees (North)
        assert np.allclose(metrics.headings_deg, [0.0, 0.0], atol=1e-3)

        # Turning angle should be 0.0
        assert np.isclose(metrics.turning_angles_deg[0], 0.0, atol=1e-3)

        # Constant speed -> acceleration ~0.0
        assert np.isclose(metrics.accelerations_m_s2[0], 0.0, atol=1e-5)

        # Speeds ~111 km/h
        assert 110.0 < metrics.speeds_kmh[0] < 112.0
        assert np.isclose(metrics.speeds_kmh[0], metrics.speeds_kmh[1], atol=1e-3)
        assert np.isclose(metrics.mean_speed_kmh, metrics.speeds_kmh[0], atol=1e-3)
        assert np.isclose(metrics.max_speed_kmh, metrics.speeds_kmh[0], atol=1e-3)

        # Cumulative distance
        assert metrics.cumulative_distance_km[0] == 0.0
        assert np.isclose(metrics.cumulative_distance_km[-1], metrics.total_distance_km)

    def test_cardinal_headings(self) -> None:
        # North
        m_north = trajectory_kinematics([[0.0, 0.0], [1.0, 0.0]], [0, 100])
        assert np.isclose(m_north.headings_deg[0], 0.0, atol=1e-3)

        # East
        m_east = trajectory_kinematics([[0.0, 0.0], [0.0, 1.0]], [0, 100])
        assert np.isclose(m_east.headings_deg[0], 90.0, atol=1e-3)

        # South
        m_south = trajectory_kinematics([[1.0, 0.0], [0.0, 0.0]], [0, 100])
        assert np.isclose(m_south.headings_deg[0], 180.0, atol=1e-3)

        # West
        m_west = trajectory_kinematics([[0.0, 1.0], [0.0, 0.0]], [0, 100])
        assert np.isclose(m_west.headings_deg[0], 270.0, atol=1e-3)

    def test_turning_angles_right_and_left(self) -> None:
        # Moving North then East: right turn (+90 degrees)
        coords_right = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])
        m_right = trajectory_kinematics(coords_right, [0, 60, 120])
        assert np.isclose(m_right.turning_angles_deg[0], 90.0, atol=1e-2)

        # Moving North then West: left turn (-90 degrees)
        coords_left = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, -1.0],
        ])
        m_left = trajectory_kinematics(coords_left, [0, 60, 120])
        assert np.isclose(m_left.turning_angles_deg[0], -90.0, atol=1e-2)

    def test_accelerating_trajectory(self) -> None:
        # Moving 1 km in 100s (v1 = 10 m/s), then 2 km in 100s (v2 = 20 m/s)
        # Coordinates along equator
        lat = 0.0
        # 1 deg lon at equator is ~111.32 km
        km_per_deg = 111.32
        coords = np.array([
            [lat, 0.0],
            [lat, 1.0 / km_per_deg],
            [lat, 3.0 / km_per_deg],
        ])
        timestamps = [0.0, 100.0, 200.0]
        m = trajectory_kinematics(coords, timestamps)
        assert m.speeds_kmh[1] > m.speeds_kmh[0]
        # Positive acceleration
        assert m.accelerations_m_s2[0] > 0.0

    def test_two_point_trajectory(self) -> None:
        coords = np.array([[10.0, 20.0], [10.5, 20.5]])
        m = trajectory_kinematics(coords, [1000, 2000])
        assert len(m.distances_km) == 1
        assert len(m.speeds_kmh) == 1
        assert len(m.headings_deg) == 1
        assert len(m.turning_angles_deg) == 0
        assert len(m.accelerations_m_s2) == 0
        assert len(m.cumulative_distance_km) == 2

    def test_timestamps_datetime_formats(self) -> None:
        coords = np.array([[40.7, -74.0], [40.8, -74.0], [40.9, -74.0]])
        # Datetime objects
        dt0 = datetime.datetime(2026, 1, 1, 10, 0, 0)
        dts = [dt0, dt0 + datetime.timedelta(seconds=60), dt0 + datetime.timedelta(seconds=120)]
        m1 = trajectory_kinematics(coords, dts)
        assert np.allclose(m1.durations_sec, [60.0, 60.0])

        # Pandas DatetimeIndex
        pdt = pd.date_range("2026-01-01 10:00:00", periods=3, freq="1min")
        m2 = trajectory_kinematics(coords, pdt)
        assert np.allclose(m2.durations_sec, [60.0, 60.0])

        # String timestamps
        str_ts = ["2026-01-01 10:00:00", "2026-01-01 10:01:00", "2026-01-01 10:02:00"]
        m3 = trajectory_kinematics(coords, str_ts)
        assert np.allclose(m3.durations_sec, [60.0, 60.0])

    def test_errors(self) -> None:
        # Fewer than 2 points
        with pytest.raises(ValueError, match="Trajectory must contain at least 2 coordinate points"):
            trajectory_kinematics([[0.0, 0.0]], [100])

        # Length mismatch
        with pytest.raises(ValueError, match="Length mismatch"):
            trajectory_kinematics([[0.0, 0.0], [1.0, 1.0]], [100, 200, 300])

        # Non-monotonic timestamps
        with pytest.raises(ValueError, match="Timestamps must be strictly monotonically increasing"):
            trajectory_kinematics([[0.0, 0.0], [1.0, 1.0]], [200, 100])

        with pytest.raises(ValueError, match="Timestamps must be strictly monotonically increasing"):
            trajectory_kinematics([[0.0, 0.0], [1.0, 1.0]], [100, 100])


class TestTrajectoryResampling:
    """Tests for resample_trajectory."""

    def test_linear_temporal_resampling(self) -> None:
        # Move 10 degrees North over 100 seconds
        coords = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
        ])
        timestamps = np.array([0.0, 100.0])

        res_coords, res_times = resample_trajectory(coords, timestamps, step_seconds=20.0)
        # Expected times: 0, 20, 40, 60, 80, 100 (6 points)
        assert len(res_times) == 6
        assert np.allclose(res_times, [0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
        # Latitudes should be linearly spaced from 0 to 10
        assert np.allclose(res_coords[:, 0], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0], atol=1e-3)
        assert np.allclose(res_coords[:, 1], 0.0, atol=1e-3)

    def test_resampling_errors(self) -> None:
        with pytest.raises(ValueError, match="step_seconds must be positive"):
            resample_trajectory([[0.0, 0.0], [1.0, 1.0]], [0, 10], step_seconds=-1.0)

        with pytest.raises(ValueError, match="step_seconds must be positive"):
            resample_trajectory([[0.0, 0.0], [1.0, 1.0]], [0, 10], step_seconds=0.0)

        with pytest.raises(ValueError, match="Trajectory must contain at least 2 coordinate points"):
            resample_trajectory([[0.0, 0.0]], [0], step_seconds=5.0)
