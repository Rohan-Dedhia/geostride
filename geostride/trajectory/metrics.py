"""Spatio-temporal trajectory analytics, kinematics, and resamplers."""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from geostride.core.coordinates import validate_coordinates
from geostride.metrics.haversine import haversine_pairwise


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    """Kinematics metrics for a spatio-temporal trajectory.

    Attributes
    ----------
    distances_km : np.ndarray
        Step-by-step distances in kilometers between consecutive points, shape (N-1,).
    durations_sec : np.ndarray
        Step-by-step elapsed time in seconds between consecutive points, shape (N-1,).
    speeds_kmh : np.ndarray
        Step-by-step average speed in km/h, shape (N-1,).
    headings_deg : np.ndarray
        Initial forward azimuth heading in degrees [0, 360) along each segment, shape (N-1,).
    turning_angles_deg : np.ndarray
        Angular change in heading between consecutive segments in [-180, 180) degrees, shape (N-2,).
    accelerations_m_s2 : np.ndarray
        Linear acceleration in m/s^2 between consecutive segments, shape (N-2,).
    cumulative_distance_km : np.ndarray
        Cumulative distance along the trajectory starting at 0.0, shape (N,).
    """

    distances_km: np.ndarray
    durations_sec: np.ndarray
    speeds_kmh: np.ndarray
    headings_deg: np.ndarray
    turning_angles_deg: np.ndarray
    accelerations_m_s2: np.ndarray
    cumulative_distance_km: np.ndarray

    @property
    def total_distance_km(self) -> float:
        """Total trajectory path length in kilometers."""
        return float(np.sum(self.distances_km))

    @property
    def total_duration_sec(self) -> float:
        """Total elapsed trajectory duration in seconds."""
        return float(np.sum(self.durations_sec))

    @property
    def mean_speed_kmh(self) -> float:
        """Overall average speed in km/h across the trajectory."""
        tot_time = self.total_duration_sec
        if tot_time <= 0:
            return 0.0
        return float(self.total_distance_km / (tot_time / 3600.0))

    @property
    def max_speed_kmh(self) -> float:
        """Peak instantaneous/step speed in km/h."""
        if len(self.speeds_kmh) == 0:
            return 0.0
        return float(np.max(self.speeds_kmh))


def _to_epoch_seconds(timestamps: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Convert a sequence of timestamps or numbers into a 1D float array of epoch seconds."""
    if isinstance(timestamps, (pd.Series, pd.DatetimeIndex)):
        return timestamps.to_numpy(dtype="datetime64[ns]").astype(np.float64) / 1e9

    if isinstance(timestamps, np.ndarray) and np.issubdtype(timestamps.dtype, np.datetime64):
        return timestamps.astype("datetime64[ns]").astype(np.float64) / 1e9

    if hasattr(timestamps, "__len__") and len(timestamps) > 0:
        first = timestamps[0]
        if isinstance(first, (str, pd.Timestamp, datetime.datetime, np.datetime64)):
            dt_idx = pd.to_datetime(timestamps)
            return dt_idx.to_numpy(dtype="datetime64[ns]").astype(np.float64) / 1e9

    return np.asarray(timestamps, dtype=np.float64)


def _compute_forward_azimuth(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Compute forward initial azimuth (heading) in degrees [0, 360) from point 1 to point 2."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)

    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360.0) % 360.0
    return bearing_deg


def trajectory_kinematics(
    coords: np.ndarray | Sequence[Sequence[float]],
    timestamps: Sequence[Any] | np.ndarray,
) -> TrajectoryMetrics:
    """Compute comprehensive kinematic attributes for a spatio-temporal GPS trajectory.

    Parameters
    ----------
    coords : np.ndarray or Sequence[Sequence[float]]
        Coordinate array of shape (N, 2) in [latitude, longitude] decimal degrees.
    timestamps : Sequence[Any] or np.ndarray
        Temporal sequence of length N (timestamps, datetimes, or epoch seconds). Must be
        strictly monotonically increasing.

    Returns
    -------
    TrajectoryMetrics
        Dataclass containing distance, speed, heading, turning angles, and accelerations.

    Raises
    ------
    ValueError
        If coords and timestamps length mismatch, N < 2, or timestamps are not strictly increasing.
    """
    arr = validate_coordinates(coords)
    n = len(arr)
    if n < 2:
        raise ValueError(f"Trajectory must contain at least 2 coordinate points; got {n}.")

    t_sec = _to_epoch_seconds(timestamps)
    if len(t_sec) != n:
        raise ValueError(
            f"Length mismatch: coords has {n} points, but timestamps has {len(t_sec)}."
        )

    durations = np.diff(t_sec)
    if np.any(durations <= 0):
        raise ValueError("Timestamps must be strictly monotonically increasing.")

    lat1, lon1 = arr[:-1, 0], arr[:-1, 1]
    lat2, lon2 = arr[1:, 0], arr[1:, 1]

    # Geodesic step distances
    distances_km = np.asarray(haversine_pairwise(lat1, lon1, lat2, lon2, unit="km"), dtype=np.float64)

    # Speeds in km/h
    hours = durations / 3600.0
    speeds_kmh = distances_km / hours

    # Forward headings in degrees
    headings_deg = _compute_forward_azimuth(lat1, lon1, lat2, lon2)

    # Cumulative distance
    cum_dist = np.empty(n, dtype=np.float64)
    cum_dist[0] = 0.0
    cum_dist[1:] = np.cumsum(distances_km)

    if n >= 3:
        # Turning angles normalized to [-180, 180)
        diff_headings = headings_deg[1:] - headings_deg[:-1]
        turning_angles_deg = (diff_headings + 180.0) % 360.0 - 180.0

        # Acceleration in m/s^2: delta_v / delta_t
        v_mps = speeds_kmh / 3.6
        delta_v = v_mps[1:] - v_mps[:-1]
        delta_t = 0.5 * (durations[:-1] + durations[1:])
        accelerations_m_s2 = delta_v / delta_t
    else:
        turning_angles_deg = np.empty(0, dtype=np.float64)
        accelerations_m_s2 = np.empty(0, dtype=np.float64)

    return TrajectoryMetrics(
        distances_km=distances_km,
        durations_sec=durations,
        speeds_kmh=speeds_kmh,
        headings_deg=headings_deg,
        turning_angles_deg=turning_angles_deg,
        accelerations_m_s2=accelerations_m_s2,
        cumulative_distance_km=cum_dist,
    )


def resample_trajectory(
    coords: np.ndarray | Sequence[Sequence[float]],
    timestamps: Sequence[Any] | np.ndarray,
    step_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a GPS trajectory to a uniform temporal interval using spherical interpolation (Slerp).

    Parameters
    ----------
    coords : np.ndarray or Sequence[Sequence[float]]
        Coordinate array of shape (N, 2) with columns [latitude, longitude].
    timestamps : Sequence[Any] or np.ndarray
        Temporal sequence of length N.
    step_seconds : float
        Uniform resampling interval in seconds.

    Returns
    -------
    resampled_coords : np.ndarray
        Resampled coordinate array of shape (M, 2) with columns [latitude, longitude].
    resampled_timestamps : np.ndarray
        Resampled epoch seconds of shape (M,).

    Raises
    ------
    ValueError
        If step_seconds <= 0, trajectory length < 2, or timestamps are not monotonic.
    """
    if step_seconds <= 0:
        raise ValueError(f"step_seconds must be positive; got {step_seconds}")

    arr = validate_coordinates(coords)
    n = len(arr)
    if n < 2:
        raise ValueError(f"Trajectory must contain at least 2 coordinate points; got {n}.")

    t_sec = _to_epoch_seconds(timestamps)
    if len(t_sec) != n:
        raise ValueError(
            f"Length mismatch: coords has {n} points, but timestamps has {len(t_sec)}."
        )

    if np.any(np.diff(t_sec) <= 0):
        raise ValueError("Timestamps must be strictly monotonically increasing.")

    t_start, t_end = t_sec[0], t_sec[-1]
    new_times = np.arange(t_start, t_end + 1e-6, step_seconds, dtype=np.float64)
    if len(new_times) == 0:
        new_times = np.array([t_start], dtype=np.float64)

    m = len(new_times)
    resampled = np.empty((m, 2), dtype=np.float64)

    # Convert coordinates to radians
    lats_rad = np.radians(arr[:, 0])
    lons_rad = np.radians(arr[:, 1])

    # Convert to 3D Cartesian coordinates on unit sphere
    x = np.cos(lats_rad) * np.cos(lons_rad)
    y = np.cos(lats_rad) * np.sin(lons_rad)
    z = np.sin(lats_rad)
    pts_3d = np.column_stack([x, y, z])

    # Segment index for each resampled time point
    seg_idx = np.searchsorted(t_sec, new_times, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, n - 2)

    for i in range(m):
        idx = seg_idx[i]
        t = new_times[i]
        t0, t1 = t_sec[idx], t_sec[idx + 1]
        dt = t1 - t0

        if dt <= 0 or t <= t0:
            resampled[i] = arr[idx]
            continue
        elif t >= t1:
            resampled[i] = arr[idx + 1]
            continue

        alpha = (t - t0) / dt
        p0 = pts_3d[idx]
        p1 = pts_3d[idx + 1]

        # Dot product
        dot = np.clip(np.dot(p0, p1), -1.0, 1.0)
        omega = np.arccos(dot)

        if omega < 1e-12:
            p_interp = p0
        else:
            sin_omega = np.sin(omega)
            p_interp = (np.sin((1.0 - alpha) * omega) / sin_omega) * p0 + (np.sin(alpha * omega) / sin_omega) * p1

        # Convert back from 3D unit sphere to [lat, lon]
        norm = np.linalg.norm(p_interp)
        if norm > 0:
            p_interp = p_interp / norm
        interp_lat = np.degrees(np.arcsin(np.clip(p_interp[2], -1.0, 1.0)))
        interp_lon = np.degrees(np.arctan2(p_interp[1], p_interp[0]))
        resampled[i] = [interp_lat, interp_lon]

    return resampled, new_times
