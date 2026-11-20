"""Trajectory kinematics, resamplers, and spatio-temporal path analytics."""

from __future__ import annotations

from geostride.trajectory.metrics import (
    TrajectoryMetrics,
    resample_trajectory,
    trajectory_kinematics,
)
from geostride.trajectory.simplify import (
    cross_track_distance_meters,
    simplify_trajectory,
)

__all__ = [
    "TrajectoryMetrics",
    "cross_track_distance_meters",
    "resample_trajectory",
    "simplify_trajectory",
    "trajectory_kinematics",
]
