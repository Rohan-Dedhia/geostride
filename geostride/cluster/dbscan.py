"""Geographic spatial DBSCAN clustering using Haversine metric space."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from geostride.core.coordinates import validate_coordinates
from geostride.metrics.haversine import EARTH_RADIUS


@dataclass(frozen=True)
class SpatialDBSCANResult:
    """Summary record for spatial DBSCAN clustering.

    Attributes
    ----------
    labels : np.ndarray of shape (N,)
        Cluster labels for each point. Noisy samples are given the label -1.
    n_clusters : int
        Number of clusters found by the algorithm (excluding noise).
    cluster_centroids : dict of int to tuple of (float, float)
        Mapping of cluster id to its geometric centroid (latitude, longitude).
    noise_ratio : float
        Fraction of samples classified as noise [-1].
    cluster_sizes : dict of int to int
        Number of points assigned to each cluster.
    """

    labels: np.ndarray
    n_clusters: int
    cluster_centroids: dict[int, tuple[float, float]]
    noise_ratio: float
    cluster_sizes: dict[int, int]

    def __str__(self) -> str:
        return (
            f"Spatial DBSCAN Clustering Results:\n"
            f"  Clusters Found  : {self.n_clusters}\n"
            f"  Noise Ratio     : {self.noise_ratio:.2%}\n"
            f"  Cluster Sizes   : {self.cluster_sizes}"
        )


def _lat_lon_to_cartesian_3d(coords: np.ndarray, radius_km: float) -> np.ndarray:
    """Convert spherical (lat, lon) coordinates in degrees to 3D Cartesian coordinates (X, Y, Z)."""
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])

    x = radius_km * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius_km * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius_km * np.sin(lat_rad)
    return np.column_stack([x, y, z])


def spatial_dbscan(
    coords: np.ndarray | Sequence[Sequence[float]],
    eps_km: float = 1.0,
    min_samples: int = 5,
) -> SpatialDBSCANResult:
    """Cluster geographic coordinates using spatial DBSCAN with Haversine distance.

    Utilizes 3D Euclidean chord space indexing over an Earth-radius sphere to achieve
    efficient O(N log N) spatial neighbor retrieval without external C++ GIS dependencies.

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Input coordinate points [latitude, longitude] in decimal degrees.
    eps_km : float, default=1.0
        Maximum geographic distance in kilometers between two samples for one
        to be considered as in the neighborhood of the other.
    min_samples : int, default=5
        Number of samples in a neighborhood for a point to be considered as a core point.

    Returns
    -------
    SpatialDBSCANResult
        Clustering results containing cluster labels, centroids, and cluster size statistics.

    Raises
    ------
    ValueError
        If eps_km <= 0, min_samples < 1, or input coordinates are invalid.
    """
    if eps_km <= 0.0:
        raise ValueError(f"eps_km must be strictly positive; got {eps_km}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be at least 1; got {min_samples}")

    arr_coords = validate_coordinates(coords)
    n = arr_coords.shape[0]

    if n == 0:
        return SpatialDBSCANResult(
            labels=np.empty(0, dtype=int),
            n_clusters=0,
            cluster_centroids={},
            noise_ratio=0.0,
            cluster_sizes={},
        )

    r_earth = EARTH_RADIUS["km"]

    # Convert great-circle distance cutoff eps_km into 3D Euclidean chord distance:
    # chord_dist = 2 * R * sin(d / (2 * R))
    angular_dist = eps_km / r_earth
    # If angular distance exceeds pi, chord distance is maximum diameter (2 * R)
    if angular_dist >= np.pi:
        chord_threshold = 2.0 * r_earth
    else:
        chord_threshold = 2.0 * r_earth * np.sin(angular_dist / 2.0)

    # Transform coordinates to 3D Cartesian space on the sphere
    pts_3d = _lat_lon_to_cartesian_3d(arr_coords, radius_km=r_earth)
    tree = cKDTree(pts_3d)

    # Query all points within chord threshold distance (includes self)
    neighbors_list = tree.query_ball_tree(tree, r=chord_threshold)

    labels = np.full(n, -1, dtype=int)  # -1 represents noise / unassigned
    visited = np.zeros(n, dtype=bool)
    current_cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors_i = neighbors_list[i]
        # Core point check: must have at least min_samples neighbors (including self)
        if len(neighbors_i) < min_samples:
            labels[i] = -1  # Noise
            continue

        # Start a new cluster
        labels[i] = current_cluster_id
        queue: deque[int] = deque(neighbors_i)

        while queue:
            neighbor_pt = queue.popleft()
            if not visited[neighbor_pt]:
                visited[neighbor_pt] = True
                neighbor_pt_neighbors = neighbors_list[neighbor_pt]
                if len(neighbor_pt_neighbors) >= min_samples:
                    queue.extend(neighbor_pt_neighbors)

            # Assign to current cluster if not yet assigned to any cluster
            if labels[neighbor_pt] == -1:
                labels[neighbor_pt] = current_cluster_id

        current_cluster_id += 1

    n_clusters = current_cluster_id
    noise_count = int(np.sum(labels == -1))
    noise_ratio = noise_count / n if n > 0 else 0.0

    # Compute centroids and cluster sizes
    cluster_centroids: dict[int, tuple[float, float]] = {}
    cluster_sizes: dict[int, int] = {}

    for cid in range(n_clusters):
        mask = labels == cid
        c_pts = arr_coords[mask]
        cluster_sizes[cid] = int(c_pts.shape[0])
        centroid_lat = float(np.mean(c_pts[:, 0]))
        centroid_lon = float(np.mean(c_pts[:, 1]))
        cluster_centroids[cid] = (centroid_lat, centroid_lon)

    return SpatialDBSCANResult(
        labels=labels,
        n_clusters=n_clusters,
        cluster_centroids=cluster_centroids,
        noise_ratio=noise_ratio,
        cluster_sizes=cluster_sizes,
    )
