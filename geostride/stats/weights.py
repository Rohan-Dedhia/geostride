"""Spatial weight matrices and neighbor graph topology representations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from geostride.core.coordinates import validate_coordinates
from geostride.metrics.haversine import haversine_matrix


@dataclass(frozen=True)
class SpatialWeights:
    """Container for spatial weight matrices and neighbor topology statistics.

    Parameters
    ----------
    matrix : np.ndarray of shape (N, N)
        Spatial weight adjacency matrix.
    transform : {"B", "R"}, default="B"
        Weight transformation scheme: "B" for binary, "R" for row-standardized.
    """

    matrix: np.ndarray
    transform: str = "B"

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"Weight matrix must be square (N, N); got shape {self.matrix.shape}"
            )
        # Ensure zero diagonal
        if np.any(np.diagonal(self.matrix) != 0.0):
            raise ValueError("Spatial weight matrix must have a zero diagonal (no self-neighbors).")

    @property
    def n_observations(self) -> int:
        """Total number of observations (nodes) in the spatial graph."""
        return self.matrix.shape[0]

    @property
    def s0(self) -> float:
        """Sum of all spatial weights: S_0 = sum_i sum_j w_ij."""
        return float(np.sum(self.matrix))

    @property
    def cardinalities(self) -> np.ndarray:
        """Vector of neighbor counts (degree) for each observation."""
        return np.sum(self.matrix > 0.0, axis=1)

    @property
    def islands(self) -> list[int]:
        """List of row indices corresponding to spatial islands (0 neighbors)."""
        return [int(i) for i in np.where(self.cardinalities == 0)[0]]

    @property
    def has_islands(self) -> bool:
        """Whether any observations are isolated with zero neighbors."""
        return len(self.islands) > 0

    def standardize(self, style: str = "row") -> SpatialWeights:
        """Return a new SpatialWeights instance transformed according to the specified standardization style.

        Parameters
        ----------
        style : {"row", "r", "binary", "b", "spectral", "v", "minmax", "m"}, default="row"
            Standardization style:
            - "row" / "r": Row-standardization where weights in each row sum to 1.0 (averages neighborhood).
            - "binary" / "b": Binary connectivity thresholding where all non-zero weights become 1.0.
            - "spectral" / "v": Rescales weights by the largest eigenvalue lambda_max (spectral radius = 1.0).
            - "minmax" / "m": Rescales weights by the minimum of maximum row or column sums.

        Returns
        -------
        SpatialWeights
            New standardized spatial weights instance.

        Raises
        ------
        ValueError
            If style is not recognized.
        """
        key = style.lower().strip()
        if key in ("row", "r"):
            row_sums = np.sum(self.matrix, axis=1, keepdims=True)
            safe_sums = np.where(row_sums == 0.0, 1.0, row_sums)
            standardized = self.matrix / safe_sums
            return SpatialWeights(matrix=standardized, transform="R")

        elif key in ("binary", "b"):
            standardized = np.where(self.matrix > 0.0, 1.0, 0.0).astype(np.float64)
            np.fill_diagonal(standardized, 0.0)
            return SpatialWeights(matrix=standardized, transform="B")

        elif key in ("spectral", "v"):
            eigs = np.linalg.eigvals(self.matrix)
            max_eig = float(np.max(np.real(eigs)))
            if max_eig > 1e-12:
                standardized = self.matrix / max_eig
            else:
                standardized = self.matrix.copy()
            return SpatialWeights(matrix=standardized, transform="V")

        elif key in ("minmax", "m"):
            r_max = float(np.max(np.sum(self.matrix, axis=1)))
            c_max = float(np.max(np.sum(self.matrix, axis=0)))
            s = min(r_max, c_max)
            if s > 1e-12:
                standardized = self.matrix / s
            else:
                standardized = self.matrix.copy()
            return SpatialWeights(matrix=standardized, transform="M")

        else:
            raise ValueError(
                f"Unsupported standardization style '{style}'. "
                "Choose from 'row' ('r'), 'binary' ('b'), 'spectral' ('v'), or 'minmax' ('m')."
            )

    def set_transform(self, style: str = "row") -> SpatialWeights:
        """Alias for standardize(style)."""
        return self.standardize(style)

    def row_standardize(self) -> SpatialWeights:
        """Return a new SpatialWeights instance with row-standardized weights (rows sum to 1.0).

        For spatial islands (rows summing to 0), weights remain 0 without division-by-zero errors.
        """
        return self.standardize("row")


def standardize_weights(
    weights: SpatialWeights | np.ndarray,
    style: str = "row",
) -> SpatialWeights | np.ndarray:
    """Standardize a spatial weight container or square weight matrix.

    Parameters
    ----------
    weights : SpatialWeights or np.ndarray of shape (N, N)
        Spatial weight container or square matrix.
    style : {"row", "binary", "spectral", "minmax"}, default="row"
        Standardization style.

    Returns
    -------
    SpatialWeights or np.ndarray
        Standardized weights matching the input type.
    """
    if isinstance(weights, SpatialWeights):
        return weights.standardize(style)

    if isinstance(weights, np.ndarray):
        w = SpatialWeights(matrix=weights.copy(), transform="B")
        w_std = w.standardize(style)
        return w_std.matrix

    raise TypeError(
        f"weights must be a SpatialWeights container or a NumPy ndarray, got {type(weights).__name__}"
    )


def knn_weights(
    coords: np.ndarray | Sequence[Sequence[float]],
    k: int = 5,
    row_standardize: bool = True,
) -> SpatialWeights:
    """Generate spatial weight matrix based on K-nearest geographic neighbors.

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Input coordinate points [lat, lon].
    k : int, default=5
        Number of nearest neighbors per node.
    row_standardize : bool, default=True
        Whether to standardize rows so weights sum to 1.0.

    Returns
    -------
    SpatialWeights
        Fitted spatial weights container.

    Raises
    ------
    ValueError
        If N <= k or k < 1.
    """
    arr = validate_coordinates(coords)
    n = arr.shape[0]

    if k < 1:
        raise ValueError(f"k must be at least 1; got {k}")
    if n <= k:
        raise ValueError(f"KNN weights require N > k; got N={n} and k={k}")

    # Compute full pairwise distance matrix in kilometers
    dist_mat = haversine_matrix(arr, arr, unit="km")
    # Exclude self by setting diagonal to infinity
    np.fill_diagonal(dist_mat, np.inf)

    # Find k smallest indices per row
    neighbor_indices = np.argpartition(dist_mat, kth=k - 1, axis=1)[:, :k]

    weight_mat = np.zeros((n, n), dtype=np.float64)
    row_idx = np.arange(n)[:, np.newaxis]
    weight_mat[row_idx, neighbor_indices] = 1.0

    w = SpatialWeights(matrix=weight_mat, transform="B")
    return w.row_standardize() if row_standardize else w


def distance_band_weights(
    coords: np.ndarray | Sequence[Sequence[float]],
    threshold_km: float,
    row_standardize: bool = True,
) -> SpatialWeights:
    """Generate spatial weight matrix based on fixed geographic distance threshold.

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Input coordinate points [lat, lon].
    threshold_km : float
        Maximum distance cutoff in kilometers. Points with d(i, j) <= threshold_km are neighbors.
    row_standardize : bool, default=True
        Whether to standardize rows so weights sum to 1.0.

    Returns
    -------
    SpatialWeights
        Fitted spatial weights container.
    """
    if threshold_km <= 0.0:
        raise ValueError(f"threshold_km must be strictly positive; got {threshold_km}")

    arr = validate_coordinates(coords)
    dist_mat = haversine_matrix(arr, arr, unit="km")
    np.fill_diagonal(dist_mat, np.inf)

    weight_mat = np.where(dist_mat <= threshold_km, 1.0, 0.0).astype(np.float64)
    w = SpatialWeights(matrix=weight_mat, transform="B")
    return w.row_standardize() if row_standardize else w


def inverse_distance_weights(
    coords: np.ndarray | Sequence[Sequence[float]],
    alpha: float = 1.0,
    threshold_km: float | None = None,
    epsilon: float = 1e-5,
    row_standardize: bool = True,
) -> SpatialWeights:
    """Generate continuous spatial weights with inverse distance decay: w_ij = (d_ij + epsilon)^(-alpha).

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Input coordinate points [lat, lon].
    alpha : float, default=1.0
        Distance decay exponent.
    threshold_km : float, optional
        Optional maximum cutoff distance. Points beyond this distance are assigned weight 0.
    epsilon : float, default=1e-5
        Small positive constant to guard against division by zero on coincident points.
    row_standardize : bool, default=True
        Whether to row-standardize weights.

    Returns
    -------
    SpatialWeights
        Fitted spatial weights container.
    """
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative; got {alpha}")

    arr = validate_coordinates(coords)
    dist_mat = haversine_matrix(arr, arr, unit="km")
    np.fill_diagonal(dist_mat, np.inf)

    weight_mat = 1.0 / np.power(dist_mat + epsilon, alpha)
    np.fill_diagonal(weight_mat, 0.0)

    if threshold_km is not None:
        weight_mat = np.where(dist_mat <= threshold_km, weight_mat, 0.0)

    w = SpatialWeights(matrix=weight_mat, transform="B")
    return w.row_standardize() if row_standardize else w
