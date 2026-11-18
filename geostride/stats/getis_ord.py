"""Getis-Ord Gi and Gi* local spatial hot-spot and cold-spot statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from geostride.core.coordinates import validate_coordinates
from geostride.stats.weights import SpatialWeights, distance_band_weights, knn_weights


@dataclass(frozen=True)
class GetisOrdResult:
    """Summary record for local Getis-Ord Gi / Gi* hot-spot analysis.

    Attributes
    ----------
    z_scores : np.ndarray of shape (N,)
        Standard normal z-scores indicating spatial clustering magnitude.
    p_values : np.ndarray of shape (N,)
        Two-tailed significance p-values.
    labels : np.ndarray of shape (N,), dtype=object
        Categorical labels classifying each point (e.g. 'Hot Spot (99%)', 'Not Significant').
    star : bool
        Whether the calculation included self-weights (Gi*) or excluded self (Gi).
    """

    z_scores: np.ndarray
    p_values: np.ndarray
    labels: np.ndarray
    star: bool

    @property
    def hot_spots_99(self) -> np.ndarray:
        """Indices of statistically significant hot spots at 99% confidence (z >= 2.576)."""
        return np.where(self.z_scores >= 2.576)[0]

    @property
    def hot_spots_95(self) -> np.ndarray:
        """Indices of statistically significant hot spots at 95% confidence (z >= 1.96)."""
        return np.where(self.z_scores >= 1.96)[0]

    @property
    def cold_spots_99(self) -> np.ndarray:
        """Indices of statistically significant cold spots at 99% confidence (z <= -2.576)."""
        return np.where(self.z_scores <= -2.576)[0]

    @property
    def cold_spots_95(self) -> np.ndarray:
        """Indices of statistically significant cold spots at 95% confidence (z <= -1.96)."""
        return np.where(self.z_scores <= -1.96)[0]

    def summary(self) -> dict[str, int]:
        """Return counts of observations in each hot/cold spot category."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, [int(c) for c in counts]))


def _classify_getis_ord(z_scores: np.ndarray) -> np.ndarray:
    """Assign human-readable classification labels based on standard z-score cutoffs."""
    labels = np.full(z_scores.shape, "Not Significant", dtype=object)

    labels[z_scores >= 2.576] = "Hot Spot (99%)"
    labels[(z_scores >= 1.96) & (z_scores < 2.576)] = "Hot Spot (95%)"
    labels[(z_scores >= 1.645) & (z_scores < 1.96)] = "Hot Spot (90%)"

    labels[z_scores <= -2.576] = "Cold Spot (99%)"
    labels[(z_scores <= -1.96) & (z_scores > -2.576)] = "Cold Spot (95%)"
    labels[(z_scores <= -1.645) & (z_scores > -1.96)] = "Cold Spot (90%)"

    return labels


def getis_ord_local(
    coords: np.ndarray | Sequence[Sequence[float]],
    values: np.ndarray | Sequence[float],
    star: bool = True,
    weights: SpatialWeights | np.ndarray | None = None,
    threshold_km: float | None = None,
    k_neighbors: int = 5,
) -> GetisOrdResult:
    """Compute local Getis-Ord Gi or Gi* statistics for each geographic point.

    The Getis-Ord statistic compares the local sum of an attribute in a neighborhood
    to the global sum across all observations. A high positive z-score indicates a
    statistically significant cluster of high values (Hot Spot); a negative z-score
    indicates a cluster of low values (Cold Spot).

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Geographic coordinates [lat, lon].
    values : array-like of shape (N,)
        Attribute values associated with each location.
    star : bool, default=True
        If True, computes Gi* (includes self in neighborhood sum).
        If False, computes Gi (excludes self).
    weights : SpatialWeights or np.ndarray, optional
        Precomputed spatial weight matrix. If None, derived from `threshold_km` or `k_neighbors`.
    threshold_km : float, optional
        Distance threshold in kilometers for distance-band weights if `weights` is None.
    k_neighbors : int, default=5
        Number of neighbors if both `weights` and `threshold_km` are None.

    Returns
    -------
    GetisOrdResult
        Summary containing z-scores, two-tailed p-values, and categorical classifications.

    Raises
    ------
    ValueError
        If sample size N < 3, dimensions mismatch, or values have zero variance.
    """
    arr_coords = validate_coordinates(coords)
    x = np.asarray(values, dtype=np.float64).ravel()
    n = arr_coords.shape[0]

    if x.shape[0] != n:
        raise ValueError(f"Mismatched lengths: coords has {n} rows, values has {x.shape[0]}.")
    if n < 3:
        raise ValueError(f"Sample size N must be at least 3 for Getis-Ord; got {n}.")

    x_mean = float(np.mean(x))
    # S = sqrt( (1 / N) * sum(x^2) - x_mean^2 )
    mean_sq = float(np.mean(x**2))
    var = mean_sq - (x_mean**2)
    if var <= 1e-14:
        raise ValueError("Values have zero variance (constant attribute). Getis-Ord is undefined.")
    s = math.sqrt(var)

    # Prepare weight matrix (binary un-standardized weights are standard for Getis-Ord)
    if weights is not None:
        if isinstance(weights, SpatialWeights):
            w_mat = weights.matrix.copy()
        else:
            w_mat = np.array(weights, dtype=np.float64, copy=True)
            if w_mat.shape != (n, n):
                raise ValueError(f"Weight matrix shape {w_mat.shape} must match ({n}, {n}).")
    elif threshold_km is not None:
        w_mat = distance_band_weights(arr_coords, threshold_km=threshold_km, row_standardize=False).matrix.copy()
    else:
        w_mat = knn_weights(arr_coords, k=min(k_neighbors, n - 1), row_standardize=False).matrix.copy()

    if star:
        # Gi* includes self: set diagonal to 1.0
        np.fill_diagonal(w_mat, 1.0)
    else:
        # Gi excludes self: ensure diagonal is 0.0
        np.fill_diagonal(w_mat, 0.0)

    # Row sums: W_i = sum_j w_ij
    w_row_sums = np.sum(w_mat, axis=1)  # shape (N,)
    # Row squared sums: W_i^2_sum = sum_j w_ij^2
    w_row_sq_sums = np.sum(w_mat**2, axis=1)  # shape (N,)

    # Numerator: sum_j w_ij * x_j - x_mean * W_i
    local_sums = w_mat @ x
    numerator = local_sums - x_mean * w_row_sums

    # Denominator: S * sqrt( (N * sum_j w_ij^2 - (sum_j w_ij)^2) / (N - 1) )
    denom_inner = (n * w_row_sq_sums - (w_row_sums**2)) / (n - 1)
    denom_inner = np.maximum(denom_inner, 0.0)
    denominator = s * np.sqrt(denom_inner)

    # Guard against zero denominator (isolated island node with 0 neighbors)
    z_scores = np.zeros(n, dtype=np.float64)
    valid_mask = denominator > 1e-12
    z_scores[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    # Two-tailed p-values: erfc(|z| / sqrt(2))
    p_values = np.array([math.erfc(abs(z) / math.sqrt(2.0)) for z in z_scores], dtype=np.float64)
    labels = _classify_getis_ord(z_scores)

    return GetisOrdResult(
        z_scores=z_scores,
        p_values=p_values,
        labels=labels,
        star=star,
    )
