"""Global Moran's I spatial autocorrelation and Monte Carlo permutation significance testing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from geostride.core.coordinates import validate_coordinates
from geostride.stats.weights import SpatialWeights, knn_weights


@dataclass(frozen=True)
class MoranResult:
    """Statistical summary of Global Moran's I spatial autocorrelation test.

    Attributes
    ----------
    moran_i : float
        Observed Global Moran's I statistic [-1.0, 1.0].
    expected_i : float
        Theoretical expectation under spatial randomness: -1 / (N - 1).
    variance_i : float
        Analytical variance under the null hypothesis of randomization.
    z_score : float
        Standard normal z-score based on analytical variance.
    p_value_analytical : float
        Two-tailed analytical p-value.
    p_value_sim : float
        Empirical pseudo p-value derived from Monte Carlo permutations.
    z_sim : float
        Standardized z-score computed against empirical permutation distribution.
    permutations : int
        Number of Monte Carlo permutations evaluated.
    sim_distribution : np.ndarray
        Array of simulated Moran's I statistics under the null hypothesis.
    """

    moran_i: float
    expected_i: float
    variance_i: float
    z_score: float
    p_value_analytical: float
    p_value_sim: float
    z_sim: float
    permutations: int
    sim_distribution: np.ndarray

    def __str__(self) -> str:
        return (
            f"Global Moran's I Results:\n"
            f"  Observed I          : {self.moran_i:>8.4f}\n"
            f"  Expected I          : {self.expected_i:>8.4f}\n"
            f"  Analytical Variance : {self.variance_i:>8.6f}\n"
            f"  Analytical Z-score  : {self.z_score:>8.4f} (p-value: {self.p_value_analytical:.4e})\n"
            f"  Permutation Z-score : {self.z_sim:>8.4f} (p-value: {self.p_value_sim:.4f}, M={self.permutations})"
        )


def _analytical_variance(w_mat: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Compute theoretical expectation and variance of Moran's I under randomization."""
    n = w_mat.shape[0]
    s0 = float(np.sum(w_mat))
    if s0 == 0.0:
        raise ValueError("Sum of spatial weights S0 is zero; cannot compute analytical variance.")

    expected_i = -1.0 / (n - 1)

    # S1 = 0.5 * sum_i sum_j (w_ij + w_ji)^2
    s1 = 0.5 * float(np.sum((w_mat + w_mat.T) ** 2))

    # S2 = sum_i (w_i. + w_.i)^2
    row_sums = np.sum(w_mat, axis=1)
    col_sums = np.sum(w_mat, axis=0)
    s2 = float(np.sum((row_sums + col_sums) ** 2))

    # Sample kurtosis b2 = m4 / m2^2
    m2 = float(np.sum(z**2)) / n
    m4 = float(np.sum(z**4)) / n
    b2 = m4 / (m2**2) if m2 > 0 else 3.0

    # Variance under randomization assumption
    term1 = n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * (s0**2))
    term2 = b2 * ((n**2 - n) * s1 - 2 * n * s2 + 6 * (s0**2))
    denom = (n - 1) * (n - 2) * (n - 3) * (s0**2)

    var_i = (term1 - term2) / denom - (expected_i**2)
    # Ensure numerical non-negativity
    var_i = max(var_i, 1e-12)
    return expected_i, var_i


def global_moran(
    coords: np.ndarray | Sequence[Sequence[float]],
    values: np.ndarray | Sequence[float],
    weights: SpatialWeights | np.ndarray | None = None,
    k_neighbors: int = 5,
    permutations: int = 999,
    random_state: int | None = 42,
) -> MoranResult:
    """Compute Global Moran's I spatial autocorrelation and statistical significance.

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Geographic coordinates [lat, lon].
    values : array-like of shape (N,)
        Observed continuous attribute values.
    weights : SpatialWeights or np.ndarray, optional
        Precomputed spatial weight matrix. If None, KNN weights (k=k_neighbors) are computed.
    k_neighbors : int, default=5
        Number of neighbors if `weights` is None.
    permutations : int, default=999
        Number of Monte Carlo random permutations for empirical significance testing.
    random_state : int, optional
        Random seed for reproducible Monte Carlo permutations.

    Returns
    -------
    MoranResult
        Dataclass containing observed Moran's I, expected value, analytical z-score/p-value,
        and permutation pseudo p-value.

    Raises
    ------
    ValueError
        If inputs have mismatched dimensions, N < 4, or values have zero variance.
    """
    arr_coords = validate_coordinates(coords)
    val_arr = np.asarray(values, dtype=np.float64).ravel()
    n = arr_coords.shape[0]

    if val_arr.shape[0] != n:
        raise ValueError(f"Mismatched lengths: coords has {n} rows, values has {val_arr.shape[0]}.")
    if n < 4:
        raise ValueError(f"Sample size N must be at least 4 for Moran's I; got {n}.")

    # Mean deviation z_i = y_i - y_mean
    y_mean = float(np.mean(val_arr))
    z = val_arr - y_mean
    ssq = float(np.sum(z**2))
    if math.isclose(ssq, 0.0, abs_tol=1e-12):
        raise ValueError("Values have zero variance (constant attribute). Moran's I is undefined.")

    # Prepare weight matrix
    if weights is None:
        w_obj = knn_weights(arr_coords, k=min(k_neighbors, n - 1), row_standardize=True)
        w_mat = w_obj.matrix
    elif isinstance(weights, SpatialWeights):
        w_mat = weights.matrix
    else:
        w_mat = np.asarray(weights, dtype=np.float64)
        if w_mat.shape != (n, n):
            raise ValueError(f"Weight matrix shape {w_mat.shape} must match (N, N)=({n}, {n}).")

    s0 = float(np.sum(w_mat))
    if s0 == 0.0:
        raise ValueError("Sum of spatial weights is zero.")

    # Calculate observed Moran's I: I = (N / S0) * (z^T W z) / (z^T z)
    spatial_lag_z = w_mat @ z
    numerator = float(z @ spatial_lag_z)
    moran_i = (n / s0) * (numerator / ssq)

    # Analytical Expectation & Variance
    expected_i, var_i = _analytical_variance(w_mat, z)
    se_i = math.sqrt(var_i)
    z_score = (moran_i - expected_i) / se_i if se_i > 0 else 0.0

    # Two-tailed standard normal p-value
    # erfc(|z| / sqrt(2)) gives two-tailed p-value
    p_value_analytical = math.erfc(abs(z_score) / math.sqrt(2.0))

    # Monte Carlo Permutation Test
    rng = np.random.default_rng(random_state)
    sim_i = np.empty(permutations, dtype=np.float64)

    for m in range(permutations):
        z_perm = rng.permutation(z)
        lag_perm = w_mat @ z_perm
        sim_i[m] = (n / s0) * (float(z_perm @ lag_perm) / ssq)

    # Pseudo p-value: proportion of permutations where |I_sim| >= |I_obs|
    extreme_count = int(np.sum(np.abs(sim_i) >= abs(moran_i)))
    p_value_sim = (extreme_count + 1) / (permutations + 1)

    sim_mean = float(np.mean(sim_i))
    sim_std = float(np.std(sim_i, ddof=1))
    z_sim = (moran_i - sim_mean) / sim_std if sim_std > 0 else 0.0

    return MoranResult(
        moran_i=moran_i,
        expected_i=expected_i,
        variance_i=var_i,
        z_score=z_score,
        p_value_analytical=p_value_analytical,
        p_value_sim=p_value_sim,
        z_sim=z_sim,
        permutations=permutations,
        sim_distribution=sim_i,
    )
