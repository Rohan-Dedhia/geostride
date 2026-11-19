"""Local Indicators of Spatial Association (LISA) and Moran quadrant classification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from geostride.core.coordinates import validate_coordinates
from geostride.stats.weights import SpatialWeights, knn_weights

_QUADRANT_LABELS = {
    0: "Not Significant",
    1: "High-High",
    2: "Low-High",
    3: "Low-Low",
    4: "High-Low",
}


@dataclass(frozen=True)
class LISAResult:
    """Summary record for Local Indicators of Spatial Association (LISA) decomposition.

    Attributes
    ----------
    local_moran : np.ndarray of shape (N,)
        Observation-level Local Moran's I statistics ($I_i$).
    quadrants : np.ndarray of shape (N,), dtype=int8
        Significant spatial cluster quadrants:
        - 1: High-High (spatial hot spots)
        - 2: Low-High (spatial outliers)
        - 3: Low-Low (spatial cold spots)
        - 4: High-Low (spatial outliers)
        - 0: Not Significant ($p > p_{\text{threshold}}$)
    raw_quadrants : np.ndarray of shape (N,), dtype=int8
        Unconditional quadrant assignments (1 to 4) regardless of statistical significance.
    quadrant_labels : np.ndarray of shape (N,), dtype=object
        Human-readable categorical quadrant string labels.
    p_values : np.ndarray of shape (N,)
        Empirical pseudo p-values computed via conditional Monte Carlo spatial permutations.
    z_scores : np.ndarray of shape (N,)
        Standardized z-scores relative to empirical permutation distributions.
    p_threshold : float
        Significance alpha threshold used for quadrant filtering (e.g. 0.05).
    """

    local_moran: np.ndarray
    quadrants: np.ndarray
    raw_quadrants: np.ndarray
    quadrant_labels: np.ndarray
    p_values: np.ndarray
    z_scores: np.ndarray
    p_threshold: float

    @property
    def high_high(self) -> np.ndarray:
        """Indices of statistically significant High-High clusters (hot spots)."""
        return np.where(self.quadrants == 1)[0]

    @property
    def low_high(self) -> np.ndarray:
        """Indices of statistically significant Low-High spatial outliers."""
        return np.where(self.quadrants == 2)[0]

    @property
    def low_low(self) -> np.ndarray:
        """Indices of statistically significant Low-Low clusters (cold spots)."""
        return np.where(self.quadrants == 3)[0]

    @property
    def high_low(self) -> np.ndarray:
        """Indices of statistically significant High-Low spatial outliers."""
        return np.where(self.quadrants == 4)[0]

    @property
    def not_significant(self) -> np.ndarray:
        """Indices of observations where spatial autocorrelation is not statistically significant."""
        return np.where(self.quadrants == 0)[0]

    def __str__(self) -> str:
        hh = len(self.high_high)
        lh = len(self.low_high)
        ll = len(self.low_low)
        hl = len(self.high_low)
        ns = len(self.not_significant)
        n = len(self.local_moran)
        return (
            f"LISA Cluster Analysis (N={n}, p<={self.p_threshold:.3f}):\n"
            f"  High-High (Hot Spots)     : {hh:>5} ({hh / n:>5.1%})\n"
            f"  Low-Low (Cold Spots)      : {ll:>5} ({ll / n:>5.1%})\n"
            f"  High-Low (Outliers)       : {hl:>5} ({hl / n:>5.1%})\n"
            f"  Low-High (Outliers)       : {lh:>5} ({lh / n:>5.1%})\n"
            f"  Not Significant           : {ns:>5} ({ns / n:>5.1%})"
        )


def local_moran(
    coords: np.ndarray | Sequence[Sequence[float]],
    values: np.ndarray | Sequence[float],
    weights: SpatialWeights | None = None,
    k_neighbors: int = 5,
    n_permutations: int = 999,
    p_threshold: float = 0.05,
    seed: int | None = None,
) -> LISAResult:
    """Compute Local Indicators of Spatial Association (LISA / Anselin 1995) statistics.

    Decomposes global spatial autocorrelation into observation-specific statistics ($I_i$)
    and classifies each spatial feature into one of four Moran quadrants:
    High-High, Low-Low, High-Low, and Low-High.

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Input coordinate points [latitude, longitude].
    values : array-like of shape (N,)
        Observation attribute values.
    weights : SpatialWeights, optional
        Pre-computed spatial weights container. If None, K-nearest neighbor weights
        with `k_neighbors` will be constructed.
    k_neighbors : int, default=5
        Number of nearest neighbors if `weights` is None.
    n_permutations : int, default=999
        Number of conditional Monte Carlo spatial permutations for pseudo p-value inference.
    p_threshold : float, default=0.05
        Statistical significance alpha threshold for identifying significant clusters.
    seed : int, optional
        Random seed for reproducible Monte Carlo permutations.

    Returns
    -------
    LISAResult
        Summary container holding observation-level $I_i$, cluster quadrants, and p-values.

    Raises
    ------
    ValueError
        If coordinates or values have incompatible shapes, values have zero variance,
        or parameters are invalid.

    References
    ----------
    Anselin, L. (1995). "Local Indicators of Spatial Association—LISA".
    Geographical Analysis, 27(2), 93-115.

    Examples
    --------
    >>> import numpy as np
    >>> from geostride.stats import local_moran
    >>> coords = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]])
    >>> vals = np.array([100.0, 105.0, 98.0, 5.0, 4.0, 6.0])
    >>> res = local_moran(coords, vals, k_neighbors=2, seed=42)
    >>> len(res.high_high) > 0
    True
    """
    arr_coords = validate_coordinates(coords)
    n = arr_coords.shape[0]

    arr_vals = np.asarray(values, dtype=np.float64).ravel()
    if arr_vals.shape[0] != n:
        raise ValueError(
            f"Shape mismatch: {n} coordinates provided but {arr_vals.shape[0]} values."
        )

    if n < 4:
        raise ValueError(f"LISA requires at least 4 observations; got {n}")

    if p_threshold <= 0.0 or p_threshold >= 1.0:
        raise ValueError(f"p_threshold must be in (0, 1); got {p_threshold}")

    if n_permutations < 9:
        raise ValueError(f"n_permutations must be at least 9; got {n_permutations}")

    # Build or row-standardize spatial weights
    if weights is None:
        actual_k = min(k_neighbors, n - 1)
        w_obj = knn_weights(arr_coords, k=actual_k, row_standardize=True)
    else:
        if weights.n_observations != n:
            raise ValueError(
                f"Weights matrix dimension ({weights.n_observations}) does not match "
                f"number of coordinates ({n})."
            )
        w_obj = weights.row_standardize()

    w_mat = w_obj.matrix

    # Center observation values
    mean_val = float(np.mean(arr_vals))
    z = arr_vals - mean_val

    # Second moment m_2
    m2 = float(np.sum(z**2) / n)
    if m2 < 1e-14:
        raise ValueError("Values array has zero variance; cannot compute spatial autocorrelation.")

    # Spatial lag of centered values: z_lag = W @ z
    z_lag = w_mat @ z

    # Local Moran's statistic I_i = (z_i / m_2) * z_lag_i
    local_i = (z * z_lag) / m2

    # Unconditional raw quadrants
    # 1: HH (z >= 0, z_lag >= 0)
    # 2: LH (z < 0, z_lag >= 0)
    # 3: LL (z < 0, z_lag < 0)
    # 4: HL (z >= 0, z_lag < 0)
    raw_quads = np.zeros(n, dtype=np.int8)
    pos_z = z >= 0.0
    pos_lag = z_lag >= 0.0

    raw_quads[pos_z & pos_lag] = 1
    raw_quads[(~pos_z) & pos_lag] = 2
    raw_quads[(~pos_z) & (~pos_lag)] = 3
    raw_quads[pos_z & (~pos_lag)] = 4

    # Monte Carlo permutation inference
    rng = np.random.default_rng(seed)

    # Fast vectorized permutation generation: (n, n_permutations)
    permuted_z = np.empty((n, n_permutations), dtype=np.float64)
    for p in range(n_permutations):
        permuted_z[:, p] = rng.permutation(z)

    # Simulated lags: W @ permuted_z of shape (n, n_permutations)
    sim_lags = w_mat @ permuted_z
    # Simulated Local Moran statistics: (z[:, None] * sim_lags) / m2
    sim_i = (z[:, np.newaxis] * sim_lags) / m2

    # Empirical pseudo p-values:
    # Count how many permutations produced a statistic as extreme as or more extreme than observed
    larger = np.sum(sim_i >= local_i[:, np.newaxis], axis=1)
    smaller = n_permutations - larger

    # Two-sided pseudo p-value
    extreme_count = np.minimum(larger, smaller)
    p_values = (2.0 * (extreme_count + 1.0)) / (n_permutations + 1.0)
    p_values = np.clip(p_values, 0.0, 1.0)

    # Permutation-based z-scores
    sim_means = np.mean(sim_i, axis=1)
    sim_stds = np.std(sim_i, axis=1)
    safe_stds = np.where(sim_stds < 1e-12, 1.0, sim_stds)
    z_scores = (local_i - sim_means) / safe_stds

    # Significant quadrants (filter out observations where p > p_threshold)
    sig_mask = p_values <= p_threshold
    quadrants = np.where(sig_mask, raw_quads, np.int8(0))

    # Categorical string labels
    labels = np.array([_QUADRANT_LABELS[q] for q in quadrants], dtype=object)

    return LISAResult(
        local_moran=local_i,
        quadrants=quadrants,
        raw_quadrants=raw_quads,
        quadrant_labels=labels,
        p_values=p_values,
        z_scores=z_scores,
        p_threshold=p_threshold,
    )
