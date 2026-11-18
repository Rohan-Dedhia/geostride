"""Spatial lag calculation using Inverse Distance Weighting (IDW) and K-Nearest Neighbors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from geostride.core.coordinates import validate_coordinates
from geostride.metrics.haversine import haversine_matrix


def spatial_lag_knn(
    coords: np.ndarray | Sequence[Sequence[float]],
    values: np.ndarray | Sequence[float] | pd.Series,
    k: int = 5,
    method: str = "idw",
    power: float = 1.0,
    epsilon: float = 1e-5,
    query_coords: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Compute spatial lag of a target attribute over K-nearest spatial neighbors.

    For each observation point, the spatial lag represents the locally aggregated or
    distance-weighted average value of neighboring observations.

    Parameters
    ----------
    coords : array-like of shape (N, 2)
        Source reference coordinates [lat, lon].
    values : array-like of shape (N,)
        Attribute values associated with each source coordinate.
    k : int, default=5
        Number of nearest neighbors to aggregate.
    method : {"idw", "mean", "gaussian"}, default="idw"
        Weighting scheme:
        - "idw": Inverse distance weighting with decay `1 / (d + epsilon)^power`.
        - "mean": Uniform unweighted average of the k nearest neighbors.
        - "gaussian": Gaussian radial basis function `exp(-d^2 / (2 * sigma^2))`.
    power : float, default=1.0
        Distance decay exponent for IDW.
    epsilon : float, default=1e-5
        Small positive constant added to distance in IDW to avoid division by zero.
    query_coords : array-like of shape (M, 2), optional
        Optional destination evaluation coordinates. If None, in-sample spatial lag is
        computed for `coords` with self-neighbor exclusion.

    Returns
    -------
    np.ndarray of shape (N,) or (M,)
        Computed spatial lag values.

    Raises
    ------
    ValueError
        If inputs have mismatched dimensions, invalid k, or unknown method.
    """
    ref_coords = validate_coordinates(coords)
    val_arr = np.asarray(values, dtype=np.float64).ravel()

    n_ref = ref_coords.shape[0]
    if val_arr.shape[0] != n_ref:
        raise ValueError(
            f"Length mismatch: coords has {n_ref} rows, but values has {val_arr.shape[0]} elements."
        )

    if np.any(np.isnan(val_arr)):
        raise ValueError("Values array contains NaN.")

    is_in_sample = query_coords is None
    if is_in_sample:
        target_coords = ref_coords
        # In-sample requires at least k + 1 observations because self is excluded
        if n_ref <= k:
            raise ValueError(
                f"In-sample spatial lag requires more reference points than k; "
                f"got N={n_ref} and k={k}. Need N > k."
            )
    else:
        target_coords = validate_coordinates(query_coords)
        if n_ref < k:
            raise ValueError(
                f"Reference coordinates has {n_ref} points, but k={k} neighbors requested."
            )

    if k < 1:
        raise ValueError(f"k must be at least 1; got {k}")

    # Compute full pairwise Haversine distance matrix between target and reference
    # Shape: (M, N) where M is target_coords.shape[0] and N is ref_coords.shape[0]
    dist_matrix = haversine_matrix(target_coords, ref_coords, unit="km")

    if is_in_sample:
        # Prevent self from being chosen as its own neighbor by setting diagonal to infinity
        np.fill_diagonal(dist_matrix, np.inf)

    # For each row, find indices of the k smallest distances
    # np.argpartition is O(N) compared to full O(N log N) sort
    neighbor_indices = np.argpartition(dist_matrix, kth=k - 1, axis=1)[:, :k]

    # Gather neighbor distances and attribute values for each target
    row_idx = np.arange(target_coords.shape[0])[:, np.newaxis]
    neighbor_dists = dist_matrix[row_idx, neighbor_indices]  # Shape: (M, k)
    neighbor_vals = val_arr[neighbor_indices]  # Shape: (M, k)

    method_clean = method.lower().strip()
    if method_clean == "mean":
        weights = np.ones_like(neighbor_dists, dtype=np.float64)
    elif method_clean == "idw":
        weights = 1.0 / np.power(neighbor_dists + epsilon, power)
    elif method_clean == "gaussian":
        # Estimate bandwidth sigma as mean neighbor distance
        sigma = np.mean(neighbor_dists)
        if sigma <= 0:
            sigma = 1.0
        weights = np.exp(-(neighbor_dists**2) / (2.0 * sigma**2))
    else:
        raise ValueError(
            f"Unknown spatial lag method '{method}'. Supported: 'idw', 'mean', 'gaussian'."
        )

    # Normalize weights so each row sums to 1.0
    weight_sums = np.sum(weights, axis=1, keepdims=True)
    # Guard against zero sum
    weight_sums = np.where(weight_sums == 0.0, 1.0, weight_sums)
    norm_weights = weights / weight_sums

    # Weighted sum: sum(w_ij * y_j)
    lags = np.sum(norm_weights * neighbor_vals, axis=1)
    return lags


class SpatialLagTransformer:
    """Scikit-Learn compatible transformer for spatial lag feature extraction.

    Fits on training geographic coordinates and attributes, then transforms in-sample
    or out-of-sample data by calculating distance-weighted neighbor statistics.

    Parameters
    ----------
    k_neighbors : int, default=5
        Number of spatial neighbors to evaluate.
    method : {"idw", "mean", "gaussian"}, default="idw"
        Weighting method.
    power : float, default=1.0
        Distance decay power for IDW.
    epsilon : float, default=1e-5
        Distance smoothing parameter for IDW.
    lat_col : str, default="latitude"
        DataFrame column name for latitude.
    lon_col : str, default="longitude"
        DataFrame column name for longitude.
    value_col : str, default="value"
        DataFrame column name for the attribute to spatially lag.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        method: str = "idw",
        power: float = 1.0,
        epsilon: float = 1e-5,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        value_col: str = "value",
    ) -> None:
        self.k_neighbors = k_neighbors
        self.method = method
        self.power = power
        self.epsilon = epsilon
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.value_col = value_col

        self.coords_train_: np.ndarray | None = None
        self.values_train_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> SpatialLagTransformer:
        """Fit the transformer by memorizing the training reference spatial index.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            If DataFrame, must contain columns `lat_col`, `lon_col`, and `value_col`.
            If ndarray, columns 0, 1 are coordinates and column 2 is attribute value.
        y : None
            Ignored. Follows scikit-learn convention.

        Returns
        -------
        SpatialLagTransformer
            Fitted instance.
        """
        if isinstance(X, pd.DataFrame):
            coords = X[[self.lat_col, self.lon_col]].to_numpy()
            vals = X[self.value_col].to_numpy()
        else:
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError(
                    "Expected 2D array with at least 3 columns (lat, lon, value) for fitting."
                )
            coords = arr[:, :2]
            vals = arr[:, 2]

        self.coords_train_ = validate_coordinates(coords)
        self.values_train_ = np.asarray(vals, dtype=np.float64).ravel()
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform input coordinates into spatial lag feature values.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Dataset containing coordinates to evaluate against training reference points.

        Returns
        -------
        np.ndarray of shape (N,)
            Spatial lag values.
        """
        if self.coords_train_ is None or self.values_train_ is None:
            raise RuntimeError("SpatialLagTransformer instance is not fitted yet. Call 'fit' first.")

        if isinstance(X, pd.DataFrame):
            coords = X[[self.lat_col, self.lon_col]].to_numpy()
        else:
            arr = np.asarray(X, dtype=np.float64)
            coords = arr[:, :2] if arr.ndim == 2 and arr.shape[1] >= 2 else arr

        target_coords = validate_coordinates(coords)

        # Check if transforming the identical training set (in-sample)
        is_identical = (
            target_coords.shape == self.coords_train_.shape
            and np.allclose(target_coords, self.coords_train_, atol=1e-9)
        )

        query = None if is_identical else target_coords

        return spatial_lag_knn(
            coords=self.coords_train_,
            values=self.values_train_,
            k=self.k_neighbors,
            method=self.method,
            power=self.power,
            epsilon=self.epsilon,
            query_coords=query,
        )

    def fit_transform(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> np.ndarray:
        """Fit to data, then transform it in-sample.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training dataset.
        y : None
            Ignored.

        Returns
        -------
        np.ndarray of shape (N,)
            Computed in-sample spatial lag values.
        """
        return self.fit(X, y).transform(X)
