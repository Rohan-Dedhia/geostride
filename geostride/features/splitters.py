"""Spatial cross-validation splitters for Scikit-Learn pipelines."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import BaseCrossValidator
except ImportError:  # pragma: no cover
    class BaseCrossValidator:
        """Fallback base class when scikit-learn is not installed."""

from geostride.core.coordinates import validate_coordinates
from geostride.indexing.morton import encode_coordinates_morton


class SpatialKFold(BaseCrossValidator):
    """Spatial K-Fold cross-validator for Scikit-Learn models and pipelines.

    Standard random K-Fold cross-validation suffers from severe optimistic bias
    in geospatial datasets because neighboring observations often share high
    spatial autocorrelation, resulting in data leakage between training and test sets.

    SpatialKFold partitions coordinates into spatially contiguous clusters (using
    2D Morton space-filling curves or spatial grid blocking), ensuring that test folds
    are geographically isolated from training observations.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    method : {"morton", "grid"}, default="morton"
        Spatial blocking algorithm:
        - "morton": Groups points ordered along a 2D Morton (Z-order) space-filling curve.
        - "grid": Partitions geographic bounding box into a 2D regular spatial grid.
    lat_col : str, default="latitude"
        DataFrame column name for latitude.
    lon_col : str, default="longitude"
        DataFrame column name for longitude.
    shuffle : bool, default=False
        Whether to shuffle the spatial fold labels prior to splitting.
    random_state : int or RandomState, optional
        Random seed for fold label permutation when `shuffle=True`.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.model_selection import cross_val_score
    >>> from geostride.features.splitters import SpatialKFold
    >>> df = pd.DataFrame({
    ...     'latitude': [40.71, 40.72, 40.75, 40.78, 40.79],
    ...     'longitude': [-74.00, -73.99, -73.98, -73.96, -73.95],
    ...     'feature': [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     'target': [10.0, 12.0, 15.0, 19.0, 22.0]
    ... })
    >>> cv = SpatialKFold(n_splits=3, lat_col='latitude', lon_col='longitude')
    >>> for train_idx, test_idx in cv.split(df):
    ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        method: str = "morton",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        shuffle: bool = False,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError(f"n_splits must be an integer >= 2; got {n_splits}")

        valid_methods = {"morton", "grid"}
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}; got '{method}'")

        self.n_splits = n_splits
        self.method = method
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.shuffle = shuffle
        self.random_state = random_state

    def _extract_coords(self, X: Any) -> np.ndarray:
        """Extract [latitude, longitude] coordinates array from input X."""
        if isinstance(X, pd.DataFrame):
            if self.lat_col not in X.columns or self.lon_col not in X.columns:
                raise ValueError(
                    f"DataFrame must contain '{self.lat_col}' and '{self.lon_col}' columns; got {list(X.columns)}"
                )
            coords = X[[self.lat_col, self.lon_col]].to_numpy()
        elif isinstance(X, (np.ndarray, list)):
            arr = np.asarray(X)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(
                    f"Coordinate array must be 2D with at least 2 columns [lat, lon]; got shape {arr.shape}"
                )
            coords = arr[:, :2]
        else:
            raise TypeError(f"X must be a pandas DataFrame or numpy array; got {type(X).__name__}")

        return validate_coordinates(coords)

    def _compute_fold_ids(self, coords: np.ndarray) -> np.ndarray:
        """Assign each coordinate to a spatial fold index [0, n_splits - 1]."""
        n_samples = len(coords)
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than the number of samples: {n_samples}"
            )

        fold_ids = np.zeros(n_samples, dtype=np.int64)

        if self.method == "morton":
            # Encode coordinates to 2D Morton codes
            morton_codes = encode_coordinates_morton(coords)
            sorted_indices = np.argsort(morton_codes)

            # Assign contiguous chunks along the space-filling curve to distinct folds
            for rank, idx in enumerate(sorted_indices):
                fold_ids[idx] = min(rank * self.n_splits // n_samples, self.n_splits - 1)

        elif self.method == "grid":
            # Compute bounding box
            min_lat, min_lon = float(np.min(coords[:, 0])), float(np.min(coords[:, 1]))
            max_lat, max_lon = float(np.max(coords[:, 0])), float(np.max(coords[:, 1]))

            lat_range = max(max_lat - min_lat, 1e-6)
            lon_range = max(max_lon - min_lon, 1e-6)

            # Choose grid dimensions nx, ny such that nx * ny >= n_splits
            k = int(np.ceil(np.sqrt(self.n_splits)))
            norm_lat = np.clip((coords[:, 0] - min_lat) / lat_range, 0.0, 0.999999)
            norm_lon = np.clip((coords[:, 1] - min_lon) / lon_range, 0.0, 0.999999)

            grid_x = (norm_lon * k).astype(np.int64)
            grid_y = (norm_lat * k).astype(np.int64)
            cell_ids = grid_y * k + grid_x

            # Map cell_ids to [0, n_splits - 1]
            unique_cells = np.unique(cell_ids)
            cell_to_fold = {cell: i % self.n_splits for i, cell in enumerate(unique_cells)}
            for i, cell in enumerate(cell_ids):
                fold_ids[i] = cell_to_fold[cell]

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            perm = rng.permutation(self.n_splits)
            fold_ids = perm[fold_ids]

        return fold_ids

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Training data containing coordinates.
        y : array-like of shape (n_samples,), optional
            Target values (ignored).
        groups : array-like of shape (n_samples,), optional
            Group labels for samples (ignored).

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        coords = self._extract_coords(X)
        fold_ids = self._compute_fold_ids(coords)

        all_indices = np.arange(len(coords))
        for fold in range(self.n_splits):
            test_mask = fold_ids == fold
            test_indices = all_indices[test_mask]
            train_indices = all_indices[~test_mask]
            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Return the number of splitting iterations in the cross-validator."""
        return self.n_splits
