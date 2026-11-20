"""Spherical Kernel Density Estimation (KDE) and continuous spatial heatmap rasterization."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from geostride.core.bbox import BoundingBox
from geostride.core.coordinates import validate_coordinates
from geostride.metrics.haversine import haversine_matrix, haversine_pairwise

KernelType = Literal["gaussian", "epanechnikov", "quartic"]


class SphericalKDE:
    """Continuous spherical kernel density estimator and heatmap rasterizer.

    Computes smooth probability density functions on the Earth's spherical surface
    using exact great-circle geodesic distances and 2D normalized kernel profiles.

    Parameters
    ----------
    bandwidth_km : float or {"silverman", "scott"}, default="silverman"
        Smoothing bandwidth parameter in kilometers, or automated estimation rule.
    kernel : {"gaussian", "epanechnikov", "quartic"}, default="gaussian"
        Kernel weighting function:
        - "gaussian": bell-shaped infinite support kernel.
        - "epanechnikov": parabolic compact support kernel (optimal in MSE).
        - "quartic": biweight compact support smooth polynomial kernel.

    Attributes
    ----------
    bandwidth_km_ : float
        Fitted smoothing bandwidth in kilometers.
    coords_ : np.ndarray of shape (N, 2)
        Fitted sample coordinates [latitude, longitude].
    weights_ : np.ndarray of shape (N,)
        Normalized observation weights summing to 1.0.

    Examples
    --------
    >>> import numpy as np
    >>> from geostride.features import SphericalKDE
    >>> pts = np.array([[40.71, -74.00], [40.72, -73.99], [40.73, -73.98]])
    >>> kde = SphericalKDE(bandwidth_km=2.0, kernel="gaussian")
    >>> kde.fit(pts)
    SphericalKDE(kernel='gaussian', bandwidth_km=2.000)
    >>> densities = kde.evaluate([[40.715, -73.995]])
    >>> densities.shape
    (1,)
    """

    def __init__(
        self,
        bandwidth_km: float | Literal["silverman", "scott"] = "silverman",
        kernel: KernelType = "gaussian",
    ) -> None:
        if isinstance(bandwidth_km, (int, float)):
            if bandwidth_km <= 0.0:
                raise ValueError(f"bandwidth_km must be strictly positive; got {bandwidth_km}")
        elif bandwidth_km not in ("silverman", "scott"):
            raise ValueError(
                f"Invalid bandwidth rule '{bandwidth_km}'. Must be a positive float, "
                f"'silverman', or 'scott'."
            )

        valid_kernels = {"gaussian", "epanechnikov", "quartic"}
        if kernel not in valid_kernels:
            raise ValueError(f"Invalid kernel '{kernel}'. Supported kernels: {sorted(valid_kernels)}")

        self.bandwidth_km = bandwidth_km
        self.kernel = kernel

        self.coords_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bandwidth_km_: float | None = None

    def fit(
        self,
        coords: np.ndarray | Sequence[Sequence[float]],
        weights: np.ndarray | Sequence[float] | None = None,
    ) -> SphericalKDE:
        """Fit the spherical KDE model on geographic sample coordinates.

        Parameters
        ----------
        coords : array-like of shape (N, 2)
            Sample coordinate points [latitude, longitude].
        weights : array-like of shape (N,), optional
            Sample weights. If None, equal weights (1/N) are assigned.

        Returns
        -------
        SphericalKDE
            Fitted estimator instance.
        """
        arr_coords = validate_coordinates(coords)
        n = len(arr_coords)
        if n == 0:
            raise ValueError("Cannot fit KDE on empty coordinates array.")

        if weights is None:
            w = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            w_arr = np.asarray(weights, dtype=np.float64).ravel()
            if len(w_arr) != n:
                raise ValueError(
                    f"Weights length ({len(w_arr)}) must match coordinates length ({n})."
                )
            if np.any(w_arr < 0.0):
                raise ValueError("Sample weights must be non-negative.")
            w_sum = float(np.sum(w_arr))
            if w_sum <= 0.0:
                raise ValueError("Sum of sample weights must be strictly positive.")
            w = w_arr / w_sum

        self.coords_ = arr_coords
        self.weights_ = w

        # Determine bandwidth in kilometers
        if isinstance(self.bandwidth_km, (int, float)):
            self.bandwidth_km_ = float(self.bandwidth_km)
        else:
            self.bandwidth_km_ = self._estimate_bandwidth(arr_coords, self.bandwidth_km)

        return self

    def _estimate_bandwidth(self, coords: np.ndarray, rule: str) -> float:
        """Estimate optimal spatial bandwidth using spherical standard distance."""
        n = len(coords)
        if n < 2:
            return 1.0

        # Mean center
        mean_lat = float(np.mean(coords[:, 0]))
        mean_lon = float(np.mean(coords[:, 1]))

        # Geodesic distances to center
        dists = haversine_pairwise(coords[:, 0], coords[:, 1], mean_lat, mean_lon, unit="km")
        if isinstance(dists, float):
            dists = np.array([dists])

        sigma = float(np.sqrt(np.mean(dists**2)))
        if sigma < 1e-6:
            return 1.0  # Identical or near-identical coordinates

        if rule == "scott":
            # Scott's rule for 2D density: sigma * n^(-1/6)
            return float(sigma * (n ** (-1.0 / 6.0)))

        # Silverman's rule of thumb (robust with IQR)
        q75, q25 = np.percentile(dists, [75, 25])
        iqr = float(q75 - q25)
        a = min(sigma, iqr / 1.34) if iqr > 1e-6 else sigma
        # 2D Silverman factor: A * (2 / (3 * n))^(1/6)
        h = a * ((2.0 / (3.0 * n)) ** (1.0 / 6.0))
        return float(max(1e-4, h))

    def evaluate(
        self,
        eval_coords: np.ndarray | Sequence[Sequence[float]],
        batch_size: int = 1000,
    ) -> np.ndarray:
        """Evaluate continuous surface density at arbitrary geographic query coordinates.

        Parameters
        ----------
        eval_coords : array-like of shape (M, 2)
            Query coordinates [latitude, longitude].
        batch_size : int, default=1000
            Number of query points to evaluate per batch for memory efficiency.

        Returns
        -------
        np.ndarray of shape (M,)
            Estimated density values (probability per square kilometer).
        """
        if self.coords_ is None or self.weights_ is None or self.bandwidth_km_ is None:
            raise RuntimeError("Model is not fitted. Call 'fit' before 'evaluate'.")

        eval_pts = validate_coordinates(eval_coords)
        m = len(eval_pts)
        if m == 0:
            return np.empty(0, dtype=np.float64)

        densities = np.empty(m, dtype=np.float64)
        h = self.bandwidth_km_
        h2 = h * h

        # Precompute kernel constants
        # In 2D: integral over 2*pi*r*K(r/h) dr from 0 to inf is 1.0
        if self.kernel == "gaussian":
            norm_const = 1.0 / (2.0 * math.pi * h2)
        elif self.kernel == "epanechnikov":
            norm_const = 2.0 / (math.pi * h2)
        elif self.kernel == "quartic":
            norm_const = 3.0 / (math.pi * h2)
        else:
            raise ValueError(f"Unknown kernel '{self.kernel}'")

        weights = self.weights_
        sample_pts = self.coords_

        for start_idx in range(0, m, batch_size):
            end_idx = min(start_idx + batch_size, m)
            batch_eval = eval_pts[start_idx:end_idx]

            # Pairwise distance matrix of shape (batch_len, n_samples)
            dist_mat = haversine_matrix(batch_eval, sample_pts, unit="km")
            u = dist_mat / h

            if self.kernel == "gaussian":
                k_vals = norm_const * np.exp(-0.5 * (u**2))
            elif self.kernel == "epanechnikov":
                k_vals = np.where(u <= 1.0, norm_const * (1.0 - (u**2)), 0.0)
            elif self.kernel == "quartic":
                k_vals = np.where(u <= 1.0, norm_const * ((1.0 - (u**2)) ** 2), 0.0)

            # Weighted sum over sample points
            densities[start_idx:end_idx] = np.dot(k_vals, weights)

        return densities

    def rasterize(
        self,
        bbox: BoundingBox | None = None,
        resolution: tuple[int, int] = (100, 100),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rasterize the continuous density surface into a regular 2D grid matrix.

        Ideal for spatial heatmaps, choropleths, and raster visualization.

        Parameters
        ----------
        bbox : BoundingBox, optional
            Geographic bounding box for the raster grid. If None, bounds are inferred
            from the fitted sample coordinates and padded by the smoothing bandwidth.
        resolution : tuple of (int, int), default=(100, 100)
            Grid dimensions `(n_lat_bins, n_lon_bins)`.

        Returns
        -------
        lat_grid : np.ndarray of shape (n_lat_bins,)
            1D array of latitude grid coordinates.
        lon_grid : np.ndarray of shape (n_lon_bins,)
            1D array of longitude grid coordinates.
        density_matrix : np.ndarray of shape (n_lat_bins, n_lon_bins)
            2D density values across the grid.
        """
        if self.coords_ is None or self.bandwidth_km_ is None:
            raise RuntimeError("Model is not fitted. Call 'fit' before 'rasterize'.")

        n_lat, n_lon = resolution
        if n_lat < 2 or n_lon < 2:
            raise ValueError(f"resolution must be at least (2, 2); got {resolution}")

        if bbox is None:
            # Infer bounds padded by 1.5 * bandwidth
            h_deg = self.bandwidth_km_ / 111.0  # approximate degree conversion
            pad_lat = max(0.01, h_deg * 1.5)
            pad_lon = max(0.01, h_deg * 1.5)
            base_box = BoundingBox.from_coordinates(self.coords_)
            bbox = base_box.expand(delta_lat=pad_lat, delta_lon=pad_lon)

        lats = np.linspace(bbox.min_lat, bbox.max_lat, n_lat)
        lons = np.linspace(bbox.min_lon, bbox.max_lon, n_lon)

        # Generate meshgrid query coordinates
        grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
        query_pts = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])

        densities_flat = self.evaluate(query_pts)
        density_matrix = densities_flat.reshape(n_lat, n_lon)

        return lats, lons, density_matrix

    def __repr__(self) -> str:
        bw = f"{self.bandwidth_km_:.3f}" if self.bandwidth_km_ is not None else str(self.bandwidth_km)
        return f"SphericalKDE(kernel='{self.kernel}', bandwidth_km={bw})"
