"""Unit tests for spatial lag calculations and SpatialLagTransformer."""

import numpy as np
import pandas as pd
import pytest

from geostride.features.lags import SpatialLagTransformer, spatial_lag_knn


def test_spatial_lag_knn_mean_uniform():
    # 3 points along the equator: 0 deg, 1 deg, 2 deg longitude
    coords = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, 3.0],
    ])
    values = np.array([10.0, 20.0, 30.0, 40.0])

    # For k=1:
    # Point 0's closest neighbor is Point 1 (val 20.0)
    # Point 3's closest neighbor is Point 2 (val 30.0)
    lags = spatial_lag_knn(coords, values, k=1, method="mean")
    assert np.isclose(lags[0], 20.0)
    assert np.isclose(lags[3], 30.0)

    # For k=2:
    # Point 1's closest neighbors are Point 0 (val 10) and Point 2 (val 30) -> mean = 20.0
    lags_k2 = spatial_lag_knn(coords, values, k=2, method="mean")
    assert np.isclose(lags_k2[1], 20.0)


def test_spatial_lag_knn_idw():
    # Point at (0, 0) with value 100
    # Point at (0, 1) with value 0
    # Point at (0, 10) with value 50
    coords = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 10.0],
    ])
    values = np.array([100.0, 0.0, 50.0])

    # From perspective of Point 1 (0, 1):
    # Point 0 is ~111 km away (distance 1 deg)
    # Point 2 is ~1000 km away (distance 9 deg)
    # Under IDW, Point 0 should have much higher weight than Point 2
    lags = spatial_lag_knn(coords, values, k=2, method="idw", power=1.0)
    assert lags[1] > 40.0  # Heavily pulled toward Point 0's value (100) rather than Point 2 (50)


def test_spatial_lag_query_coords_out_of_sample():
    ref_coords = np.array([
        [40.71, -74.00],
        [40.72, -73.99],
        [40.75, -73.98],
    ])
    ref_vals = np.array([100.0, 200.0, 300.0])

    # Query point directly on top of point 0
    query = np.array([[40.71, -74.00]])
    lag = spatial_lag_knn(ref_coords, ref_vals, k=1, query_coords=query)
    assert np.isclose(lag[0], 100.0, atol=1e-3)


def test_spatial_lag_gaussian():
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    values = np.array([1.0, 2.0, 3.0])
    lags = spatial_lag_knn(coords, values, k=2, method="gaussian")
    assert len(lags) == 3
    assert np.all(np.isfinite(lags))


def test_spatial_lag_transformer_dataframe():
    df = pd.DataFrame({
        "latitude": [40.71, 40.72, 40.75, 40.78],
        "longitude": [-74.00, -73.99, -73.98, -73.96],
        "price": [1200000.0, 1350000.0, 980000.0, 1100000.0],
    })

    transformer = SpatialLagTransformer(
        k_neighbors=2,
        method="idw",
        lat_col="latitude",
        lon_col="longitude",
        value_col="price",
    )

    lags = transformer.fit_transform(df)
    assert len(lags) == 4
    assert np.all(lags > 0.0)

    # Test out-of-sample transform
    df_new = pd.DataFrame({
        "latitude": [40.715],
        "longitude": [-73.995],
    })
    lag_new = transformer.transform(df_new)
    assert len(lag_new) == 1
    assert 1000000.0 < lag_new[0] < 1400000.0


def test_spatial_lag_transformer_unfitted_raises():
    transformer = SpatialLagTransformer()
    with pytest.raises(RuntimeError, match="not fitted yet"):
        transformer.transform(np.zeros((2, 2)))


def test_spatial_lag_validation_errors():
    coords = np.array([[0.0, 0.0], [0.0, 1.0]])
    values = np.array([1.0, 2.0])

    # k >= N for in-sample
    with pytest.raises(ValueError, match="requires more reference points than k"):
        spatial_lag_knn(coords, values, k=2)

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        spatial_lag_knn(coords, values[:1], k=1)

    # Unknown method
    with pytest.raises(ValueError, match="Unknown spatial lag method"):
        spatial_lag_knn(coords, values, k=1, method="quantum_magic")

    # NaN in values
    with pytest.raises(ValueError, match="contains NaN"):
        spatial_lag_knn(coords, [1.0, np.nan], k=1)
