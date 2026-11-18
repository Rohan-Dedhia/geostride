"""Unit tests for SpatialKFold cross-validator and Scikit-Learn pipeline integration."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from geostride.features.lags import SpatialLagTransformer
from geostride.features.splitters import SpatialKFold


@pytest.fixture
def spatial_dataset():
    rng = np.random.default_rng(42)
    n = 100
    # Cluster points across 3 spatial regions
    centers = [(40.71, -74.00), (34.05, -118.24), (41.88, -87.63)]
    lats = []
    lons = []
    for c_lat, c_lon in centers:
        lats.extend(rng.normal(c_lat, 0.05, size=n // 3))
        lons.extend(rng.normal(c_lon, 0.05, size=n // 3))

    # Add remainder
    rem = n - len(lats)
    lats.extend(rng.normal(40.71, 0.05, size=rem))
    lons.extend(rng.normal(-74.00, 0.05, size=rem))

    df = pd.DataFrame(
        {
            "latitude": lats,
            "longitude": lons,
            "feature1": rng.standard_normal(n),
            "feature2": rng.standard_normal(n),
            "target": np.array(lats) * 2.0 + rng.standard_normal(n),
        }
    )
    return df


def test_spatial_kfold_basic_splits(spatial_dataset):
    n_splits = 4
    cv = SpatialKFold(n_splits=n_splits, method="morton")

    assert cv.get_n_splits(spatial_dataset) == n_splits

    all_test_indices = []
    for train_idx, test_idx in cv.split(spatial_dataset):
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        assert len(train_idx) + len(test_idx) == len(spatial_dataset)
        # Check no intersection
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        all_test_indices.extend(test_idx)

    # All indices should be tested exactly once
    assert sorted(all_test_indices) == list(range(len(spatial_dataset)))


def test_spatial_kfold_grid_method(spatial_dataset):
    cv = SpatialKFold(n_splits=3, method="grid")
    splits = list(cv.split(spatial_dataset))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(train_idx) + len(test_idx) == len(spatial_dataset)


def test_spatial_kfold_numpy_input(spatial_dataset):
    coords = spatial_dataset[["latitude", "longitude"]].to_numpy()
    cv = SpatialKFold(n_splits=3)
    splits = list(cv.split(coords))
    assert len(splits) == 3


def test_spatial_kfold_shuffle_reproducibility(spatial_dataset):
    cv1 = SpatialKFold(n_splits=3, shuffle=True, random_state=42)
    cv2 = SpatialKFold(n_splits=3, shuffle=True, random_state=42)
    cv3 = SpatialKFold(n_splits=3, shuffle=True, random_state=99)

    splits1 = [(list(tr), list(te)) for tr, te in cv1.split(spatial_dataset)]
    splits2 = [(list(tr), list(te)) for tr, te in cv2.split(spatial_dataset)]
    splits3 = [(list(tr), list(te)) for tr, te in cv3.split(spatial_dataset)]

    assert splits1 == splits2
    assert splits1 != splits3


def test_spatial_kfold_invalid_params(spatial_dataset):
    with pytest.raises(ValueError, match="n_splits must be an integer >= 2"):
        SpatialKFold(n_splits=1)

    with pytest.raises(ValueError, match="method must be one of"):
        SpatialKFold(method="unknown_method")

    with pytest.raises(ValueError, match="greater than the number of samples"):
        cv = SpatialKFold(n_splits=500)
        list(cv.split(spatial_dataset))

    with pytest.raises(ValueError, match="DataFrame must contain 'latitude' and 'longitude'"):
        bad_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        cv = SpatialKFold(n_splits=2)
        list(cv.split(bad_df))

    with pytest.raises(TypeError, match="X must be a pandas DataFrame or numpy array"):
        cv = SpatialKFold(n_splits=2)
        list(cv.split("not an array"))


def test_spatial_kfold_cross_val_score(spatial_dataset):
    # End-to-end integration with scikit-learn cross_val_score
    X = spatial_dataset[["latitude", "longitude", "feature1", "feature2"]]
    y = spatial_dataset["target"]

    cv = SpatialKFold(n_splits=3, lat_col="latitude", lon_col="longitude")
    scores = cross_val_score(Ridge(), X, y, cv=cv, scoring="r2")

    assert len(scores) == 3
    assert not np.isnan(scores).any()


def test_spatial_kfold_pipeline_integration(spatial_dataset):
    # Full Scikit-Learn Pipeline test with SpatialLagTransformer
    X = spatial_dataset[["latitude", "longitude", "feature1", "feature2"]]
    y = spatial_dataset["target"]

    # In a pipeline, SpatialKFold correctly partitions spatial blocks
    cv = SpatialKFold(n_splits=3, lat_col="latitude", lon_col="longitude")

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        # Fit SpatialLagTransformer on train and evaluate on test
        transformer = SpatialLagTransformer(
            lat_col="latitude",
            lon_col="longitude",
            value_col="feature1",
            k_neighbors=3,
        )
        train_lag = transformer.fit_transform(X_train)
        test_lag = transformer.transform(X_test)

        assert len(train_lag) == len(X_train)
        assert len(test_lag) == len(X_test)
        assert not np.isnan(train_lag).any()
        assert not np.isnan(test_lag).any()
