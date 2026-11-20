"""Unit tests for vectorized spatial join (sjoin) operations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geostride import spatial_join


def test_spatial_join_within_inner() -> None:
    """Test inner spatial join using predicate='within'."""
    pts_df = pd.DataFrame({
        "point_id": [1, 2, 3],
        "city": ["Downtown", "Midtown", "FarAway"],
        "latitude": [40.71, 40.75, 41.50],
        "longitude": [-74.00, -73.98, -73.00],
    })

    # Two distinct polygons
    poly_downtown = np.array([
        [40.70, -74.02],
        [40.73, -74.02],
        [40.73, -73.98],
        [40.70, -73.98],
    ])
    poly_midtown = np.array([
        [40.74, -74.00],
        [40.77, -74.00],
        [40.77, -73.96],
        [40.74, -73.96],
    ])

    polys = [poly_downtown, poly_midtown]

    joined = spatial_join(pts_df, polys, how="inner", predicate="within", poly_id_col="district_id")

    assert len(joined) == 2
    assert set(joined["point_id"]) == {1, 2}

    row1 = joined[joined["point_id"] == 1].iloc[0]
    assert row1["district_id"] == 0
    assert row1["city"] == "Downtown"

    row2 = joined[joined["point_id"] == 2].iloc[0]
    assert row2["district_id"] == 1
    assert row2["city"] == "Midtown"


def test_spatial_join_within_left() -> None:
    """Test left spatial join preserving unmatched rows."""
    pts_df = pd.DataFrame({
        "point_id": [10, 20],
        "latitude": [40.71, 42.00],
        "longitude": [-74.00, -72.00],
    })

    poly = np.array([
        [40.70, -74.02],
        [40.73, -74.02],
        [40.73, -73.98],
        [40.70, -73.98],
    ])

    joined = spatial_join(pts_df, [poly], how="left", predicate="within")

    assert len(joined) == 2
    row_matched = joined[joined["point_id"] == 10].iloc[0]
    assert row_matched["polygon_id"] == 0

    row_unmatched = joined[joined["point_id"] == 20].iloc[0]
    assert pd.isna(row_unmatched["polygon_id"])


def test_spatial_join_geojson_properties() -> None:
    """Test merging GeoJSON feature properties during spatial join."""
    pts_df = pd.DataFrame({
        "station_id": ["S1"],
        "latitude": [10.5],
        "longitude": [20.5],
    })

    geojson_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "ZONE_A",
                "properties": {
                    "zone_name": "Commercial",
                    "speed_limit": 50,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [20.0, 10.0],
                            [21.0, 10.0],
                            [21.0, 11.0],
                            [20.0, 11.0],
                            [20.0, 10.0],
                        ]
                    ],
                },
            }
        ],
    }

    res = spatial_join(pts_df, geojson_collection, how="inner", predicate="within")
    assert len(res) == 1
    assert res.iloc[0]["polygon_id"] == "ZONE_A"
    assert res.iloc[0]["zone_name"] == "Commercial"
    assert res.iloc[0]["speed_limit"] == 50


def test_spatial_join_dict_polygons() -> None:
    """Test passing dictionary mapping polygon names to geometries."""
    pts_df = pd.DataFrame({
        "latitude": [0.5, 5.5],
        "longitude": [0.5, 5.5],
    })

    polys = {
        "Sector_Alpha": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "Sector_Beta": [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0]],
    }

    res = spatial_join(pts_df, polys, how="inner", predicate="within")
    assert len(res) == 2
    assert set(res["polygon_id"]) == {"Sector_Alpha", "Sector_Beta"}


def test_spatial_join_nearest_predicate() -> None:
    """Test predicate='nearest' matching to closest polygon with distance."""
    pts_df = pd.DataFrame({
        "latitude": [40.70, 40.79],
        "longitude": [-74.00, -73.95],
    })

    poly1 = [[40.71, -74.01], [40.73, -74.01], [40.73, -73.99], [40.71, -73.99]]
    poly2 = [[40.80, -73.96], [40.82, -73.96], [40.82, -73.94], [40.80, -73.94]]

    res = spatial_join(pts_df, [poly1, poly2], predicate="nearest", distance_col="dist_km")
    assert len(res) == 2
    assert "dist_km" in res.columns
    assert res.iloc[0]["polygon_id"] == 0
    assert res.iloc[1]["polygon_id"] == 1
    assert res.iloc[0]["dist_km"] >= 0.0

    # Max distance threshold test
    res_thresh = spatial_join(
        pts_df,
        [poly1, poly2],
        how="inner",
        predicate="nearest",
        max_distance_km=0.5,
    )
    # The points might exceed 0.5 km
    assert len(res_thresh) <= len(res)


def test_spatial_join_one_to_many_overlap() -> None:
    """Test point falling in two overlapping polygons generates 2 joined rows."""
    pts_df = pd.DataFrame({
        "id": [1],
        "latitude": [0.5],
        "longitude": [0.5],
    })

    # Both squares cover (0.5, 0.5)
    polyA = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    polyB = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]

    res = spatial_join(pts_df, [polyA, polyB], how="inner")
    assert len(res) == 2
    assert set(res["polygon_id"]) == {0, 1}


def test_spatial_join_empty_inputs_and_errors() -> None:
    """Test validation and edge cases for spatial_join."""
    df_empty = pd.DataFrame(columns=["latitude", "longitude"])
    res_empty = spatial_join(df_empty, [])
    assert len(res_empty) == 0

    df_valid = pd.DataFrame({"latitude": [10.0], "longitude": [20.0]})
    # Empty polygons list
    res_no_polys = spatial_join(df_valid, [], how="left")
    assert len(res_no_polys) == 1
    assert pd.isna(res_no_polys.iloc[0]["polygon_id"])

    # Missing coordinate columns
    with pytest.raises(ValueError, match="left_df must contain"):
        spatial_join(pd.DataFrame({"x": [1]}), [])

    # Invalid how
    with pytest.raises(ValueError, match="Invalid how"):
        spatial_join(df_valid, [], how="outer")

    # Invalid predicate
    with pytest.raises(ValueError, match="Invalid predicate"):
        spatial_join(df_valid, [], predicate="touches")
