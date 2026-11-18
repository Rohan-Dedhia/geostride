"""Unit tests for GeoJSON serialization and parsing utilities."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geostride.io.geojson import (
    dataframe_to_geojson,
    dump_geojson,
    geojson_to_dataframe,
    load_geojson,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "latitude": [40.7128, 51.5074, 35.6762],
            "longitude": [-74.0060, -0.1278, 139.6503],
            "city": ["New York", "London", "Tokyo"],
            "population": [8336817, 8982000, 13960000],
        }
    )


def test_dataframe_to_geojson_rfc7946(sample_df):
    fc = dataframe_to_geojson(sample_df)

    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 3

    # Check RFC 7946 coordinate ordering: [longitude, latitude]
    feat0 = fc["features"][0]
    assert feat0["type"] == "Feature"
    assert feat0["geometry"]["type"] == "Point"
    assert feat0["geometry"]["coordinates"] == [-74.0060, 40.7128]

    # Check properties
    assert feat0["properties"]["city"] == "New York"
    assert feat0["properties"]["population"] == 8336817
    assert "latitude" not in feat0["properties"]
    assert "longitude" not in feat0["properties"]


def test_dataframe_to_geojson_property_filter(sample_df):
    fc = dataframe_to_geojson(sample_df, properties=["city"])
    assert fc["features"][0]["properties"] == {"city": "New York"}


def test_dataframe_to_geojson_missing_columns():
    df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    with pytest.raises(ValueError, match="DataFrame must contain 'latitude' and 'longitude'"):
        dataframe_to_geojson(df)


def test_dataframe_to_geojson_missing_properties(sample_df):
    with pytest.raises(ValueError, match="Requested properties missing"):
        dataframe_to_geojson(sample_df, properties=["non_existent_col"])


def test_dataframe_to_geojson_invalid_coordinates():
    df = pd.DataFrame({"latitude": [95.0], "longitude": [0.0]})
    with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
        dataframe_to_geojson(df)


def test_dataframe_to_geojson_with_nans():
    df = pd.DataFrame(
        {
            "latitude": [40.0],
            "longitude": [-73.0],
            "label": [None],
            "score": [np.nan],
        }
    )
    fc = dataframe_to_geojson(df)
    props = fc["features"][0]["properties"]
    assert props["label"] is None
    assert props["score"] is None


def test_geojson_to_dataframe_roundtrip(sample_df):
    fc = dataframe_to_geojson(sample_df)
    recovered_df = geojson_to_dataframe(fc)

    assert len(recovered_df) == 3
    assert list(recovered_df["city"]) == ["New York", "London", "Tokyo"]
    assert np.allclose(recovered_df["latitude"], sample_df["latitude"])
    assert np.allclose(recovered_df["longitude"], sample_df["longitude"])


def test_geojson_to_dataframe_from_json_string(sample_df):
    fc = dataframe_to_geojson(sample_df)
    json_str = json.dumps(fc)
    recovered_df = geojson_to_dataframe(json_str)

    assert len(recovered_df) == 3
    assert "city" in recovered_df.columns


def test_geojson_to_dataframe_polygon_centroid():
    polygon_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-74.0, 40.0],
                            [-74.0, 42.0],
                            [-72.0, 42.0],
                            [-72.0, 40.0],
                            [-74.0, 40.0],
                        ]
                    ],
                },
                "properties": {"name": "Test Polygon"},
            }
        ],
    }
    df = geojson_to_dataframe(polygon_geojson)
    assert len(df) == 1
    # Average of [-74, -74, -72, -72, -74] = -366 / 5 = -73.2
    # Average of [40, 42, 42, 40, 40] = 204 / 5 = 40.8
    assert np.isclose(df.iloc[0]["longitude"], -73.2)
    assert np.isclose(df.iloc[0]["latitude"], 40.8)
    assert df.iloc[0]["name"] == "Test Polygon"


def test_geojson_to_dataframe_invalid_inputs():
    with pytest.raises(ValueError, match="Invalid JSON string"):
        geojson_to_dataframe("{not valid json")

    with pytest.raises(ValueError, match="FeatureCollection"):
        geojson_to_dataframe({"type": "Point", "coordinates": [0, 0]})

    with pytest.raises(TypeError, match="'features' attribute must be a list"):
        geojson_to_dataframe({"type": "FeatureCollection", "features": "not a list"})

    with pytest.raises(ValueError, match="Feature at index 0 must be a dict"):
        geojson_to_dataframe({"type": "FeatureCollection", "features": ["invalid"]})


def test_dump_and_load_geojson(sample_df, tmp_path: Path):
    fc = dataframe_to_geojson(sample_df)
    file_path = tmp_path / "subdir" / "test_data.geojson"

    dump_geojson(fc, file_path)
    assert file_path.exists()

    loaded_fc = load_geojson(file_path)
    assert loaded_fc == fc


def test_load_geojson_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="GeoJSON file not found"):
        load_geojson(tmp_path / "nonexistent.geojson")
