"""Pure-Python zero-dependency GeoJSON serializer and parser for Pandas DataFrames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from geostride.core.coordinates import Coordinate


def dataframe_to_geojson(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    properties: list[str] | None = None,
) -> dict[str, Any]:
    """Serialize a Pandas DataFrame containing geographic coordinates into a GeoJSON FeatureCollection.

    Complies with RFC 7946 coordinate ordering [longitude, latitude].

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing coordinate columns.
    lat_col : str, default="latitude"
        Column name for latitude.
    lon_col : str, default="longitude"
        Column name for longitude.
    properties : list of str, optional
        Subset of column names to serialize into feature properties. If None, all
        columns other than `lat_col` and `lon_col` are included.

    Returns
    -------
    dict
        GeoJSON FeatureCollection object.

    Raises
    ------
    ValueError
        If coordinate columns are missing or values fall outside geographic bounds.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{lat_col}' and '{lon_col}' columns; got {list(df.columns)}"
        )

    if properties is None:
        prop_cols = [c for c in df.columns if c not in (lat_col, lon_col)]
    else:
        missing = [c for c in properties if c not in df.columns]
        if missing:
            raise ValueError(f"Requested properties missing from DataFrame: {missing}")
        prop_cols = properties

    features: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        # Validate bounds via Coordinate class
        coord = Coordinate(latitude=lat, longitude=lon)

        # Build feature properties converting non-JSON native types
        props: dict[str, Any] = {}
        for col in prop_cols:
            val = row[col]
            if pd.isna(val):
                props[col] = None
            elif hasattr(val, "item"):
                props[col] = val.item()
            else:
                props[col] = val

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [coord.longitude, coord.latitude],  # RFC 7946: [lon, lat]
            },
            "properties": props,
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def geojson_to_dataframe(
    geojson_data: dict[str, Any] | str,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """Parse a GeoJSON FeatureCollection dictionary or JSON string into a flat Pandas DataFrame.

    Parameters
    ----------
    geojson_data : dict or str
        GeoJSON FeatureCollection dictionary or raw JSON string.
    lat_col : str, default="latitude"
        Output column name for extracted latitude coordinates.
    lon_col : str, default="longitude"
        Output column name for extracted longitude coordinates.

    Returns
    -------
    pd.DataFrame
        DataFrame with coordinates and extracted feature properties.

    Raises
    ------
    ValueError
        If the data is not a valid GeoJSON FeatureCollection.
    """
    if isinstance(geojson_data, str):
        try:
            data = json.loads(geojson_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
    else:
        data = geojson_data

    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON object must have 'type': 'FeatureCollection'")

    features = data.get("features", [])
    if not isinstance(features, list):
        raise TypeError("GeoJSON 'features' attribute must be a list")

    rows: list[dict[str, Any]] = []

    for idx, feat in enumerate(features):
        if not isinstance(feat, dict) or feat.get("type") != "Feature":
            raise ValueError(f"Feature at index {idx} must be a dict with 'type': 'Feature'")

        geom = feat.get("geometry")
        if not geom or not isinstance(geom, dict):
            continue

        geom_type = geom.get("type")
        coords = geom.get("coordinates")

        if geom_type == "Point" and coords and len(coords) >= 2:
            lon = float(coords[0])
            lat = float(coords[1])
        elif geom_type == "Polygon" and coords and len(coords) > 0 and len(coords[0]) > 0:
            # Approximate Polygon by its centroid
            pts = coords[0]
            lon = float(sum(p[0] for p in pts) / len(pts))
            lat = float(sum(p[1] for p in pts) / len(pts))
        else:
            continue

        row: dict[str, Any] = {
            lat_col: lat,
            lon_col: lon,
        }

        props = feat.get("properties")
        if isinstance(props, dict):
            row.update(props)

        rows.append(row)

    return pd.DataFrame(rows)


def dump_geojson(
    geojson_dict: dict[str, Any],
    filepath: str | Path,
    indent: int = 2,
) -> None:
    """Write a GeoJSON dictionary directly to disk formatted as UTF-8 JSON.

    Parameters
    ----------
    geojson_dict : dict
        GeoJSON dictionary to serialize.
    filepath : str or Path
        Destination file path.
    indent : int, default=2
        JSON indentation formatting level.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f, indent=indent, ensure_ascii=False)


def load_geojson(filepath: str | Path) -> dict[str, Any]:
    """Read and parse a GeoJSON file from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to the GeoJSON file.

    Returns
    -------
    dict
        Parsed GeoJSON dictionary.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
