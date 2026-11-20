"""Pure-Python zero-dependency Well-Known Text (WKT) parser and serializer.

Supports OGC WKT standards for POINT, LINESTRING, POLYGON (with hole rings),
and MULTIPOINT geometries with bidirectional conversion for Pandas DataFrames.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from geostride.core.coordinates import validate_coordinates


def _clean_num(val: float) -> str:
    """Format floating point numbers without trailing zeroes."""
    formatted = f"{float(val):.8f}".rstrip("0").rstrip(".")
    if formatted == "-0" or formatted == "":
        return "0"
    return formatted


def _fmt_coord(lat: float, lon: float) -> str:
    """Format coordinate pair into WKT order [longitude latitude]."""
    return f"{_clean_num(lon)} {_clean_num(lat)}"


def to_wkt(geometry: np.ndarray | Sequence[Any], geom_type: str = "auto") -> str:
    """Serialize a coordinate array or collection into an OGC Well-Known Text (WKT) string.

    Parameters
    ----------
    geometry : np.ndarray or Sequence[Any]
        Input coordinates in GeoStride order [latitude, longitude].
        - For POINT: 1D sequence of [latitude, longitude].
        - For LINESTRING: 2D array or sequence of shape (N, 2).
        - For MULTIPOINT: 2D array or sequence of shape (N, 2).
        - For POLYGON: 2D array of exterior ring vertices of shape (N, 2), or a list of rings
          where geometry[0] is the exterior ring and geometry[1:] are interior hole rings.
    geom_type : str, default="auto"
        Geometry type: "auto", "POINT", "LINESTRING", "POLYGON", or "MULTIPOINT" (case-insensitive).

    Returns
    -------
    str
        Standard OGC Well-Known Text (WKT) string with coordinates ordered as [longitude, latitude].

    Raises
    ------
    ValueError
        If geom_type is invalid or geometry dimensions are incompatible.
    """
    gtype = geom_type.upper().strip()

    # Empty geometry check
    if hasattr(geometry, "__len__") and len(geometry) == 0:
        if gtype == "AUTO":
            gtype = "GEOMETRYCOLLECTION"
        return f"{gtype} EMPTY"

    # Auto-detection
    if gtype == "AUTO":
        if isinstance(geometry, (list, tuple)) and len(geometry) > 0 and isinstance(geometry[0], (list, tuple, np.ndarray)) and hasattr(geometry[0], "__len__") and len(geometry[0]) > 0 and isinstance(geometry[0][0], (list, tuple, np.ndarray)):
            # List of rings -> POLYGON
            gtype = "POLYGON"
        else:
            arr = np.asarray(geometry, dtype=float)
            if arr.ndim == 1 and arr.shape[0] == 2:
                gtype = "POINT"
            elif arr.ndim == 2 and arr.shape[1] == 2:
                gtype = "LINESTRING"
            else:
                raise ValueError(
                    f"Cannot automatically infer geometry type for input with shape {arr.shape}."
                )

    if gtype == "POINT":
        arr = np.asarray(geometry, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError(f"POINT coordinates must have 2 elements [lat, lon]; got shape {arr.shape}.")
            lat, lon = arr[0], arr[1]
        elif arr.ndim == 2 and arr.shape == (1, 2):
            lat, lon = arr[0, 0], arr[0, 1]
        else:
            raise ValueError(f"POINT geometry must have shape (2,) or (1, 2); got {arr.shape}.")
        return f"POINT ({_fmt_coord(lat, lon)})"

    elif gtype == "LINESTRING":
        arr = np.asarray(geometry, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"LINESTRING coordinates must have shape (N, 2); got {arr.shape}.")
        if len(arr) < 2:
            raise ValueError(f"LINESTRING must contain at least 2 vertices; got {len(arr)}.")
        coords_str = ", ".join(_fmt_coord(lat, lon) for lat, lon in arr)
        return f"LINESTRING ({coords_str})"

    elif gtype == "MULTIPOINT":
        arr = np.asarray(geometry, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"MULTIPOINT coordinates must have shape (N, 2); got {arr.shape}.")
        pts_str = ", ".join(f"({_fmt_coord(lat, lon)})" for lat, lon in arr)
        return f"MULTIPOINT ({pts_str})"

    elif gtype == "POLYGON":
        # Handle single ring or list of rings
        rings: list[np.ndarray] = []
        if isinstance(geometry, (list, tuple)) and len(geometry) > 0 and isinstance(geometry[0], (list, tuple, np.ndarray)) and hasattr(geometry[0], "__len__") and len(geometry[0]) > 0 and isinstance(geometry[0][0], (list, tuple, np.ndarray)):
            for r in geometry:
                ring_arr = np.asarray(r, dtype=float)
                if ring_arr.ndim != 2 or ring_arr.shape[1] != 2:
                    raise ValueError(f"Polygon ring must have shape (N, 2); got {ring_arr.shape}.")
                rings.append(ring_arr)
        else:
            arr = np.asarray(geometry, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"POLYGON coordinates must have shape (N, 2) or list of rings; got {arr.shape}.")
            rings.append(arr)

        serialized_rings: list[str] = []
        for ring in rings:
            if len(ring) < 3:
                raise ValueError(f"Polygon ring must have at least 3 vertices; got {len(ring)}.")
            # Ensure closed ring
            if not np.allclose(ring[0], ring[-1]):
                ring = np.vstack([ring, ring[0]])
            ring_str = ", ".join(_fmt_coord(lat, lon) for lat, lon in ring)
            serialized_rings.append(f"({ring_str})")

        return f"POLYGON ({', '.join(serialized_rings)})"

    else:
        raise ValueError(
            f"Unsupported geometry type '{geom_type}'. Supported types: 'POINT', 'LINESTRING', 'POLYGON', 'MULTIPOINT'."
        )


def from_wkt(wkt_str: str) -> tuple[str, np.ndarray | list[np.ndarray]]:
    """Parse an OGC Well-Known Text (WKT) string into geometry type and coordinate matrices.

    Parameters
    ----------
    wkt_str : str
        OGC WKT string representation (e.g., 'POINT (-74.006 40.7128)').

    Returns
    -------
    geom_type : str
        Canonical uppercase geometry type ("POINT", "LINESTRING", "POLYGON", or "MULTIPOINT").
    coordinates : np.ndarray or list of np.ndarray
        - For "POINT": 1D array [latitude, longitude].
        - For "LINESTRING" or "MULTIPOINT": 2D array of shape (N, 2) with columns [latitude, longitude].
        - For "POLYGON": List of 2D arrays where index 0 is the exterior ring and indices 1+ are hole rings.

    Raises
    ------
    ValueError
        If the WKT string contains invalid syntax or unsupported geometry types.
    """
    if not isinstance(wkt_str, str):
        raise TypeError(f"WKT input must be a string, got {type(wkt_str).__name__}")

    s = wkt_str.strip()
    if not s:
        raise ValueError("Cannot parse empty WKT string.")

    # Match geometry type token
    m = re.match(r"^([A-Za-z]+)\s*(.*)$", s, re.DOTALL)
    if not m:
        raise ValueError(f"Invalid WKT format: '{wkt_str}'")

    geom_type_raw = m.group(1).upper()
    body = m.group(2).strip()

    valid_types = {"POINT", "LINESTRING", "POLYGON", "MULTIPOINT", "GEOMETRYCOLLECTION"}
    if geom_type_raw not in valid_types:
        raise ValueError(
            f"Unsupported WKT geometry type '{geom_type_raw}'. Supported types: 'POINT', 'LINESTRING', 'POLYGON', 'MULTIPOINT'."
        )

    if body.upper() == "EMPTY":
        if geom_type_raw == "POLYGON":
            return geom_type_raw, []
        elif geom_type_raw == "POINT":
            return geom_type_raw, np.empty((0,), dtype=float)
        else:
            return geom_type_raw, np.empty((0, 2), dtype=float)

    if not (body.startswith("(") and body.endswith(")")):
        raise ValueError(f"Invalid WKT syntax: coordinates must be enclosed in parentheses; got '{body}'")

    # Helper to parse a comma-separated list of coordinate pairs "lon lat, lon lat"
    def _parse_coord_list(coord_text: str) -> np.ndarray:
        parts = [p.strip() for p in coord_text.split(",") if p.strip()]
        coords: list[list[float]] = []
        for p in parts:
            tokens = p.split()
            if len(tokens) != 2:
                raise ValueError(f"Expected 2 coordinate values (lon lat); got '{p}'")
            try:
                lon = float(tokens[0])
                lat = float(tokens[1])
            except ValueError as err:
                raise ValueError(f"Non-numeric coordinates in WKT: '{p}'") from err
            coords.append([lat, lon])
        return np.asarray(coords, dtype=float)

    if geom_type_raw == "POINT":
        inner = body[1:-1].strip()
        tokens = inner.split()
        if len(tokens) != 2:
            raise ValueError(f"POINT requires exactly 2 coordinate tokens (lon lat); got '{inner}'")
        try:
            lon = float(tokens[0])
            lat = float(tokens[1])
        except ValueError as err:
            raise ValueError(f"Non-numeric POINT coordinates: '{inner}'") from err
        return "POINT", np.array([lat, lon], dtype=float)

    elif geom_type_raw == "LINESTRING":
        inner = body[1:-1].strip()
        arr = _parse_coord_list(inner)
        if len(arr) < 2:
            raise ValueError(f"LINESTRING requires at least 2 coordinate pairs; got {len(arr)}")
        return "LINESTRING", arr

    elif geom_type_raw == "MULTIPOINT":
        inner = body[1:-1].strip()
        # MULTIPOINT can be formatted as ((lon lat), (lon lat)) or (lon lat, lon lat)
        ring_matches = re.findall(r"\(([^()]+)\)", inner)
        if ring_matches:
            coords: list[list[float]] = []
            for item in ring_matches:
                tokens = item.strip().split()
                if len(tokens) != 2:
                    raise ValueError(f"Expected 2 coordinate values (lon lat); got '{item}'")
                lon = float(tokens[0])
                lat = float(tokens[1])
                coords.append([lat, lon])
            return "MULTIPOINT", np.asarray(coords, dtype=float)
        else:
            arr = _parse_coord_list(inner)
            return "MULTIPOINT", arr

    elif geom_type_raw == "POLYGON":
        # Extract each ring enclosed in parentheses
        inner = body[1:-1].strip()
        ring_matches = re.findall(r"\(([^()]+)\)", inner)
        if not ring_matches:
            raise ValueError(f"Invalid POLYGON syntax: expected nested rings '((...))'; got '{body}'")

        rings: list[np.ndarray] = []
        for ring_text in ring_matches:
            ring_coords = _parse_coord_list(ring_text)
            if len(ring_coords) < 3:
                raise ValueError(f"Polygon ring must have at least 3 vertices; got {len(ring_coords)}")
            # Auto-close ring if necessary
            if not np.allclose(ring_coords[0], ring_coords[-1]):
                ring_coords = np.vstack([ring_coords, ring_coords[0]])
            rings.append(ring_coords)

        return "POLYGON", rings

    return geom_type_raw, np.empty((0, 2), dtype=float)


def dataframe_to_wkt(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    wkt_col: str = "wkt",
) -> pd.DataFrame:
    """Add an OGC Well-Known Text (WKT) POINT column to a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing coordinate columns.
    lat_col : str, default="latitude"
        Column name for latitude.
    lon_col : str, default="longitude"
        Column name for longitude.
    wkt_col : str, default="wkt"
        Name of the newly created WKT column.

    Returns
    -------
    pd.DataFrame
        New DataFrame with the added WKT column.

    Raises
    ------
    ValueError
        If lat_col or lon_col are not found or contain values outside valid geographic ranges.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{lat_col}' and '{lon_col}' columns; got {list(df.columns)}"
        )

    lats = df[lat_col].to_numpy(dtype=float)
    lons = df[lon_col].to_numpy(dtype=float)

    validate_coordinates(np.column_stack([lats, lons]))

    wkt_series = [f"POINT ({_fmt_coord(lat, lon)})" for lat, lon in zip(lats, lons, strict=True)]

    out_df = df.copy()
    out_df[wkt_col] = wkt_series
    return out_df


def wkt_to_dataframe(
    df: pd.DataFrame,
    wkt_col: str = "wkt",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """Extract latitude and longitude columns from a WKT POINT column in a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a WKT column.
    wkt_col : str, default="wkt"
        Column name containing WKT POINT strings.
    lat_col : str, default="latitude"
        Output column name for extracted latitudes.
    lon_col : str, default="longitude"
        Output column name for extracted longitudes.

    Returns
    -------
    pd.DataFrame
        New DataFrame with extracted latitude and longitude columns.

    Raises
    ------
    ValueError
        If wkt_col is missing or contains non-POINT or malformed geometries.
    """
    if wkt_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{wkt_col}' column; got {list(df.columns)}")

    lats: list[float] = []
    lons: list[float] = []

    for idx, val in enumerate(df[wkt_col]):
        if not isinstance(val, str):
            raise TypeError(f"Row {idx}: Expected WKT string in column '{wkt_col}', got {type(val).__name__}")
        geom_type, coords = from_wkt(val)
        if geom_type != "POINT":
            raise ValueError(f"Row {idx}: Expected WKT POINT geometry, got '{geom_type}'")
        if not isinstance(coords, np.ndarray) or coords.shape != (2,):
            raise ValueError(f"Row {idx}: Invalid POINT coordinates: {coords}")
        lats.append(float(coords[0]))
        lons.append(float(coords[1]))

    out_df = df.copy()
    out_df[lat_col] = lats
    out_df[lon_col] = lons
    return out_df
