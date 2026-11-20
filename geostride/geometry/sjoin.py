"""Vectorized spatial join (sjoin) for point-in-polygon and proximity matching."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from geostride.core.bbox import BoundingBox
from geostride.core.coordinates import validate_coordinates
from geostride.geometry.pip import _parse_polygon_rings, point_in_polygon
from geostride.indexing.rtree import SpatialRTree, _haversine_km


def _extract_polygons(
    polygons: Sequence[Any] | dict[str, Any] | pd.DataFrame,
    poly_id_col: str = "polygon_id",
) -> list[tuple[Any, np.ndarray, list[np.ndarray], BoundingBox, dict[str, Any]]]:
    """Extract and validate normalized polygon entries from diverse input formats.

    Returns a list of tuples:
    `(poly_id, exterior_ring, holes, bounding_box, properties_dict)`
    """
    extracted: list[tuple[Any, np.ndarray, list[np.ndarray], BoundingBox, dict[str, Any]]] = []

    if isinstance(polygons, dict) and polygons.get("type") == "FeatureCollection":
        features = polygons.get("features", [])
        for idx, feat in enumerate(features):
            geom = feat.get("geometry", {})
            props = dict(feat.get("properties") or {})
            poly_id = feat.get("id", props.get(poly_id_col, idx))
            ext, holes = _parse_polygon_rings(geom)
            bbox = BoundingBox(
                min_lat=float(np.min(ext[:, 0])),
                max_lat=float(np.max(ext[:, 0])),
                min_lon=float(np.min(ext[:, 1])),
                max_lon=float(np.max(ext[:, 1])),
            )
            extracted.append((poly_id, ext, holes, bbox, props))
    elif isinstance(polygons, dict):
        # Dictionary mapping ID -> polygon geometry
        for poly_id, geom in polygons.items():
            ext, holes = _parse_polygon_rings(geom)
            bbox = BoundingBox(
                min_lat=float(np.min(ext[:, 0])),
                max_lat=float(np.max(ext[:, 0])),
                min_lon=float(np.min(ext[:, 1])),
                max_lon=float(np.max(ext[:, 1])),
            )
            extracted.append((poly_id, ext, holes, bbox, {}))
    elif isinstance(polygons, (list, tuple)):
        for idx, item in enumerate(polygons):
            props = {}
            poly_id = idx
            if isinstance(item, dict) and item.get("type") == "Feature":
                geom = item.get("geometry", {})
                props = dict(item.get("properties") or {})
                poly_id = item.get("id", props.get(poly_id_col, idx))
                ext, holes = _parse_polygon_rings(geom)
            else:
                ext, holes = _parse_polygon_rings(item)

            bbox = BoundingBox(
                min_lat=float(np.min(ext[:, 0])),
                max_lat=float(np.max(ext[:, 0])),
                min_lon=float(np.min(ext[:, 1])),
                max_lon=float(np.max(ext[:, 1])),
            )
            extracted.append((poly_id, ext, holes, bbox, props))
    elif isinstance(polygons, pd.DataFrame):
        geom_col = "geometry" if "geometry" in polygons.columns else "polygon"
        if geom_col not in polygons.columns:
            raise ValueError(
                f"Polygons DataFrame must contain a '{geom_col}' column; got {list(polygons.columns)}"
            )
        for idx, row in polygons.iterrows():
            geom = row[geom_col]
            props = {k: v for k, v in row.items() if k != geom_col}
            poly_id = row.get(poly_id_col, idx)
            ext, holes = _parse_polygon_rings(geom)
            bbox = BoundingBox(
                min_lat=float(np.min(ext[:, 0])),
                max_lat=float(np.max(ext[:, 0])),
                min_lon=float(np.min(ext[:, 1])),
                max_lon=float(np.max(ext[:, 1])),
            )
            extracted.append((poly_id, ext, holes, bbox, props))
    else:
        raise TypeError(f"Unsupported polygons container type: {type(polygons)}")

    return extracted


def spatial_join(
    left_df: pd.DataFrame,
    polygons: Sequence[Any] | dict[str, Any] | pd.DataFrame,
    how: str = "inner",
    predicate: str = "within",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    poly_id_col: str = "polygon_id",
    distance_col: str | None = None,
    max_distance_km: float | None = None,
) -> pd.DataFrame:
    """Perform a high-throughput spatial join between a DataFrame of coordinates and polygons.

    Accelerated by hierarchical R-Tree bounding box indexing and pure-NumPy vectorized
    ray-casting containment tests.

    Parameters
    ----------
    left_df : pd.DataFrame
        Input DataFrame containing point coordinates.
    polygons : Sequence, dict, or GeoJSON FeatureCollection
        Collection of polygon boundaries. Can be:
        - List/sequence of 2D polygon vertex arrays.
        - Dict mapping polygon IDs to geometries `{poly_id: geometry}`.
        - GeoJSON FeatureCollection dictionary with properties.
        - DataFrame containing a 'geometry' column.
    how : {"inner", "left"}, default="inner"
        Join topology:
        - "inner": Retain only coordinate points matching at least one polygon.
        - "left": Retain all rows from left_df; fill unmatched polygon columns with NaN/None.
    predicate : {"within", "intersects", "nearest"}, default="within"
        Spatial relationship operator:
        - "within": Coordinate point lies strictly inside or on the polygon boundary.
        - "intersects": Coordinate point lies within the polygon's bounding box.
        - "nearest": Matches each coordinate point to its closest polygon.
    lat_col : str, default="latitude"
        Column name in `left_df` for latitude coordinates.
    lon_col : str, default="longitude"
        Column name in `left_df` for longitude coordinates.
    poly_id_col : str, default="polygon_id"
        Name of the polygon identifier column in the output DataFrame.
    distance_col : str, optional
        Column name to store geodesic distance in km (populated when predicate="nearest").
    max_distance_km : float, optional
        Maximum threshold distance in kilometers for "nearest" matching.

    Returns
    -------
    pd.DataFrame
        Joined DataFrame containing original left columns, `poly_id_col`, and any
        polygon feature properties.

    Raises
    ------
    ValueError
        If coordinate columns are missing, how/predicate options are invalid, or
        coordinates contain NaN/out-of-bounds values.

    Examples
    --------
    >>> import pandas as pd
    >>> from geostride.geometry import spatial_join
    >>> df = pd.DataFrame({
    ...     "point_id": [1, 2],
    ...     "latitude": [40.7128, 40.7580],
    ...     "longitude": [-74.0060, -73.9855],
    ... })
    >>> district = [[40.70, -74.02], [40.73, -74.02], [40.73, -73.99], [40.70, -73.99]]
    >>> res = spatial_join(df, [district], how="inner", predicate="within")
    >>> len(res)
    1
    """
    if lat_col not in left_df.columns or lon_col not in left_df.columns:
        raise ValueError(
            f"left_df must contain '{lat_col}' and '{lon_col}' columns; got {list(left_df.columns)}"
        )

    norm_how = how.lower()
    if norm_how not in ("inner", "left"):
        raise ValueError(f"Invalid how '{how}'. Supported join types are: 'inner', 'left'")

    norm_pred = predicate.lower()
    if norm_pred not in ("within", "intersects", "nearest"):
        raise ValueError(
            f"Invalid predicate '{predicate}'. Supported predicates: 'within', 'intersects', 'nearest'"
        )

    poly_entries = _extract_polygons(polygons, poly_id_col=poly_id_col)
    num_polys = len(poly_entries)

    # Empty inputs handling
    if len(left_df) == 0 or num_polys == 0:
        if norm_how == "inner":
            cols = list(left_df.columns) + [poly_id_col]
            if distance_col:
                cols.append(distance_col)
            return pd.DataFrame(columns=cols)
        else:
            out = left_df.copy()
            out[poly_id_col] = None
            if distance_col:
                out[distance_col] = None
            return out

    # Validate coordinate points
    raw_coords = left_df[[lat_col, lon_col]].values
    valid_coords = validate_coordinates(raw_coords)
    n_pts = len(valid_coords)

    # Build R-Tree index over polygon bounding boxes
    rtree = SpatialRTree()
    rtree.bulk_load(
        ids=list(range(num_polys)),
        bboxes=[entry[3] for entry in poly_entries],
    )

    matches: list[tuple[int, int, float | None]] = []  # (pt_idx, poly_idx, distance_km)

    if norm_pred in ("within", "intersects"):
        for pt_idx in range(n_pts):
            lat = float(valid_coords[pt_idx, 0])
            lon = float(valid_coords[pt_idx, 1])

            # Query candidate polygons overlapping point coordinate
            cand_poly_indices = rtree.query_range((lat, lat, lon, lon))
            if not cand_poly_indices:
                continue

            for poly_idx in cand_poly_indices:
                poly_id, ext, holes, bbox, _ = poly_entries[poly_idx]
                if norm_pred == "intersects":
                    # Bounding box intersection already confirmed by R-Tree
                    matches.append((pt_idx, poly_idx, None))
                else:
                    # Exact point-in-polygon ray-casting test
                    poly_geom = [ext] + holes if holes else ext
                    if point_in_polygon([lat, lon], poly_geom, include_boundary=True)[0]:
                        matches.append((pt_idx, poly_idx, None))

    elif norm_pred == "nearest":
        # Nearest neighbor join
        for pt_idx in range(n_pts):
            lat = float(valid_coords[pt_idx, 0])
            lon = float(valid_coords[pt_idx, 1])

            nearest_indices = rtree.query_nearest(lat, lon, k=1)
            if not nearest_indices:
                continue

            poly_idx = nearest_indices[0]
            # Compute distance from point to polygon bounding box / centroid
            poly_id, ext, _, bbox, _ = poly_entries[poly_idx]
            c_lat = (bbox.min_lat + bbox.max_lat) / 2.0
            c_lon = (bbox.min_lon + bbox.max_lon) / 2.0
            dist_km = _haversine_km(lat, lon, c_lat, c_lon)

            if max_distance_km is not None and dist_km > max_distance_km:
                continue

            matches.append((pt_idx, poly_idx, dist_km))

    # Construct result DataFrame
    # Collect all unique property keys from polygons
    all_prop_keys: list[str] = []
    seen_keys: set[str] = set()
    for _, _, _, _, props in poly_entries:
        for k in props:
            if k not in seen_keys and k != poly_id_col and k not in left_df.columns:
                seen_keys.add(k)
                all_prop_keys.append(k)

    dist_field_name = distance_col if distance_col else ("distance_km" if norm_pred == "nearest" else None)

    if norm_how == "inner":
        if not matches:
            cols = list(left_df.columns) + [poly_id_col] + all_prop_keys
            if dist_field_name:
                cols.append(dist_field_name)
            return pd.DataFrame(columns=cols)

        out_rows = []
        for pt_idx, poly_idx, d_km in matches:
            poly_id, _, _, _, props = poly_entries[poly_idx]
            row_dict = dict(left_df.iloc[pt_idx])
            row_dict[poly_id_col] = poly_id
            for k in all_prop_keys:
                row_dict[k] = props.get(k)
            if dist_field_name:
                row_dict[dist_field_name] = d_km
            out_rows.append(row_dict)

        return pd.DataFrame(out_rows).reset_index(drop=True)

    else:
        # Left join: preserve all left_df points
        matched_dict: dict[int, list[tuple[int, float | None]]] = {}
        for pt_idx, poly_idx, d_km in matches:
            matched_dict.setdefault(pt_idx, []).append((poly_idx, d_km))

        out_rows = []
        for pt_idx in range(n_pts):
            row_base = dict(left_df.iloc[pt_idx])
            if pt_idx in matched_dict:
                for poly_idx, d_km in matched_dict[pt_idx]:
                    poly_id, _, _, _, props = poly_entries[poly_idx]
                    row_dict = dict(row_base)
                    row_dict[poly_id_col] = poly_id
                    for k in all_prop_keys:
                        row_dict[k] = props.get(k)
                    if dist_field_name:
                        row_dict[dist_field_name] = d_km
                    out_rows.append(row_dict)
            else:
                row_dict = dict(row_base)
                row_dict[poly_id_col] = None
                for k in all_prop_keys:
                    row_dict[k] = None
                if dist_field_name:
                    row_dict[dist_field_name] = None
                out_rows.append(row_dict)

        return pd.DataFrame(out_rows).reset_index(drop=True)
