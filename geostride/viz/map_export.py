"""Standalone interactive Leaflet HTML map exporter."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from geostride.core.coordinates import validate_coordinates


def render_html_map(
    coords: np.ndarray | Sequence[Sequence[float]],
    labels: Sequence[str] | np.ndarray | None = None,
    tooltips: Sequence[str] | None = None,
    title: str = "GeoStride Map",
    radius_px: int = 6,
) -> str:
    """Render a standalone interactive Leaflet HTML map string for given coordinates.

    The returned HTML string is completely self-contained and loads Leaflet.js
    and OpenStreetMap tiles via public CDNs. It requires no active backend server.

    Parameters
    ----------
    coords : np.ndarray or sequence of shape (N, 2)
        Array of (latitude, longitude) coordinate pairs.
    labels : sequence or np.ndarray of shape (N,), optional
        Label strings or cluster IDs associated with each coordinate point.
    tooltips : sequence of str of shape (N,), optional
        HTML-safe or text tooltip content displayed inside marker popups.
    title : str, default="GeoStride Map"
        HTML document title and map header.
    radius_px : int, default=6
        CircleMarker radius in pixels.

    Returns
    -------
    str
        Complete self-contained HTML document string.

    Raises
    ------
    ValueError
        If `coords` is empty or if lengths of `labels`/`tooltips` do not match `coords`.
    """
    arr = validate_coordinates(coords)
    n_points = len(arr)

    if n_points == 0:
        raise ValueError("Coordinate array cannot be empty.")

    if labels is not None:
        if len(labels) != n_points:
            raise ValueError(
                f"Length of labels ({len(labels)}) does not match coordinates ({n_points})."
            )
        labels_list = [str(lbl) for lbl in labels]
    else:
        labels_list = [f"Point #{i + 1}" for i in range(n_points)]

    if tooltips is not None:
        if len(tooltips) != n_points:
            raise ValueError(
                f"Length of tooltips ({len(tooltips)}) does not match coordinates ({n_points})."
            )
        tooltips_list = [str(t) for t in tooltips]
    else:
        tooltips_list = labels_list

    # Compute bounding box and center
    min_lat, min_lon = float(np.min(arr[:, 0])), float(np.min(arr[:, 1]))
    max_lat, max_lon = float(np.max(arr[:, 0])), float(np.max(arr[:, 1]))
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0

    # Prepare marker data as JSON for safe injection into JS
    markers_data = []
    for i in range(n_points):
        lat = float(arr[i, 0])
        lon = float(arr[i, 1])
        label = labels_list[i]
        tip = tooltips_list[i]
        popup_html = (
            f"<strong>{html.escape(label)}</strong><br>"
            f"<span style='color:#666;'>Lat: {lat:.5f}, Lon: {lon:.5f}</span>"
        )
        if tip and tip != label:
            popup_html += f"<hr style='margin:4px 0;border:0;border-top:1px solid #ddd;'><div>{html.escape(tip)}</div>"

        markers_data.append(
            {
                "lat": lat,
                "lon": lon,
                "popup": popup_html,
            }
        )

    markers_json = json.dumps(markers_data)
    escaped_title = html.escape(title)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{escaped_title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <style>
    html, body {{
      height: 100%;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    #header {{
      background: #1e293b;
      color: #ffffff;
      padding: 10px 18px;
      font-size: 16px;
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    #map {{
      width: 100%;
      height: calc(100% - 45px);
    }}
  </style>
</head>
<body>
  <div id="header">
    <span>{escaped_title}</span>
    <span style="font-size: 12px; font-weight: normal; opacity: 0.85;">{n_points} point{'s' if n_points != 1 else ''} rendered &bull; GeoStride</span>
  </div>
  <div id="map"></div>
  <script>
    const map = L.map('map');

    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }}).addTo(map);

    const points = {markers_json};
    const bounds = [];

    points.forEach(pt => {{
      const marker = L.circleMarker([pt.lat, pt.lon], {{
        radius: {radius_px},
        fillColor: '#2563eb',
        color: '#ffffff',
        weight: 1.5,
        opacity: 1,
        fillOpacity: 0.85
      }}).addTo(map);

      marker.bindPopup(pt.popup);
      bounds.push([pt.lat, pt.lon]);
    }});

    if (bounds.length > 1) {{
      map.fitBounds(bounds, {{ padding: [30, 30] }});
    }} else if (bounds.length === 1) {{
      map.setView([{center_lat}, {center_lon}], 13);
    }} else {{
      map.setView([0, 0], 2);
    }}
  </script>
</body>
</html>"""


def export_html_map(
    filepath: str | Path,
    coords: np.ndarray | Sequence[Sequence[float]],
    labels: Sequence[str] | np.ndarray | None = None,
    tooltips: Sequence[str] | None = None,
    title: str = "GeoStride Map",
    radius_px: int = 6,
) -> None:
    """Export an interactive Leaflet HTML map directly to a local file.

    Parameters
    ----------
    filepath : str or Path
        Target destination file path for the HTML map.
    coords : np.ndarray or sequence of shape (N, 2)
        Array of (latitude, longitude) coordinate pairs.
    labels : sequence or np.ndarray of shape (N,), optional
        Label strings or cluster IDs associated with each coordinate point.
    tooltips : sequence of str of shape (N,), optional
        HTML-safe or text tooltip content displayed inside marker popups.
    title : str, default="GeoStride Map"
        HTML document title and map header.
    radius_px : int, default=6
        CircleMarker radius in pixels.
    """
    html_content = render_html_map(
        coords=coords,
        labels=labels,
        tooltips=tooltips,
        title=title,
        radius_px=radius_px,
    )
    dest = Path(filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(html_content)
