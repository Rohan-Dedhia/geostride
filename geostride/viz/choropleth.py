"""Standalone interactive Leaflet Choropleth HTML map exporter."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from geostride.core.coordinates import validate_coordinates

COLORMAPS: dict[str, list[str]] = {
    "viridis": ["#440154", "#3b528b", "#21908d", "#5dc963", "#fde725"],
    "plasma": ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636"],
    "blues": ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"],
    "reds": ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"],
    "greens": ["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"],
    "coolwarm": ["#3b4cc0", "#8cb2e9", "#f2f2f2", "#f49a7b", "#b40426"],
}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(rgb: tuple[int, int, int] | Sequence[float]) -> str:
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def _get_palette_colors(cmap_name: str, n_bins: int) -> list[str]:
    """Retrieve or interpolate n_bins hex colors from a named color palette."""
    base_hex = COLORMAPS.get(cmap_name.lower().strip())
    if base_hex is None:
        valid_options = ", ".join(COLORMAPS.keys())
        raise ValueError(
            f"Unsupported colormap '{cmap_name}'. Choose from: {valid_options}."
        )

    if n_bins == len(base_hex):
        return base_hex

    base_rgb = np.array([_hex_to_rgb(c) for c in base_hex])
    old_x = np.linspace(0.0, 1.0, len(base_hex))
    new_x = np.linspace(0.0, 1.0, n_bins)
    interp_r = np.interp(new_x, old_x, base_rgb[:, 0])
    interp_g = np.interp(new_x, old_x, base_rgb[:, 1])
    interp_b = np.interp(new_x, old_x, base_rgb[:, 2])
    return [_rgb_to_hex((r, g, b)) for r, g, b in zip(interp_r, interp_g, interp_b, strict=True)]


def _assign_bins(
    values: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Assign values into n_bins quantile or equal-interval ranges."""
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if np.isclose(vmin, vmax) or len(values) < n_bins:
        indices = np.zeros(len(values), dtype=int)
        edges = np.array([vmin, vmax])
        labels = [f"{vmin:.2f}"]
        return indices, edges, labels

    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    edges = np.percentile(values, quantiles)
    if len(np.unique(edges)) < n_bins + 1:
        edges = np.linspace(vmin, vmax, n_bins + 1)

    indices = np.digitize(values, edges[1:-1])
    indices = np.clip(indices, 0, n_bins - 1)
    labels = [f"{edges[i]:.2f} \u2013 {edges[i+1]:.2f}" for i in range(n_bins)]
    return indices, edges, labels


def render_choropleth_map(
    polygons: Sequence[Any] | Mapping[str, Any],
    values: Sequence[float] | np.ndarray,
    colormap: str = "viridis",
    n_bins: int = 5,
    title: str = "Choropleth Map",
    region_names: Sequence[str] | None = None,
    opacity: float = 0.75,
) -> str:
    """Render a standalone interactive Leaflet Choropleth HTML map string.

    Parameters
    ----------
    polygons : Sequence of array-like or Mapping of str to array-like
        Collection of polygon coordinate arrays [[lat, lon], ...]. If a mapping is passed,
        keys are used as region names.
    values : Sequence of float or np.ndarray of shape (N,)
        Numeric statistical values corresponding to each polygon region.
    colormap : {"viridis", "plasma", "blues", "reds", "greens", "coolwarm"}, default="viridis"
        Color ramp for choropleth thematic shading.
    n_bins : int, default=5
        Number of value classification bins for color mapping and legend swatches.
    title : str, default="Choropleth Map"
        HTML page title and map legend header.
    region_names : Sequence of str, optional
        Names or identifiers for each polygon region displayed in tooltips and popups.
    opacity : float, default=0.75
        Polygon fill opacity in [0.0, 1.0].

    Returns
    -------
    str
        Self-contained HTML document string with embedded Leaflet script and styling.

    Raises
    ------
    ValueError
        If polygons is empty, values length does not match polygons, or n_bins < 1.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1; got {n_bins}")

    # Extract polygon coordinates and names
    poly_list: list[np.ndarray] = []
    extracted_names: list[str] = []

    if isinstance(polygons, Mapping):
        for k, v in polygons.items():
            arr = validate_coordinates(v)
            if len(arr) < 3:
                raise ValueError(f"Polygon '{k}' must have at least 3 vertices; got {len(arr)}.")
            poly_list.append(arr)
            extracted_names.append(str(k))
    else:
        for idx, item in enumerate(polygons):
            arr = validate_coordinates(item)
            if len(arr) < 3:
                raise ValueError(f"Polygon index {idx} must have at least 3 vertices; got {len(arr)}.")
            poly_list.append(arr)
            extracted_names.append(f"Region #{idx + 1}")

    n_polys = len(poly_list)
    if n_polys == 0:
        raise ValueError("Polygon collection cannot be empty.")

    vals_arr = np.asarray(values, dtype=float)
    if len(vals_arr) != n_polys:
        raise ValueError(
            f"Length mismatch: {n_polys} polygons provided but values has length {len(vals_arr)}."
        )

    if region_names is not None:
        if len(region_names) != n_polys:
            raise ValueError(
                f"Length mismatch: {n_polys} polygons provided but region_names has length {len(region_names)}."
            )
        names = [str(n) for n in region_names]
    else:
        names = extracted_names

    # Determine colors and binning
    palette = _get_palette_colors(colormap, n_bins)
    bin_indices, _edges, bin_labels = _assign_bins(vals_arr, n_bins=len(palette))

    # Construct GeoJSON FeatureCollection
    features: list[dict[str, Any]] = []
    for i in range(n_polys):
        ring = poly_list[i]
        # Auto-close ring for GeoJSON standard
        if not np.allclose(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[0]])

        # GeoJSON coordinates order is [longitude, latitude]
        geojson_coords = [[float(lon), float(lat)] for lat, lon in ring]
        b_idx = int(bin_indices[i])
        fill_color = palette[min(b_idx, len(palette) - 1)]
        val = float(vals_arr[i])
        val_str = f"{val:.4f}".rstrip("0").rstrip(".") if abs(val) < 1000 else f"{val:,.2f}"

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [geojson_coords],
            },
            "properties": {
                "name": html.escape(names[i]),
                "value": val,
                "valueFormatted": val_str,
                "binLabel": bin_labels[min(b_idx, len(bin_labels) - 1)],
                "fillColor": fill_color,
            },
        })

    geojson_data = {
        "type": "FeatureCollection",
        "features": features,
    }
    geojson_json = json.dumps(geojson_data)

    # Build legend swatches
    legend_items: list[dict[str, str]] = []
    for i, label in enumerate(bin_labels):
        col = palette[min(i, len(palette) - 1)]
        legend_items.append({"color": col, "label": label})
    legend_json = json.dumps(legend_items)

    safe_title = html.escape(title)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <style>
        html, body {{
            height: 100%;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #f4f6f8;
        }}
        #header {{
            position: absolute;
            top: 16px;
            left: 60px;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.95);
            padding: 10px 18px;
            border-radius: 8px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.15);
            pointer-events: auto;
        }}
        #header h2 {{
            margin: 0;
            font-size: 16px;
            font-weight: 600;
            color: #1e293b;
        }}
        #header p {{
            margin: 3px 0 0;
            font-size: 12px;
            color: #64748b;
        }}
        #map {{
            width: 100%;
            height: 100%;
        }}
        .info.legend {{
            background: rgba(255, 255, 255, 0.95);
            padding: 12px 16px;
            font-size: 12px;
            line-height: 20px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.15);
            border-radius: 8px;
            color: #334155;
            min-width: 140px;
        }}
        .info.legend h4 {{
            margin: 0 0 8px;
            font-size: 13px;
            font-weight: 600;
            color: #0f172a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
        }}
        .info.legend i {{
            width: 16px;
            height: 16px;
            float: left;
            margin-right: 8px;
            margin-top: 2px;
            border-radius: 3px;
            box-shadow: inset 0 0 0 1px rgba(0,0,0,0.15);
        }}
    </style>
</head>
<body>
    <div id="header">
        <h2>{safe_title}</h2>
        <p>GeoStride Interactive Choropleth &bull; {n_polys} regions</p>
    </div>
    <div id="map"></div>
    <script>
        var map = L.map('map', {{
            center: [0, 0],
            zoom: 2,
            zoomControl: true
        }});

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            maxZoom: 19
        }}).addTo(map);

        var geojsonData = {geojson_json};
        var defaultOpacity = {opacity};

        var choroplethLayer = L.geoJSON(geojsonData, {{
            style: function(feature) {{
                return {{
                    fillColor: feature.properties.fillColor,
                    weight: 1.5,
                    opacity: 1.0,
                    color: '#ffffff',
                    fillOpacity: defaultOpacity
                }};
            }},
            onEachFeature: function(feature, layer) {{
                var p = feature.properties;
                var popupHtml = '<div style="font-size:13px;min-width:140px;">' +
                    '<strong style="font-size:14px;color:#0f172a;">' + p.name + '</strong><br>' +
                    '<hr style="margin:6px 0;border:0;border-top:1px solid #e2e8f0;">' +
                    '<span style="color:#64748b;">Value:</span> <strong style="color:#0284c7;">' + p.valueFormatted + '</strong><br>' +
                    '<span style="color:#64748b;">Bin:</span> ' + p.binLabel +
                    '</div>';

                layer.bindPopup(popupHtml);
                layer.bindTooltip(p.name + ': ' + p.valueFormatted, {{
                    sticky: true,
                    direction: 'auto'
                }});

                layer.on({{
                    mouseover: function(e) {{
                        var l = e.target;
                        l.setStyle({{
                            weight: 3.0,
                            color: '#0f172a',
                            fillOpacity: 0.95
                        }});
                        l.bringToFront();
                    }},
                    mouseout: function(e) {{
                        choroplethLayer.resetStyle(e.target);
                    }},
                    click: function(e) {{
                        map.fitBounds(e.target.getBounds(), {{ padding: [30, 30] }});
                    }}
                }});
            }}
        }}).addTo(map);

        map.fitBounds(choroplethLayer.getBounds(), {{ padding: [40, 40] }});

        // Add legend control
        var legendItems = {legend_json};
        var legend = L.control({{ position: 'bottomright' }});
        legend.onAdd = function (map) {{
            var div = L.DomUtil.create('div', 'info legend');
            div.innerHTML = '<h4>' + '{safe_title}' + '</h4>';
            for (var i = 0; i < legendItems.length; i++) {{
                div.innerHTML +=
                    '<i style="background:' + legendItems[i].color + '"></i> ' +
                    legendItems[i].label + '<br>';
            }}
            return div;
        }};
        legend.addTo(map);
    </script>
</body>
</html>"""

    return html_template


def export_choropleth_map(
    polygons: Sequence[Any] | Mapping[str, Any],
    values: Sequence[float] | np.ndarray,
    output_path: str | Path,
    colormap: str = "viridis",
    n_bins: int = 5,
    title: str = "Choropleth Map",
    region_names: Sequence[str] | None = None,
    opacity: float = 0.75,
) -> str:
    """Render and write a standalone interactive Leaflet Choropleth HTML map to disk.

    Parameters
    ----------
    polygons : Sequence of array-like or Mapping of str to array-like
        Collection of polygon coordinate arrays.
    values : Sequence of float or np.ndarray
        Statistical feature values for each polygon region.
    output_path : str or Path
        Destination HTML file path.
    colormap : {"viridis", "plasma", "blues", "reds", "greens", "coolwarm"}, default="viridis"
        Colormap name for polygon fill shading.
    n_bins : int, default=5
        Number of classification buckets.
    title : str, default="Choropleth Map"
        Title and legend header text.
    region_names : Sequence of str, optional
        Names or labels for regions.
    opacity : float, default=0.75
        Fill opacity.

    Returns
    -------
    str
        Resolved absolute path of the generated HTML map file.
    """
    html_content = render_choropleth_map(
        polygons=polygons,
        values=values,
        colormap=colormap,
        n_bins=n_bins,
        title=title,
        region_names=region_names,
        opacity=opacity,
    )

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_content, encoding="utf-8")
    return str(out_file)
