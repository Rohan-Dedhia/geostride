"""Unit tests for standalone interactive Leaflet HTML map exporter."""

from pathlib import Path

import numpy as np
import pytest

from geostride.viz.map_export import export_html_map, render_html_map


@pytest.fixture
def sample_coords():
    return np.array(
        [
            [40.7128, -74.0060],  # New York
            [51.5074, -0.1278],  # London
            [35.6762, 139.6503],  # Tokyo
        ]
    )


def test_render_html_map_basic(sample_coords):
    html = render_html_map(sample_coords)

    # Basic document structure
    assert "<!DOCTYPE html>" in html
    assert "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" in html
    assert "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" in html
    assert "tile.openstreetmap.org" in html
    assert "L.circleMarker" in html
    assert "map.fitBounds" in html
    assert "40.7128" in html
    assert "-74.006" in html
    assert "3 points rendered" in html


def test_render_html_map_single_point():
    coords = np.array([[37.7749, -122.4194]])
    html = render_html_map(coords, title="San Francisco")

    assert "San Francisco" in html
    assert "1 point rendered" in html
    assert "map.setView([37.7749, -122.4194], 13)" in html


def test_render_html_map_custom_labels_tooltips(sample_coords):
    labels = ["NYC", "LDN", "TYO"]
    tooltips = ["Capital of Finance", "Capital of UK", "Capital of Japan"]
    html = render_html_map(
        sample_coords,
        labels=labels,
        tooltips=tooltips,
        title="World Hubs",
        radius_px=10,
    )

    assert "World Hubs" in html
    assert "radius: 10" in html
    assert "NYC" in html
    assert "Capital of Finance" in html
    assert "TYO" in html
    assert "Capital of Japan" in html


def test_render_html_map_html_escaping():
    coords = np.array([[10.0, 20.0]])
    labels = ["<script>alert('xss')</script>"]
    tooltips = ['"Double & Single" <tags>']
    html = render_html_map(coords, labels=labels, tooltips=tooltips)

    # Assert raw script tags are escaped
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html
    assert "&quot;Double &amp; Single&quot; &lt;tags&gt;" in html


def test_render_html_map_empty_coordinates():
    with pytest.raises(ValueError, match="Coordinate array cannot be empty"):
        render_html_map(np.empty((0, 2)))


def test_render_html_map_mismatched_labels(sample_coords):
    with pytest.raises(ValueError, match="Length of labels .* does not match"):
        render_html_map(sample_coords, labels=["A", "B"])


def test_render_html_map_mismatched_tooltips(sample_coords):
    with pytest.raises(ValueError, match="Length of tooltips .* does not match"):
        render_html_map(sample_coords, tooltips=["Tip 1"])


def test_render_html_map_invalid_coordinates():
    with pytest.raises(ValueError, match=r"Latitude values must be within \[-90\.0, 90\.0\]"):
        render_html_map([[100.0, 0.0]])


def test_export_html_map_file_writing(sample_coords, tmp_path: Path):
    target = tmp_path / "nested" / "map_output.html"
    export_html_map(target, sample_coords, title="Exported Map")

    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "<title>Exported Map</title>" in content
    assert "L.circleMarker" in content
