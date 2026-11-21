"""Unit tests for standalone interactive Choropleth HTML map exporter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geostride.viz.choropleth import (
    COLORMAPS,
    _assign_bins,
    _get_palette_colors,
    export_choropleth_map,
    render_choropleth_map,
)


@pytest.fixture
def sample_polygons() -> list[np.ndarray]:
    # 3 square polygons adjacent along equator
    p1 = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    p2 = np.array([[0.0, 1.0], [0.0, 2.0], [1.0, 2.0], [1.0, 1.0], [0.0, 1.0]])
    p3 = np.array([[0.0, 2.0], [0.0, 3.0], [1.0, 3.0], [1.0, 2.0], [0.0, 2.0]])
    return [p1, p2, p3]


class TestChoroplethColorAndBinning:
    """Tests for color ramp interpolation and binning utilities."""

    def test_palette_retrieval_and_interpolation(self) -> None:
        # Default 5-color palette
        v5 = _get_palette_colors("viridis", 5)
        assert len(v5) == 5
        assert v5[0].startswith("#")

        # Interpolated 3-color palette
        v3 = _get_palette_colors("viridis", 3)
        assert len(v3) == 3

        # Interpolated 8-color palette
        v8 = _get_palette_colors("plasma", 8)
        assert len(v8) == 8

    def test_unsupported_colormap(self) -> None:
        with pytest.raises(ValueError, match="Unsupported colormap"):
            _get_palette_colors("rainbow_glow", 5)

    def test_assign_bins_quantile(self) -> None:
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        indices, edges, labels = _assign_bins(vals, n_bins=5)
        assert len(indices) == 5
        assert len(edges) == 6
        assert len(labels) == 5
        assert indices[0] == 0
        assert indices[-1] == 4

    def test_assign_bins_constant_values(self) -> None:
        vals = np.array([42.0, 42.0, 42.0])
        indices, _edges, labels = _assign_bins(vals, n_bins=5)
        assert np.all(indices == 0)
        assert len(labels) == 1
        assert "42.00" in labels[0]


class TestRenderChoroplethMap:
    """Tests for render_choropleth_map HTML generation."""

    def test_render_basic_html(self, sample_polygons: list[np.ndarray]) -> None:
        values = [15.2, 85.7, 42.0]
        html_out = render_choropleth_map(
            polygons=sample_polygons,
            values=values,
            colormap="viridis",
            n_bins=3,
            title="Test Choropleth",
        )

        assert "<!DOCTYPE html>" in html_out
        assert "leaflet.js" in html_out
        assert "Test Choropleth" in html_out
        assert "L.geoJSON" in html_out
        assert "info legend" in html_out
        assert "#440154" in html_out

    def test_render_with_mapping(self) -> None:
        p_dict = {
            "Zone Alpha": [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            "Zone Beta": [[0.0, 1.0], [0.0, 2.0], [1.0, 2.0], [1.0, 1.0]],
        }
        values = [100.0, 250.0]

        html_out = render_choropleth_map(
            polygons=p_dict,
            values=values,
            colormap="blues",
            n_bins=2,
        )
        assert "Zone Alpha" in html_out
        assert "Zone Beta" in html_out

    def test_all_colormaps_render_cleanly(self, sample_polygons: list[np.ndarray]) -> None:
        values = [1.0, 2.0, 3.0]
        for cmap in COLORMAPS:
            out = render_choropleth_map(sample_polygons, values, colormap=cmap)
            assert "<!DOCTYPE html>" in out

    def test_validation_errors(self, sample_polygons: list[np.ndarray]) -> None:
        with pytest.raises(ValueError, match="n_bins must be at least 1"):
            render_choropleth_map(sample_polygons, [1, 2, 3], n_bins=0)

        with pytest.raises(ValueError, match="Polygon collection cannot be empty"):
            render_choropleth_map([], [])

        with pytest.raises(ValueError, match="Length mismatch"):
            render_choropleth_map(sample_polygons, [1.0, 2.0])

        with pytest.raises(ValueError, match="Length mismatch"):
            render_choropleth_map(sample_polygons, [1.0, 2.0, 3.0], region_names=["A", "B"])

        with pytest.raises(ValueError, match="must have at least 3 vertices"):
            render_choropleth_map([[[0.0, 0.0], [1.0, 1.0]]], [10.0])


class TestExportChoroplethMap:
    """Tests for export_choropleth_map file writing."""

    def test_export_file_creation(
        self, sample_polygons: list[np.ndarray], tmp_path: Path
    ) -> None:
        out_file = tmp_path / "subdir" / "test_choropleth.html"
        result_path = export_choropleth_map(
            polygons=sample_polygons,
            values=[10.0, 20.0, 30.0],
            output_path=out_file,
            colormap="plasma",
            title="Exported Map",
        )

        assert Path(result_path).exists()
        content = Path(result_path).read_text(encoding="utf-8")
        assert "Exported Map" in content
        assert "leaflet" in content
