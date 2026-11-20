"""Unit tests for OGC Well-Known Text (WKT) parsing and serialization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geostride.io.wkt import (
    dataframe_to_wkt,
    from_wkt,
    to_wkt,
    wkt_to_dataframe,
)


class TestWktSerialization:
    """Tests for to_wkt serialization."""

    def test_point_serialization(self) -> None:
        coords = np.array([40.7128, -74.0060])
        wkt = to_wkt(coords)
        assert wkt == "POINT (-74.006 40.7128)"

        # Explicit type with 2D shape (1, 2)
        wkt2 = to_wkt(np.array([[51.5074, -0.1278]]), geom_type="POINT")
        assert wkt2 == "POINT (-0.1278 51.5074)"

    def test_linestring_serialization(self) -> None:
        coords = np.array([
            [40.7128, -74.0060],
            [34.0522, -118.2437],
            [41.8781, -87.6298],
        ])
        wkt = to_wkt(coords)
        assert wkt.startswith("LINESTRING (")
        assert "-74.006 40.7128" in wkt
        assert "-118.2437 34.0522" in wkt
        assert "-87.6298 41.8781" in wkt

    def test_multipoint_serialization(self) -> None:
        coords = np.array([
            [40.7128, -74.0060],
            [34.0522, -118.2437],
        ])
        wkt = to_wkt(coords, geom_type="MULTIPOINT")
        assert wkt == "MULTIPOINT ((-74.006 40.7128), (-118.2437 34.0522))"

    def test_polygon_serialization_single_ring(self) -> None:
        # Open ring should auto-close
        poly_coords = np.array([
            [0.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [10.0, 0.0],
        ])
        wkt = to_wkt(poly_coords, geom_type="POLYGON")
        assert wkt.startswith("POLYGON ((")
        assert wkt.endswith("0 0))")
        assert "0 0, 10 0, 10 10, 0 10, 0 0" in wkt

    def test_polygon_serialization_with_holes(self) -> None:
        outer = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]])
        hole = np.array([[2.0, 2.0], [2.0, 4.0], [4.0, 4.0], [4.0, 2.0], [2.0, 2.0]])
        wkt = to_wkt([outer, hole], geom_type="POLYGON")
        assert wkt.startswith("POLYGON ((")
        assert "), (" in wkt

    def test_empty_serialization(self) -> None:
        assert to_wkt([], geom_type="POINT") == "POINT EMPTY"
        assert to_wkt([], geom_type="POLYGON") == "POLYGON EMPTY"

    def test_invalid_serialization_inputs(self) -> None:
        with pytest.raises(ValueError, match="Cannot automatically infer geometry type"):
            to_wkt(np.zeros((2, 3, 4)))

        with pytest.raises(ValueError, match="POINT coordinates must have 2 elements"):
            to_wkt(np.array([1.0, 2.0, 3.0]), geom_type="POINT")

        with pytest.raises(ValueError, match="LINESTRING must contain at least 2 vertices"):
            to_wkt(np.array([[1.0, 2.0]]), geom_type="LINESTRING")

        with pytest.raises(ValueError, match="Polygon ring must have at least 3 vertices"):
            to_wkt(np.array([[1.0, 2.0], [3.0, 4.0]]), geom_type="POLYGON")

        with pytest.raises(ValueError, match="Unsupported geometry type"):
            to_wkt(np.array([1.0, 2.0]), geom_type="TRIANGLE")


class TestWktParsing:
    """Tests for from_wkt parsing."""

    def test_parse_point(self) -> None:
        gtype, coords = from_wkt("POINT (-74.006 40.7128)")
        assert gtype == "POINT"
        assert isinstance(coords, np.ndarray)
        assert np.allclose(coords, [40.7128, -74.0060])

        # Test case-insensitivity and extra whitespace
        gtype2, coords2 = from_wkt("  point   (   -0.1278    51.5074  ) ")
        assert gtype2 == "POINT"
        assert np.allclose(coords2, [51.5074, -0.1278])

    def test_parse_linestring(self) -> None:
        wkt = "LINESTRING (-74.006 40.7128, -118.2437 34.0522, -87.6298 41.8781)"
        gtype, coords = from_wkt(wkt)
        assert gtype == "LINESTRING"
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (3, 2)
        assert np.allclose(coords[0], [40.7128, -74.0060])
        assert np.allclose(coords[1], [34.0522, -118.2437])
        assert np.allclose(coords[2], [41.8781, -87.6298])

    def test_parse_multipoint(self) -> None:
        # Standard nested syntax
        wkt1 = "MULTIPOINT ((-74.006 40.7128), (-118.2437 34.0522))"
        gtype1, coords1 = from_wkt(wkt1)
        assert gtype1 == "MULTIPOINT"
        assert coords1.shape == (2, 2)
        assert np.allclose(coords1[0], [40.7128, -74.0060])

        # Simplified flat syntax
        wkt2 = "MULTIPOINT (-74.006 40.7128, -118.2437 34.0522)"
        gtype2, coords2 = from_wkt(wkt2)
        assert gtype2 == "MULTIPOINT"
        assert coords2.shape == (2, 2)
        assert np.allclose(coords2[1], [34.0522, -118.2437])

    def test_parse_polygon_auto_close(self) -> None:
        # Unclosed ring in WKT should be automatically closed
        wkt = "POLYGON ((0 0, 10 0, 10 10, 0 10))"
        gtype, rings = from_wkt(wkt)
        assert gtype == "POLYGON"
        assert isinstance(rings, list)
        assert len(rings) == 1
        assert rings[0].shape == (5, 2)
        assert np.allclose(rings[0][0], rings[0][-1])

    def test_parse_polygon_with_hole(self) -> None:
        wkt = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))"
        gtype, rings = from_wkt(wkt)
        assert gtype == "POLYGON"
        assert len(rings) == 2
        assert rings[0].shape == (5, 2)
        assert rings[1].shape == (5, 2)
        # Coordinate order is [lat, lon]
        assert np.allclose(rings[1][0], [2.0, 2.0])

    def test_parse_empty(self) -> None:
        gtype, coords = from_wkt("POINT EMPTY")
        assert gtype == "POINT"
        assert len(coords) == 0

        gtype, rings = from_wkt("POLYGON EMPTY")
        assert gtype == "POLYGON"
        assert rings == []

    def test_parse_errors(self) -> None:
        with pytest.raises(TypeError, match="WKT input must be a string"):
            from_wkt(12345)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Cannot parse empty WKT"):
            from_wkt("   ")

        with pytest.raises(ValueError, match="Unsupported WKT geometry type"):
            from_wkt("CIRCULARSTRING (0 0, 1 1, 2 0)")

        with pytest.raises(ValueError, match="coordinates must be enclosed in parentheses"):
            from_wkt("POINT -74.006 40.7128")

        with pytest.raises(ValueError, match="POINT requires exactly 2 coordinate tokens"):
            from_wkt("POINT (-74.006 40.7128 100.0)")

        with pytest.raises(ValueError, match="Non-numeric POINT coordinates"):
            from_wkt("POINT (abc def)")

        with pytest.raises(ValueError, match="LINESTRING requires at least 2 coordinate pairs"):
            from_wkt("LINESTRING (-74.006 40.7128)")

        with pytest.raises(ValueError, match="Polygon ring must have at least 3 vertices"):
            from_wkt("POLYGON ((0 0, 1 1))")


class TestDataFrameWktIntegration:
    """Tests for DataFrame WKT conversion."""

    def test_dataframe_to_wkt_and_back(self) -> None:
        df = pd.DataFrame({
            "city": ["New York", "London", "Tokyo"],
            "latitude": [40.7128, 51.5074, 35.6762],
            "longitude": [-74.0060, -0.1278, 139.6503],
            "population": [8800000, 9000000, 14000000],
        })

        wkt_df = dataframe_to_wkt(df, lat_col="latitude", lon_col="longitude", wkt_col="wkt")
        assert "wkt" in wkt_df.columns
        assert wkt_df["wkt"].iloc[0] == "POINT (-74.006 40.7128)"
        assert wkt_df["wkt"].iloc[1] == "POINT (-0.1278 51.5074)"
        assert wkt_df["wkt"].iloc[2] == "POINT (139.6503 35.6762)"
        assert "population" in wkt_df.columns

        # Convert back
        recovered_df = wkt_to_dataframe(wkt_df, wkt_col="wkt", lat_col="latitude_rec", lon_col="longitude_rec")
        assert np.allclose(recovered_df["latitude_rec"], df["latitude"])
        assert np.allclose(recovered_df["longitude_rec"], df["longitude"])
        assert list(recovered_df["city"]) == list(df["city"])

    def test_dataframe_to_wkt_missing_columns(self) -> None:
        df = pd.DataFrame({"lat": [10.0], "lon": [20.0]})
        with pytest.raises(ValueError, match="DataFrame must contain 'latitude' and 'longitude'"):
            dataframe_to_wkt(df)

    def test_dataframe_to_wkt_invalid_coordinates(self) -> None:
        df = pd.DataFrame({"latitude": [95.0], "longitude": [0.0]})
        with pytest.raises(ValueError, match="Latitude values must be within"):
            dataframe_to_wkt(df)

    def test_wkt_to_dataframe_errors(self) -> None:
        df_invalid_type = pd.DataFrame({"wkt": [12345]})
        with pytest.raises(TypeError, match="Expected WKT string"):
            wkt_to_dataframe(df_invalid_type)

        df_wrong = pd.DataFrame({"geom": ["LINESTRING (0 0, 1 1)"]})
        with pytest.raises(ValueError, match="Expected WKT POINT geometry, got 'LINESTRING'"):
            wkt_to_dataframe(df_wrong, wkt_col="geom")

        df_missing = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="DataFrame must contain 'wkt' column"):
            wkt_to_dataframe(df_missing)
