"""Unit tests for Coordinate and BoundingBox classes."""

import numpy as np
import pytest

from geostride.core.bbox import BoundingBox
from geostride.core.coordinates import Coordinate, validate_coordinates


def test_coordinate_valid():
    coord = Coordinate(latitude=40.7128, longitude=-74.0060)
    assert coord.latitude == 40.7128
    assert coord.longitude == -74.0060
    assert coord.to_tuple() == (40.7128, -74.0060)

    rad = coord.radians
    assert np.isclose(rad[0], np.radians(40.7128))
    assert np.isclose(rad[1], np.radians(-74.0060))


def test_coordinate_invalid_bounds():
    with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
        Coordinate(latitude=91.0, longitude=0.0)

    with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
        Coordinate(latitude=0.0, longitude=181.0)


def test_coordinate_from_sequence():
    c = Coordinate.from_sequence([12.34, 56.78])
    assert c.latitude == 12.34
    assert c.longitude == 56.78

    with pytest.raises(ValueError, match="must have length 2"):
        Coordinate.from_sequence([1.0])


def test_validate_coordinates():
    # 1D valid
    arr1 = validate_coordinates([40.0, -73.0])
    assert arr1.shape == (1, 2)
    assert arr1[0, 0] == 40.0

    # 2D valid
    arr2 = validate_coordinates([[40.0, -73.0], [51.5, -0.1]])
    assert arr2.shape == (2, 2)

    # Invalid dimension
    with pytest.raises(ValueError, match="dimensions"):
        validate_coordinates(np.zeros((2, 2, 2)))

    # Out of range
    with pytest.raises(ValueError, match="Latitude values must be within"):
        validate_coordinates([[100.0, 0.0]])

    with pytest.raises(ValueError, match="Longitude values must be within"):
        validate_coordinates([[0.0, -200.0]])

    # NaN handling
    with pytest.raises(ValueError, match="contains NaN"):
        validate_coordinates([[np.nan, 0.0]])


def test_bounding_box():
    bbox = BoundingBox(min_lat=10.0, max_lat=20.0, min_lon=30.0, max_lon=40.0)
    assert bbox.height_deg == 10.0
    assert bbox.width_deg == 10.0
    assert bbox.center.latitude == 15.0
    assert bbox.center.longitude == 35.0

    # Contains point
    assert bbox.contains((15.0, 35.0))
    assert bbox.contains(Coordinate(10.0, 30.0))
    assert not bbox.contains((25.0, 35.0))

    # Intersections
    overlapping = BoundingBox(min_lat=15.0, max_lat=25.0, min_lon=35.0, max_lon=45.0)
    disjoint = BoundingBox(min_lat=25.0, max_lat=30.0, min_lon=50.0, max_lon=60.0)
    assert bbox.intersects(overlapping)
    assert not bbox.intersects(disjoint)

    # Expand
    expanded = bbox.expand(delta_lat=1.0, delta_lon=2.0)
    assert expanded.min_lat == 9.0
    assert expanded.max_lat == 21.0
    assert expanded.min_lon == 28.0
    assert expanded.max_lon == 42.0

    # From coordinates
    pts = [[10.0, 30.0], [12.0, 28.0], [18.0, 35.0]]
    from_pts = BoundingBox.from_coordinates(pts)
    assert from_pts.min_lat == 10.0
    assert from_pts.max_lat == 18.0
    assert from_pts.min_lon == 28.0
    assert from_pts.max_lon == 35.0
