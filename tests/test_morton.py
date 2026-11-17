"""Unit tests for Morton (Z-order) space-filling curve indexer."""

import numpy as np
import pytest

from geostride.core.bbox import BoundingBox
from geostride.indexing.morton import (
    decode_morton_2d,
    encode_coordinates_morton,
    encode_morton_2d,
)


def test_morton_encode_decode_roundtrip():
    lat = 40.7128
    lon = -74.0060
    bits = 16

    code = encode_morton_2d(lat, lon, bits=bits)
    assert isinstance(code, int)
    assert code >= 0

    dec_lat, dec_lon = decode_morton_2d(code, bits=bits)
    # 16 bits provides ~0.002 degree precision (~200 meters)
    assert abs(dec_lat - lat) < 0.01
    assert abs(dec_lon - lon) < 0.01


def test_morton_bounds():
    # Min boundary: (-90.0, -180.0)
    code_min = encode_morton_2d(-90.0, -180.0, bits=16)
    assert code_min == 0

    # Max boundary: (90.0, 180.0)
    code_max = encode_morton_2d(90.0, 180.0, bits=16)
    assert code_max == 0xFFFFFFFF


def test_morton_custom_bbox():
    bbox = BoundingBox(min_lat=40.0, max_lat=41.0, min_lon=-75.0, max_lon=-74.0)
    code = encode_morton_2d(40.5, -74.5, bits=10, bbox=bbox)
    assert isinstance(code, int)

    dec_lat, dec_lon = decode_morton_2d(code, bits=10, bbox=bbox)
    assert abs(dec_lat - 40.5) < 0.01
    assert abs(dec_lon - (-74.5)) < 0.01


def test_morton_vectorized():
    coords = np.array([
        [40.7128, -74.0060],
        [34.0522, -118.2437],
        [51.5074, -0.1278]
    ])
    codes = encode_coordinates_morton(coords, bits=16)
    assert len(codes) == 3
    assert codes.dtype == np.uint64
    assert len(set(codes)) == 3


def test_morton_invalid_inputs():
    with pytest.raises(ValueError, match="Bits per axis must be between 1 and 16"):
        encode_morton_2d(0.0, 0.0, bits=0)

    with pytest.raises(ValueError, match="Bits per axis must be between 1 and 16"):
        encode_morton_2d(0.0, 0.0, bits=17)

    with pytest.raises(ValueError, match="outside the bounding box"):
        bbox = BoundingBox(min_lat=10.0, max_lat=20.0, min_lon=10.0, max_lon=20.0)
        encode_morton_2d(0.0, 0.0, bits=16, bbox=bbox)

    with pytest.raises(ValueError, match="Morton code must be non-negative"):
        decode_morton_2d(-1, bits=16)
