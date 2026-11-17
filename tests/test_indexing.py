"""Unit tests for Geohash encoding, decoding, bounding boxes, and neighbors."""

import numpy as np
import pytest

from geostride.indexing.geohash import (
    decode_geohash,
    encode_coordinates_geohash,
    encode_geohash,
    geohash_bbox,
    geohash_neighbors,
)


def test_encode_geohash_known_landmarks():
    # Empire State Building, NYC
    esb_lat, esb_lon = 40.748817, -73.985428
    gh_esb = encode_geohash(esb_lat, esb_lon, precision=7)
    assert gh_esb.startswith("dr5ru")

    # Big Ben, London
    bb_lat, bb_lon = 51.5007, -0.1246
    gh_bb = encode_geohash(bb_lat, bb_lon, precision=7)
    assert gh_bb.startswith("gcpuv")


def test_encode_decode_roundtrip():
    lat_orig = 37.774929
    lon_orig = -122.419416
    gh = encode_geohash(lat_orig, lon_orig, precision=9)
    assert len(gh) == 9

    lat_dec, lon_dec, lat_err, lon_err = decode_geohash(gh)
    assert abs(lat_dec - lat_orig) <= lat_err
    assert abs(lon_dec - lon_orig) <= lon_err


def test_geohash_bbox():
    lat, lon = 34.0522, -118.2437
    gh = encode_geohash(lat, lon, precision=6)
    bbox = geohash_bbox(gh)

    assert bbox.contains((lat, lon))
    assert bbox.height_deg > 0.0
    assert bbox.width_deg > 0.0


def test_geohash_neighbors():
    gh = "dr5ru"
    neighbors = geohash_neighbors(gh)

    assert set(neighbors.keys()) == {"n", "s", "e", "w", "ne", "nw", "se", "sw"}
    for n_gh in neighbors.values():
        assert len(n_gh) == len(gh)
        assert n_gh != gh

    # All neighbors should be distinct
    unique_neighbors = set(neighbors.values())
    assert len(unique_neighbors) == 8


def test_encode_coordinates_geohash_vectorized():
    coords = np.array([
        [40.7128, -74.0060],  # NYC
        [51.5074, -0.1278],   # London
        [35.6762, 139.6503]   # Tokyo
    ])
    hashes = encode_coordinates_geohash(coords, precision=6)
    assert len(hashes) == 3
    assert all(isinstance(h, str) and len(h) == 6 for h in hashes)


def test_invalid_geohash_inputs():
    # Out of range coordinates
    with pytest.raises(ValueError, match="Latitude must be within"):
        encode_geohash(95.0, 0.0)

    with pytest.raises(ValueError, match="Longitude must be within"):
        encode_geohash(0.0, 200.0)

    # Invalid precision
    with pytest.raises(ValueError, match="Precision must be between"):
        encode_geohash(0.0, 0.0, precision=15)

    # Empty decode
    with pytest.raises(ValueError, match="empty geohash"):
        decode_geohash("")

    # Invalid base32 character (e.g. 'a', 'i', 'l', 'o' are excluded in standard Geohash)
    with pytest.raises(ValueError, match="Invalid character"):
        decode_geohash("dr5rua")
