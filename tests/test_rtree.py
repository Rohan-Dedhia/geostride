"""Unit tests for pure-Python Spatial R-Tree hierarchical indexer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geostride import SpatialRTree
from geostride.core.bbox import BoundingBox
from geostride.indexing.rtree import _haversine_km


def test_rtree_initialization_and_empty() -> None:
    """Test empty R-Tree properties and queries."""
    tree = SpatialRTree(max_entries=8, min_entries=2)
    assert len(tree) == 0
    assert tree.bounds is None

    # Queries on empty tree
    assert tree.query_range((0.0, 1.0, 0.0, 1.0)) == []
    assert tree.query_nearest(0.0, 0.0, k=5) == []


def test_rtree_invalid_arguments() -> None:
    """Test validation of tree configuration and query arguments."""
    with pytest.raises(ValueError, match="max_entries must be at least 4"):
        SpatialRTree(max_entries=3)

    with pytest.raises(ValueError, match="min_entries must be between 2 and max_entries // 2"):
        SpatialRTree(max_entries=8, min_entries=5)

    tree = SpatialRTree()
    with pytest.raises(ValueError, match="Length mismatch"):
        tree.bulk_load([1, 2], [(0, 1, 0, 1)])

    with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
        tree.query_nearest(95.0, 0.0)

    with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
        tree.query_nearest(0.0, 200.0)


def test_rtree_insert_and_range_query() -> None:
    """Test incremental insertion and spatial range queries."""
    tree = SpatialRTree(max_entries=4, min_entries=2)

    # Insert 3 non-overlapping boxes
    tree.insert(1, (10.0, 12.0, 20.0, 22.0))
    tree.insert(2, (15.0, 17.0, 25.0, 27.0))
    tree.insert(3, (30.0, 32.0, 40.0, 42.0))

    assert len(tree) == 3
    bounds = tree.bounds
    assert bounds is not None
    assert bounds.min_lat == 10.0
    assert bounds.max_lat == 32.0
    assert bounds.min_lon == 20.0
    assert bounds.max_lon == 42.0

    # Query covering item 1 and 2
    res1 = tree.query_range((9.0, 18.0, 19.0, 28.0))
    assert set(res1) == {1, 2}

    # Query covering only item 3
    res2 = tree.query_range(BoundingBox(min_lat=29.0, max_lat=35.0, min_lon=39.0, max_lon=45.0))
    assert res2 == [3]

    # Query covering none
    res3 = tree.query_range((0.0, 5.0, 0.0, 5.0))
    assert res3 == []


def test_rtree_split_overflow() -> None:
    """Test quadratic node splitting when entries exceed max_entries."""
    # Tree with max_entries=4; inserting 20 entries forces multiple splits
    tree = SpatialRTree(max_entries=4, min_entries=2)

    for i in range(20):
        # Dispersed point-boxes
        lat = float(i)
        lon = float(i * 2)
        tree.insert(i, (lat, lat + 0.5, lon, lon + 0.5))

    assert len(tree) == 20

    # Query all items
    all_items = tree.query_range((-1.0, 25.0, -1.0, 50.0))
    assert set(all_items) == set(range(20))

    # Query localized sub-region
    sub_items = tree.query_range((2.5, 6.5, 5.0, 13.0))
    # Boxes:
    # 2: [2, 2.5, 4, 4.5] (lon 4.5 < 5.0, outside)
    # 3: [3, 3.5, 6, 6.5]
    # 4: [4, 4.5, 8, 8.5]
    # 5: [5, 5.5, 10, 10.5]
    # 6: [6, 6.5, 12, 12.5]
    assert set(sub_items) == {3, 4, 5, 6}


def test_rtree_bulk_loading_str() -> None:
    """Test Sort-Tile-Recursive bulk loading on grid points."""
    tree = SpatialRTree(max_entries=8, min_entries=2)

    ids = list(range(100))
    bboxes = []
    for i in range(10):
        for j in range(10):
            bboxes.append((float(i), float(i) + 0.8, float(j), float(j) + 0.8))

    tree.bulk_load(ids, bboxes)
    assert len(tree) == 100

    # Query central cluster
    res = tree.query_range((3.0, 5.0, 3.0, 5.0))
    # Coordinates where i in [3, 4, 5] and j in [3, 4, 5]
    expected_ids = {i * 10 + j for i in range(3, 6) for j in range(3, 6)}
    assert set(res) == expected_ids


def test_rtree_nearest_neighbor_accuracy() -> None:
    """Test k-nearest neighbor query against ground truth geodesic distances."""
    tree = SpatialRTree(max_entries=8, min_entries=2)

    # 10 features placed along diagonal
    points = [(float(i) * 2.0, float(i) * 2.0) for i in range(10)]
    bboxes = [(p[0], p[0], p[1], p[1]) for p in points]
    tree.bulk_load(list(range(10)), bboxes)

    # Query near item 3 (lat=6.0, lon=6.0)
    query_lat, query_lon = 6.1, 6.1
    nearest_3 = tree.query_nearest(query_lat, query_lon, k=3)
    assert len(nearest_3) == 3

    # Closest must be item 3, followed by 4 or 2
    assert nearest_3[0] == 3

    # Verify distances are monotonically increasing
    dists = [
        _haversine_km(query_lat, query_lon, points[idx][0], points[idx][1])
        for idx in nearest_3
    ]
    assert dists[0] <= dists[1] <= dists[2]


def test_rtree_clear() -> None:
    """Test clearing and re-using the R-Tree."""
    tree = SpatialRTree()
    tree.insert(1, (10, 11, 20, 21))
    assert len(tree) == 1
    tree.clear()
    assert len(tree) == 0
    assert tree.bounds is None
    assert tree.query_range((0, 50, 0, 50)) == []


def test_rtree_random_scale_against_brute_force() -> None:
    """Fuzz test 500 items and multiple random range queries against brute force."""
    rng = np.random.default_rng(123)
    n = 300

    lats = rng.uniform(20.0, 40.0, size=n)
    lons = rng.uniform(-100.0, -80.0, size=n)
    sizes = rng.uniform(0.1, 0.5, size=n)

    bboxes = [
        BoundingBox(
            min_lat=float(lats[i]),
            max_lat=float(lats[i] + sizes[i]),
            min_lon=float(lons[i]),
            max_lon=float(lons[i] + sizes[i]),
        )
        for i in range(n)
    ]
    ids = list(range(n))

    tree = SpatialRTree(max_entries=16, min_entries=4)
    tree.bulk_load(ids, bboxes)
    assert len(tree) == n

    # Perform 20 random range queries
    for _ in range(20):
        qlat = float(rng.uniform(22.0, 38.0))
        qlon = float(rng.uniform(-98.0, -82.0))
        qbox = BoundingBox(min_lat=qlat, max_lat=qlat + 2.0, min_lon=qlon, max_lon=qlon + 2.0)

        # R-Tree query
        rtree_hits = set(tree.query_range(qbox))

        # Brute-force linear scan
        brute_hits = {i for i, b in enumerate(bboxes) if qbox.intersects(b)}

        assert rtree_hits == brute_hits

    # Perform 5 random k-nearest neighbor queries against brute force
    for _ in range(5):
        qlat = float(rng.uniform(20.0, 40.0))
        qlon = float(rng.uniform(-100.0, -80.0))
        k = 5

        rtree_k = tree.query_nearest(qlat, qlon, k=k)
        assert len(rtree_k) == k

        # Brute force distances to bbox centers
        brute_dists = [
            (
                _haversine_km(
                    qlat,
                    qlon,
                    max(b.min_lat, min(qlat, b.max_lat)),
                    max(b.min_lon, min(qlon, b.max_lon)),
                ),
                i,
            )
            for i, b in enumerate(bboxes)
        ]
        brute_dists.sort(key=lambda x: x[0])
        expected_top_k = [x[1] for x in brute_dists[:k]]
        assert rtree_k[0] == expected_top_k[0]

        # The closest distance should match exactly
        assert math.isclose(
            brute_dists[0][0],
            _haversine_km(
                qlat,
                qlon,
                max(bboxes[rtree_k[0]].min_lat, min(qlat, bboxes[rtree_k[0]].max_lat)),
                max(bboxes[rtree_k[0]].min_lon, min(qlon, bboxes[rtree_k[0]].max_lon)),
            ),
            rel_tol=1e-6,
        )
