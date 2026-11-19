"""Pure-Python spatial R-Tree indexer for range and nearest-neighbor search."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from heapq import heappop, heappush
from typing import Any

from geostride.core.bbox import BoundingBox

# Mean Earth radius in kilometers (IUGG R1)
_EARTH_RADIUS_KM = 6371.0088


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two coordinates in kilometers."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return _EARTH_RADIUS_KM * c


def _min_dist_point_to_bbox_km(lat: float, lon: float, bbox: BoundingBox) -> float:
    """Calculate the minimum geodetic distance from a point to a bounding box in km."""
    clamped_lat = max(bbox.min_lat, min(lat, bbox.max_lat))
    clamped_lon = max(bbox.min_lon, min(lon, bbox.max_lon))
    return _haversine_km(lat, lon, clamped_lat, clamped_lon)


def _bbox_area(bbox: BoundingBox) -> float:
    """Calculate planar degree-squared area of a bounding box."""
    return (bbox.max_lat - bbox.min_lat) * (bbox.max_lon - bbox.min_lon)


def _bbox_union(b1: BoundingBox, b2: BoundingBox) -> BoundingBox:
    """Calculate the minimum bounding box enclosing both bounding boxes."""
    return BoundingBox(
        min_lat=min(b1.min_lat, b2.min_lat),
        max_lat=max(b1.max_lat, b2.max_lat),
        min_lon=min(b1.min_lon, b2.min_lon),
        max_lon=max(b1.max_lon, b2.max_lon),
    )


def _coerce_bbox(
    bbox: BoundingBox | Sequence[float] | tuple[float, float, float, float],
) -> BoundingBox:
    """Convert a bounding box instance or 4-element sequence into a BoundingBox."""
    if isinstance(bbox, BoundingBox):
        return bbox
    if len(bbox) != 4:
        raise ValueError(
            f"Expected bounding box sequence of length 4 (min_lat, max_lat, min_lon, max_lon); "
            f"got length {len(bbox)}"
        )
    return BoundingBox(
        min_lat=float(bbox[0]),
        max_lat=float(bbox[1]),
        min_lon=float(bbox[2]),
        max_lon=float(bbox[3]),
    )


class _SpatialRTreeNode:
    """Internal or leaf node in the Spatial R-Tree hierarchy."""

    __slots__ = ("entries", "is_leaf", "mbr")

    def __init__(self, is_leaf: bool = True) -> None:
        self.is_leaf = is_leaf
        # List of tuples: (BoundingBox, child_node_or_feature_id)
        self.entries: list[tuple[BoundingBox, Any]] = []
        self.mbr: BoundingBox | None = None

    def update_mbr(self) -> None:
        """Recompute the minimum bounding rectangle enclosing all entries in this node."""
        if not self.entries:
            self.mbr = None
            return
        box = self.entries[0][0]
        for b, _ in self.entries[1:]:
            box = _bbox_union(box, b)
        self.mbr = box


class SpatialRTree:
    """Pure-Python 2D spatial R-Tree bounding-box hierarchical indexer.

    Provides high-throughput spatial range queries, bounding box intersections, and
    best-first branch-and-bound k-nearest neighbor searches without external C libraries.

    Parameters
    ----------
    max_entries : int, default=16
        Maximum number of entries per R-Tree node before splitting.
    min_entries : int, default=4
        Minimum number of entries per R-Tree node (except the root).

    Examples
    --------
    >>> from geostride.indexing import SpatialRTree
    >>> tree = SpatialRTree()
    >>> tree.insert(1, (40.71, 40.73, -74.01, -73.99))
    >>> tree.insert(2, (40.75, 40.77, -73.98, -73.96))
    >>> tree.query_range((40.70, 40.74, -74.02, -73.98))
    [1]
    >>> tree.query_nearest(40.72, -74.00, k=1)
    [1]
    """

    def __init__(self, max_entries: int = 16, min_entries: int = 4) -> None:
        if max_entries < 4:
            raise ValueError(f"max_entries must be at least 4; got {max_entries}")
        if min_entries < 2 or min_entries > max_entries // 2:
            raise ValueError(
                f"min_entries must be between 2 and max_entries // 2 ({max_entries // 2}); "
                f"got {min_entries}"
            )
        self.max_entries = max_entries
        self.min_entries = min_entries
        self._root = _SpatialRTreeNode(is_leaf=True)
        self._size = 0

    def __len__(self) -> int:
        """Return the total number of indexed spatial items."""
        return self._size

    @property
    def bounds(self) -> BoundingBox | None:
        """Return the overall minimum bounding box enclosing all indexed items, or None if empty."""
        return self._root.mbr

    def clear(self) -> None:
        """Reset the R-Tree, clearing all indexed spatial items."""
        self._root = _SpatialRTreeNode(is_leaf=True)
        self._size = 0

    def insert(
        self,
        item_id: int,
        bbox: BoundingBox | Sequence[float] | tuple[float, float, float, float],
    ) -> None:
        """Insert a single spatial feature ID with its bounding box into the R-Tree.

        Parameters
        ----------
        item_id : int
            Unique identifier of the spatial feature.
        bbox : BoundingBox or 4-element sequence
            Spatial bounding box `(min_lat, max_lat, min_lon, max_lon)`.
        """
        box = _coerce_bbox(bbox)
        leaf = self._choose_leaf(self._root, box)
        leaf.entries.append((box, item_id))
        leaf.update_mbr()
        self._size += 1

        if len(leaf.entries) > self.max_entries:
            self._split_and_propagate(leaf)
        else:
            self._adjust_mbr_upward(leaf)

    def bulk_load(
        self,
        ids: Sequence[int],
        bboxes: Sequence[BoundingBox | Sequence[float]],
    ) -> None:
        """Bulk load a collection of items using the Sort-Tile-Recursive (STR) algorithm.

        Builds a compact, well-clustered, and balanced R-Tree in O(N log N) time with
        minimal bounding box overlap.

        Parameters
        ----------
        ids : Sequence of int
            Feature IDs.
        bboxes : Sequence of BoundingBox or 4-element sequences
            Corresponding bounding boxes.

        Raises
        ------
        ValueError
            If lengths of `ids` and `bboxes` do not match.
        """
        n = len(ids)
        if len(bboxes) != n:
            raise ValueError(
                f"Length mismatch: {len(ids)} IDs provided but {len(bboxes)} bounding boxes."
            )
        if n == 0:
            return

        items = [(_coerce_bbox(b), int(item_id)) for b, item_id in zip(bboxes, ids, strict=True)]
        self._root = self._str_pack(items)
        self._size += n

    def query_range(
        self,
        bbox: BoundingBox | Sequence[float] | tuple[float, float, float, float],
    ) -> list[int]:
        """Query for all feature IDs whose bounding boxes intersect the given query box.

        Parameters
        ----------
        bbox : BoundingBox or 4-element sequence
            Query bounding box `(min_lat, max_lat, min_lon, max_lon)`.

        Returns
        -------
        list of int
            List of matching feature IDs.
        """
        if self._size == 0 or self._root.mbr is None:
            return []

        query_box = _coerce_bbox(bbox)
        results: list[int] = []
        stack: list[_SpatialRTreeNode] = [self._root]

        while stack:
            node = stack.pop()
            for entry_box, child_or_id in node.entries:
                if query_box.intersects(entry_box):
                    if node.is_leaf:
                        results.append(child_or_id)
                    else:
                        stack.append(child_or_id)

        return results

    def query_nearest(
        self,
        latitude: float,
        longitude: float,
        k: int = 1,
    ) -> list[int]:
        """Query for the k nearest feature IDs using Best-First branch-and-bound search.

        Computes geodesic distances in kilometers and explores the most promising
        subtrees first, guaranteeing exact k-nearest neighbor retrieval.

        Parameters
        ----------
        latitude : float
            Query point latitude in [-90.0, 90.0].
        longitude : float
            Query point longitude in [-180.0, 180.0].
        k : int, default=1
            Number of nearest neighbors to retrieve.

        Returns
        -------
        list of int
            List of the k nearest feature IDs ordered by ascending distance.

        Raises
        ------
        ValueError
            If latitude/longitude coordinates are out of bounds.
        """
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90 degrees; got {latitude}")
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180 degrees; got {longitude}")
        if k <= 0 or self._size == 0 or self._root.mbr is None:
            return []

        # Priority queue holds: (min_dist_km, counter, is_feature_id, item_or_node)
        counter = itertools.count()
        heap: list[tuple[float, int, bool, Any]] = []

        root_dist = _min_dist_point_to_bbox_km(latitude, longitude, self._root.mbr)
        heappush(heap, (root_dist, next(counter), False, self._root))

        nearest: list[int] = []

        while heap and len(nearest) < k:
            _, _, is_item, obj = heappop(heap)
            if is_item:
                nearest.append(obj)
            else:
                node: _SpatialRTreeNode = obj
                if node.is_leaf:
                    for entry_box, feat_id in node.entries:
                        dist = _min_dist_point_to_bbox_km(latitude, longitude, entry_box)
                        heappush(heap, (dist, next(counter), True, feat_id))
                else:
                    for entry_box, child_node in node.entries:
                        dist = _min_dist_point_to_bbox_km(latitude, longitude, entry_box)
                        heappush(heap, (dist, next(counter), False, child_node))

        return nearest

    # -------------------------------------------------------------------------
    # Internal R-Tree helper routines
    # -------------------------------------------------------------------------

    def _choose_leaf(self, node: _SpatialRTreeNode, bbox: BoundingBox) -> _SpatialRTreeNode:
        """Select the leaf node requiring minimum area enlargement to contain the new box."""
        if node.is_leaf:
            return node

        best_child: _SpatialRTreeNode | None = None
        best_enlargement = float("inf")
        best_area = float("inf")

        for child_bbox, child_node in node.entries:
            cur_area = _bbox_area(child_bbox)
            union_area = _bbox_area(_bbox_union(child_bbox, bbox))
            enlargement = union_area - cur_area

            if enlargement < best_enlargement or (
                math.isclose(enlargement, best_enlargement) and cur_area < best_area
            ):
                best_enlargement = enlargement
                best_area = cur_area
                best_child = child_node

        return self._choose_leaf(best_child or node.entries[0][1], bbox)

    def _adjust_mbr_upward(self, node: _SpatialRTreeNode) -> None:
        """Update MBRs along ancestors from leaf up to root."""
        cur: _SpatialRTreeNode | None = node
        while cur is not None:
            cur.update_mbr()
            cur = self._find_parent(self._root, cur)

    def _find_parent(
        self,
        current: _SpatialRTreeNode,
        target: _SpatialRTreeNode,
    ) -> _SpatialRTreeNode | None:
        """Find the parent node of target by top-down search."""
        if current.is_leaf:
            return None
        for _, child in current.entries:
            if child is target:
                return current
            parent = self._find_parent(child, target)
            if parent is not None:
                return parent
        return None

    def _split_and_propagate(self, node: _SpatialRTreeNode) -> None:
        """Split an overflowing node using Quadratic Split and propagate upward."""
        node1, node2 = self._quadratic_split(node)

        parent = self._find_parent(self._root, node)
        if parent is None:
            # Splitting the root creates a new root node
            new_root = _SpatialRTreeNode(is_leaf=False)
            new_root.entries.append((node1.mbr, node1))  # type: ignore[arg-type]
            new_root.entries.append((node2.mbr, node2))  # type: ignore[arg-type]
            new_root.update_mbr()
            self._root = new_root
        else:
            # Replace node in parent with node1 and append node2
            for i, (_, child) in enumerate(parent.entries):
                if child is node:
                    parent.entries[i] = (node1.mbr, node1)  # type: ignore[arg-type]
                    break
            parent.entries.append((node2.mbr, node2))  # type: ignore[arg-type]
            parent.update_mbr()

            if len(parent.entries) > self.max_entries:
                self._split_and_propagate(parent)
            else:
                self._adjust_mbr_upward(parent)

    def _quadratic_split(
        self,
        node: _SpatialRTreeNode,
    ) -> tuple[_SpatialRTreeNode, _SpatialRTreeNode]:
        """Divide node entries into two nodes using Guttman's Quadratic Split algorithm."""
        entries = node.entries
        m = len(entries)

        # 1. PickSeeds: pair of entries that waste the most area if put together
        best_waste = -1.0
        seed1_idx, seed2_idx = 0, 1

        for i in range(m):
            area_i = _bbox_area(entries[i][0])
            for j in range(i + 1, m):
                area_j = _bbox_area(entries[j][0])
                u_area = _bbox_area(_bbox_union(entries[i][0], entries[j][0]))
                waste = u_area - area_i - area_j
                if waste > best_waste:
                    best_waste = waste
                    seed1_idx, seed2_idx = i, j

        group1 = _SpatialRTreeNode(is_leaf=node.is_leaf)
        group2 = _SpatialRTreeNode(is_leaf=node.is_leaf)

        group1.entries.append(entries[seed1_idx])
        group2.entries.append(entries[seed2_idx])
        group1.update_mbr()
        group2.update_mbr()

        remaining = [e for idx, e in enumerate(entries) if idx not in (seed1_idx, seed2_idx)]

        # 2. Distribute remaining entries
        for entry in remaining:
            box = entry[0]
            # Check if one group needs all remaining entries to satisfy min_entries
            entries_left = len(remaining)
            if len(group1.entries) + entries_left == self.min_entries:
                group1.entries.append(entry)
                group1.update_mbr()
                continue
            if len(group2.entries) + entries_left == self.min_entries:
                group2.entries.append(entry)
                group2.update_mbr()
                continue

            # Compare area enlargement
            mbr1 = group1.mbr  # type: ignore[arg-type]
            mbr2 = group2.mbr  # type: ignore[arg-type]
            d1 = _bbox_area(_bbox_union(mbr1, box)) - _bbox_area(mbr1)
            d2 = _bbox_area(_bbox_union(mbr2, box)) - _bbox_area(mbr2)

            if d1 < d2:
                group1.entries.append(entry)
                group1.update_mbr()
            elif d2 < d1:
                group2.entries.append(entry)
                group2.update_mbr()
            elif _bbox_area(mbr1) < _bbox_area(mbr2) or len(group1.entries) <= len(
                group2.entries
            ):
                group1.entries.append(entry)
                group1.update_mbr()
            else:
                group2.entries.append(entry)
                group2.update_mbr()

        return group1, group2

    def _str_pack(self, items: list[tuple[BoundingBox, int]]) -> _SpatialRTreeNode:
        """Recursively build balanced R-Tree layers via Sort-Tile-Recursive (STR)."""
        n = len(items)
        m = self.max_entries

        if n <= m:
            leaf = _SpatialRTreeNode(is_leaf=True)
            leaf.entries = list(items)
            leaf.update_mbr()
            return leaf

        # Build leaf level nodes
        num_leaves = math.ceil(n / m)
        num_slices = math.ceil(math.sqrt(num_leaves))
        slice_capacity = num_slices * m

        # Sort items by centroid latitude
        items.sort(key=lambda item: (item[0].min_lat + item[0].max_lat) / 2.0)

        leaf_nodes: list[_SpatialRTreeNode] = []
        for slice_idx in range(num_slices):
            slice_items = items[slice_idx * slice_capacity : (slice_idx + 1) * slice_capacity]
            if not slice_items:
                continue
            # Sort slice items by centroid longitude
            slice_items.sort(key=lambda item: (item[0].min_lon + item[0].max_lon) / 2.0)
            for j in range(0, len(slice_items), m):
                chunk = slice_items[j : j + m]
                leaf = _SpatialRTreeNode(is_leaf=True)
                leaf.entries = list(chunk)
                leaf.update_mbr()
                leaf_nodes.append(leaf)

        # Recursively construct internal node levels
        current_layer: list[_SpatialRTreeNode] = leaf_nodes
        while len(current_layer) > 1:
            next_layer: list[_SpatialRTreeNode] = []
            layer_n = len(current_layer)
            layer_num_nodes = math.ceil(layer_n / m)
            layer_num_slices = math.ceil(math.sqrt(layer_num_nodes))
            layer_slice_cap = layer_num_slices * m

            current_layer.sort(
                key=lambda node: (node.mbr.min_lat + node.mbr.max_lat) / 2.0  # type: ignore[union-attr]
            )

            for slice_idx in range(layer_num_slices):
                slice_nodes = current_layer[
                    slice_idx * layer_slice_cap : (slice_idx + 1) * layer_slice_cap
                ]
                if not slice_nodes:
                    continue
                slice_nodes.sort(
                    key=lambda node: (node.mbr.min_lon + node.mbr.max_lon) / 2.0  # type: ignore[union-attr]
                )
                for j in range(0, len(slice_nodes), m):
                    chunk_nodes = slice_nodes[j : j + m]
                    parent = _SpatialRTreeNode(is_leaf=False)
                    parent.entries = [(node.mbr, node) for node in chunk_nodes]  # type: ignore[misc]
                    parent.update_mbr()
                    next_layer.append(parent)

            current_layer = next_layer

        return current_layer[0]
