"""
tracker.py — Centroid-Based Object Tracker
==========================================
Assigns stable unique IDs to detected objects and tracks them across frames
using nearest-neighbour matching on centroids. No libraries beyond stdlib.

Algorithm (per frame)
---------------------
  1. Compute centroid of each incoming bounding box.
  2. If no existing objects → register all as new.
  3. Build Euclidean distance matrix: existing centroids × new centroids.
  4. Greedy matching: sort all (dist, row, col) pairs ascending;
     assign closest pairs first (each row and col used at most once).
  5. Skip pairs where distance > max_distance (different objects).
  6. Unmatched existing objects → increment disappeared counter;
     deregister if disappeared > max_disappeared.
  7. Unmatched new centroids → register as new objects.
  8. Append matched centroid to per-object trail for trajectory drawing.
"""

import math
from collections import OrderedDict


class CentroidTracker:
    """
    Parameters
    ----------
    max_disappeared : int   Frames an object can be absent before removal.
    max_distance    : float Max pixel distance for a valid centroid match.
    max_trail_len   : int   Number of past positions kept for trail drawing.
    """

    def __init__(self,
                 max_disappeared: int   = 50,
                 max_distance:    float = 100,
                 max_trail_len:   int   = 40):

        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance
        self.max_trail_len   = max_trail_len

        self._next_id    = 0
        self.objects:    OrderedDict = OrderedDict()   # id → (cx, cy)
        self.disappeared:OrderedDict = OrderedDict()   # id → frame count
        self.trails:     OrderedDict = OrderedDict()   # id → [(cx,cy), ...]


    def update(self, rects: list[tuple]) -> OrderedDict:
        """
        Update tracker with bounding boxes from the current frame.

        Parameters
        ----------
        rects : list of (x, y, w, h)

        Returns
        -------
        OrderedDict  id → (cx, cy)  for all currently tracked objects.
        """
        if not rects:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self.objects

        new_centroids = [(x + w // 2, y + h // 2) for x, y, w, h in rects]

        if not self.objects:
            for c in new_centroids:
                self._register(c)
            return self.objects

        obj_ids  = list(self.objects.keys())
        obj_cs   = list(self.objects.values())

        pairs = sorted(
            (self._dist(obj_cs[r], new_centroids[c]), r, c)
            for r in range(len(obj_ids))
            for c in range(len(new_centroids))
        )

        used_rows, used_cols = set(), set()

        for dist, row, col in pairs:
            if row in used_rows or col in used_cols:
                continue
            if dist > self.max_distance:
                break   # remaining distances only get larger

            oid = obj_ids[row]
            self.objects[oid]     = new_centroids[col]
            self.disappeared[oid] = 0
            self._append_trail(oid, new_centroids[col])

            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(len(obj_ids))) - used_rows:
            oid = obj_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        for col in set(range(len(new_centroids))) - used_cols:
            self._register(new_centroids[col])

        return self.objects


    def _register(self, centroid: tuple) -> None:
        self.objects[self._next_id]     = centroid
        self.disappeared[self._next_id] = 0
        self.trails[self._next_id]      = [centroid]
        self._next_id += 1

    def _deregister(self, oid: int) -> None:
        del self.objects[oid]
        del self.disappeared[oid]
        del self.trails[oid]

    def _append_trail(self, oid: int, centroid: tuple) -> None:
        self.trails[oid].append(centroid)
        if len(self.trails[oid]) > self.max_trail_len:
            self.trails[oid].pop(0)

    @staticmethod
    def _dist(a: tuple, b: tuple) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
