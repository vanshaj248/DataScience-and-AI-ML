"""
tracker.py — Centroid-Based Object Tracker
==========================================
Assigns unique IDs to detected objects and tracks them across frames
using Euclidean distance between centroids. No deep learning used.

Algorithm:
  1. On each frame, receive a list of bounding-box rectangles.
  2. Compute centroids of incoming boxes.
  3. Match incoming centroids to existing tracked objects using
     nearest-neighbour (minimum Euclidean distance).
  4. Register new objects and deregister ones that have been
     absent for too many consecutive frames.
"""

import math
from collections import OrderedDict


class CentroidTracker:
    """
    Tracks objects by matching centroids frame-to-frame.

    Parameters
    ----------
    max_disappeared : int
        How many consecutive frames an object may be absent before
        it is deregistered. Increase for slower / occluded objects.
    max_distance : float
        Maximum pixel distance allowed when matching a centroid to
        an existing object. Prevents incorrect merging of far-apart
        detections.
    max_trail_len : int
        Maximum number of centroid positions stored for trajectory
        drawing.
    """

    def __init__(self, max_disappeared: int = 40,
                 max_distance: float = 80,
                 max_trail_len: int = 30):
        self.next_id = 0                        # ever-incrementing unique ID
        self.objects: OrderedDict = OrderedDict()     # id → centroid (cx, cy)
        self.disappeared: OrderedDict = OrderedDict() # id → missing-frame count
        self.trails: OrderedDict = OrderedDict()      # id → list of centroids

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_trail_len = max_trail_len

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(self, centroid: tuple) -> None:
        """Add a brand-new object with a fresh ID."""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.trails[self.next_id] = [centroid]
        self.next_id += 1

    def _deregister(self, object_id: int) -> None:
        """Remove an object that has been absent too long."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.trails[object_id]

    @staticmethod
    def _euclidean(a: tuple, b: tuple) -> float:
        """Straight-line distance between two (x, y) points."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, rects: list) -> OrderedDict:
        """
        Update tracker with a list of bounding-box rectangles.

        Parameters
        ----------
        rects : list of (x, y, w, h) tuples
            Detected bounding boxes from the current frame.

        Returns
        -------
        OrderedDict mapping object_id → (cx, cy) centroid.
        """

        # ── Case A: No detections this frame ───────────────────────────
        if len(rects) == 0:
            # Increment disappeared count for every tracked object
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self.objects

        # ── Compute centroids of incoming rectangles ───────────────────
        input_centroids = []
        for (x, y, w, h) in rects:
            cx = x + w // 2
            cy = y + h // 2
            input_centroids.append((cx, cy))

        # ── Case B: No existing objects — register all as new ──────────
        if len(self.objects) == 0:
            for c in input_centroids:
                self._register(c)
            return self.objects

        # ── Case C: Match incoming centroids to existing objects ────────
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Build a distance matrix: rows = existing objects, cols = new centroids
        distance_matrix = []
        for obj_c in object_centroids:
            row = [self._euclidean(obj_c, inp_c) for inp_c in input_centroids]
            distance_matrix.append(row)

        # Greedy matching: sort by minimum distance, assign closest pairs first
        # rows_sorted / cols_sorted track which rows/cols have been used
        used_rows = set()
        used_cols = set()

        # Flatten all (distance, row, col) then sort ascending by distance
        flat = []
        for r, row in enumerate(distance_matrix):
            for c, dist in enumerate(row):
                flat.append((dist, r, c))
        flat.sort(key=lambda x: x[0])

        for dist, row, col in flat:
            if row in used_rows or col in used_cols:
                continue

            # Reject match if centroid has moved too far (likely different object)
            if dist > self.max_distance:
                break   # remaining distances will only be larger

            obj_id = object_ids[row]
            new_centroid = input_centroids[col]

            # Update centroid and reset disappeared counter
            self.objects[obj_id] = new_centroid
            self.disappeared[obj_id] = 0

            # Append to trajectory trail
            self.trails[obj_id].append(new_centroid)
            if len(self.trails[obj_id]) > self.max_trail_len:
                self.trails[obj_id].pop(0)

            used_rows.add(row)
            used_cols.add(col)

        # Existing objects with no match this frame
        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)

        # New centroids with no matching existing object → register
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self._register(input_centroids[col])

        return self.objects
