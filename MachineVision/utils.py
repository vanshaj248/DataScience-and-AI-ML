"""
utils.py — Drawing Helpers, FPS Counter & Video Writer
=======================================================
Utility functions used by main.py:
  - FPS computation
  - Bounding-box / ID / centroid drawing
  - Trajectory (trail) drawing
  - Motion-mask display
  - Output video writer wrapper
"""

import time
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# FPS Counter
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """
    Computes frames-per-second using a rolling average over the last N frames.
    More stable than per-frame delta timing.
    """

    def __init__(self, avg_over: int = 30):
        self._avg_over = avg_over
        self._timestamps: list = []

    def tick(self) -> float:
        """Call once per frame; returns current smoothed FPS."""
        now = time.monotonic()
        self._timestamps.append(now)

        # Keep only the last N timestamps
        if len(self._timestamps) > self._avg_over:
            self._timestamps.pop(0)

        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Drawing Utilities
# ─────────────────────────────────────────────────────────────────────────────

# Colour palette — up to 20 distinct colours for object IDs
_PALETTE = [
    (255,  82,  82),   # red
    ( 82, 255,  82),   # green
    ( 82, 178, 255),   # sky-blue
    (255, 178,  82),   # orange
    (178,  82, 255),   # violet
    (255, 255,  82),   # yellow
    ( 82, 255, 220),   # teal
    (255,  82, 178),   # pink
    (160, 255,  82),   # lime
    ( 82, 119, 255),   # indigo
    (255, 140,   0),   # dark-orange
    (  0, 255, 255),   # cyan
    (255,   0, 128),   # magenta
    (128, 255,   0),   # chartreuse
    (  0, 128, 255),   # azure
    (255, 128,   0),   # amber
    (128,   0, 255),   # purple
    (  0, 255, 128),   # spring green
    (255,   0,   0),   # pure red
    (  0,   0, 255),   # pure blue
]


def get_colour(object_id: int) -> tuple:
    """Return a BGR colour that is consistent for a given object ID."""
    # Convert from RGB to BGR for OpenCV
    r, g, b = _PALETTE[object_id % len(_PALETTE)]
    return (b, g, r)


def draw_bounding_box(frame: np.ndarray, rect: tuple,
                      object_id: int, label: str = "") -> None:
    """
    Draw a rounded-corner-style bounding box and label.

    Parameters
    ----------
    frame     : BGR image (modified in-place)
    rect      : (x, y, w, h)
    object_id : used to select a consistent colour
    label     : optional extra text after the ID
    """
    x, y, w, h = rect
    colour = get_colour(object_id)

    # Outer rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

    # Label background pill
    text = f"ID {object_id}" + (f"  {label}" if label else "")
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x1 = x
    label_y1 = max(y - th - baseline - 4, 0)
    label_x2 = x + tw + 6
    label_y2 = max(y - 2, th + 4)

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), colour, -1)
    cv2.putText(frame, text,
                (x + 3, max(y - baseline - 2, th + 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)


def draw_centroid(frame: np.ndarray, centroid: tuple, object_id: int) -> None:
    """Draw a small filled circle at the centroid position."""
    colour = get_colour(object_id)
    cv2.circle(frame, centroid, 4, colour, -1)
    cv2.circle(frame, centroid, 4, (255, 255, 255), 1)  # white outline


def draw_trail(frame: np.ndarray, trail: list, object_id: int) -> None:
    """
    Draw the motion trajectory as a fading polyline.
    Older positions are drawn thinner and more transparent-looking
    by varying the thickness of each segment.

    Parameters
    ----------
    trail : list of (cx, cy) — ordered oldest → newest
    """
    if len(trail) < 2:
        return

    colour = get_colour(object_id)
    n = len(trail)

    for i in range(1, n):
        # Fade effect: thickness increases toward the most-recent point
        thickness = max(1, int((i / n) * 3))
        # Alpha-like fade via colour darkening for older segments
        fade = i / n  # 0.0 (oldest) → 1.0 (newest)
        faded_colour = tuple(int(c * (0.3 + 0.7 * fade)) for c in colour)
        cv2.line(frame, trail[i - 1], trail[i], faded_colour, thickness, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Overlay FPS in the top-right corner."""
    text = f"FPS: {fps:.1f}"
    (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = frame.shape[1] - tw - 10
    cv2.putText(frame, text, (x, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def draw_object_count(frame: np.ndarray, count: int) -> None:
    """Overlay active object count in the top-left corner."""
    cv2.putText(frame, f"Objects: {count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def show_motion_mask(mask: np.ndarray,
                     window_name: str = "Motion Mask") -> None:
    """
    Display the binary motion mask in a separate window.
    Converts single-channel mask to a coloured heat-map for clarity.
    """
    coloured = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
    cv2.imshow(window_name, coloured)


# ─────────────────────────────────────────────────────────────────────────────
# Video Writer Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class VideoWriter:
    """
    Thin wrapper around cv2.VideoWriter.
    Automatically infers frame size from the first write call.
    """

    def __init__(self, output_path: str, fps: float = 25.0):
        self._path = output_path
        self._fps = fps
        self._writer = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, (w, h))
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            print(f"[INFO] Output saved → {self._path}")
