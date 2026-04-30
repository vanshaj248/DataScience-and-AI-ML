"""
display.py — Drawing & Overlay Helpers
=======================================
All visual output in one place: bounding boxes, trails, HUD, mask window.
"""

import time
import cv2
import numpy as np



_PALETTE_RGB = [
    (255, 80,  80),  (80,  255, 80),  (80,  180, 255), (255, 180, 80),
    (180, 80,  255), (255, 255, 80),  (80,  255, 220), (255, 80,  180),
    (160, 255, 80),  (80,  120, 255), (255, 140, 0),   (0,   255, 255),
    (255, 0,   128), (128, 255, 0),   (0,   128, 255), (255, 128, 0),
    (128, 0,   255), (0,   255, 128), (220, 50,  50),  (50,  50,  220),
]

def _colour(oid: int) -> tuple:
    r, g, b = _PALETTE_RGB[oid % len(_PALETTE_RGB)]
    return (b, g, r)   # BGR for OpenCV



class FPSCounter:
    def __init__(self, window: int = 30):
        self._ts = []
        self._n  = window

    def tick(self) -> float:
        self._ts.append(time.monotonic())
        if len(self._ts) > self._n:
            self._ts.pop(0)
        if len(self._ts) < 2:
            return 0.0
        elapsed = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / elapsed if elapsed > 0 else 0.0



def draw_box(frame: np.ndarray, rect: tuple, oid: int) -> None:
    """Draw a colour-coded bounding box with an ID label."""
    x, y, w, h = rect
    col = _colour(oid)

    cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)

    label = f" ID {oid} "
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lx1, ly1 = x, max(y - th - bl - 4, 0)
    lx2, ly2 = x + tw + 4, max(y - 2, th + 4)
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), col, -1)
    cv2.putText(frame, label, (x + 2, max(y - bl - 2, th + 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_centroid(frame: np.ndarray, pt: tuple, oid: int) -> None:
    """Draw a small filled dot at the centroid."""
    col = _colour(oid)
    cv2.circle(frame, pt, 5, col, -1)
    cv2.circle(frame, pt, 5, (255, 255, 255), 1)


def draw_trail(frame: np.ndarray, trail: list, oid: int) -> None:
    """
    Draw the motion trajectory as a fading polyline.
    Segments fade from dim (oldest) to full-colour (newest).
    """
    if len(trail) < 2:
        return
    col = _colour(oid)
    n   = len(trail)
    for i in range(1, n):
        fade    = i / n                              # 0 → 1
        c       = tuple(int(v * (0.25 + 0.75 * fade)) for v in col)
        thick   = max(1, int(fade * 3))
        cv2.line(frame, trail[i - 1], trail[i], c, thick, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, fps: float, count: int) -> None:
    """Overlay FPS (top-right), object count (top-left), hint (bottom-left)."""
    h, w = frame.shape[:2]

    cv2.putText(frame, f"Objects: {count}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    fps_txt = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, fps_txt, (w - tw - 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, "q:quit  p:pause  m:mask  t:trails",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (160, 160, 160), 1, cv2.LINE_AA)


def draw_calibrating(frame: np.ndarray) -> None:
    """Overlay 'calibrating' banner on the first few frames."""
    cv2.putText(frame, "Calibrating background — please wait...",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 165, 255), 2, cv2.LINE_AA)



def show_mask_window(diff: np.ndarray, mask: np.ndarray) -> None:
    """
    Side-by-side debug window:
      Left  — raw diff image (normalised), shows what the detector sees
      Right — cleaned binary mask coloured with COLORMAP_HOT
    Shows content even when nothing is moving (left panel always has data).
    """
    diff_norm  = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    diff_bgr   = cv2.cvtColor(diff_norm, cv2.COLOR_GRAY2BGR)
    mask_colour= cv2.applyColorMap(mask, cv2.COLORMAP_HOT)

    cv2.putText(diff_bgr,    "Raw diff",    (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(mask_colour, "Clean mask",  (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    h = diff_bgr.shape[0]
    combined = np.hstack([
        cv2.resize(diff_bgr,    (diff_bgr.shape[1],    h)),
        cv2.resize(mask_colour, (mask_colour.shape[1], h)),
    ])
    cv2.imshow("Motion Mask", combined)



class VideoWriter:
    """Lazy-init wrapper around cv2.VideoWriter."""

    def __init__(self, path: str, fps: float = 25.0):
        self._path   = path
        self._fps    = fps
        self._writer = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w   = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, (w, h))
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer:
            self._writer.release()
            print(f"[display] Saved output → {self._path}")
