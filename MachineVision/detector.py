"""
detector.py — Frame-Differencing Motion Detector
=================================================
Uses a rolling weighted-average background model instead of MOG2.

Why not MOG2?
    MOG2's C++ internals silently fail on macOS AVFoundation cameras —
    they receive valid frames but return an all-zero mask. A plain
    numpy weighted average is transparent, debuggable, and works on
    every frame format.

Pipeline per frame
------------------
  BGR frame
    → Grayscale + Gaussian blur      (noise reduction)
    → |frame - background|           (absolute difference)
    → Background update              (slow rolling average)
    → Binary threshold               (hard foreground/background split)
    → Morphological open             (kill noise specks)
    → Morphological dilate           (fill holes, merge fragments)
    → findContours + area filter     (list of bounding boxes)
"""

import cv2
import numpy as np


class MotionDetector:
    """
    Detects moving objects and returns bounding-box rectangles.

    Parameters
    ----------
    bg_alpha   : float   Background learning rate (0=frozen, 1=instant).
                         0.05 means the background updates at 5 % per frame;
                         a walking person stays visible for ~20 s.
    threshold  : int     Pixel-difference threshold (0-255).
                         Lower → more sensitive (more false positives).
                         Higher → less sensitive (misses subtle motion).
    min_area   : int     Minimum contour area (px²) to count as an object.
                         Increase to ignore small vibrations / noise.
    blur_ksize : int     Gaussian blur kernel size (must be odd).
    """

    def __init__(self,
                 bg_alpha:   float = 0.05,
                 threshold:  int   = 200,
                 min_area:   int   = 10000,
                 blur_ksize: int   = 21):

        self.bg_alpha   = bg_alpha
        self.threshold  = threshold
        self.min_area   = min_area
        self.blur_ksize = blur_ksize

        self._bg:        np.ndarray | None = None   # float32 background model
        self._last_diff: np.ndarray | None = None   # raw diff (for debug)
        self._last_mask: np.ndarray | None = None   # cleaned mask (for debug)


    def update(self, frame: np.ndarray) -> list[tuple]:
        """
        Feed the next BGR frame.

        Returns
        -------
        List of (x, y, w, h) bounding boxes for detected moving objects.
        Empty list if nothing moves or it's the first frame.
        """
        gray    = self._preprocess(frame)
        diff    = self._diff_and_update(gray)
        mask    = self._clean_mask(diff)

        self._last_diff = diff
        self._last_mask = mask

        return self._find_boxes(mask)

    @property
    def debug_diff(self) -> np.ndarray | None:
        """Raw absolute-difference image (uint8) from the last update()."""
        return self._last_diff

    @property
    def debug_mask(self) -> np.ndarray | None:
        """Cleaned binary motion mask from the last update()."""
        return self._last_mask


    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        BGR (or BGRA) → grayscale → Gaussian blur.

        Gaussian blur suppresses high-frequency sensor noise so that tiny
        single-pixel fluctuations don't register as motion.
        """
        if frame.ndim == 2:
            gray = frame
        elif frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        k = self.blur_ksize
        return cv2.GaussianBlur(gray, (k, k), 0)

    def _diff_and_update(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute absolute difference between current frame and background,
        then update the background model.

        The update uses cv2.accumulateWeighted which implements:
            bg = (1 - alpha) * bg  +  alpha * current

        We compute the diff BEFORE updating the background so that the
        current frame is compared against the previous background state,
        not one that already includes itself.
        """
        f32 = gray.astype(np.float32)

        if self._bg is None:
            self._bg = f32.copy()
            return np.zeros_like(gray)

        bg_uint8 = np.clip(self._bg, 0, 255).astype(np.uint8)
        diff     = cv2.absdiff(gray, bg_uint8)

        cv2.accumulateWeighted(f32, self._bg, self.bg_alpha)

        return diff

    def _clean_mask(self, diff: np.ndarray) -> np.ndarray:
        """
        Turn the raw difference image into a clean binary motion mask.

        Steps
        -----
        1. Threshold   — pixels brighter than self.threshold become 255
        2. Morph open  — erode then dilate removes isolated noise specks
        3. Morph dilate — expand blobs to fill holes and connect fragments
        """
        _, binary = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=2)

        k_dil   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(opened, k_dil, iterations=3)

        return dilated

    def _find_boxes(self, mask: np.ndarray) -> list[tuple]:
        """
        Find contours in the binary mask and return bounding boxes
        for all contours whose area exceeds self.min_area.

        cv2.RETR_EXTERNAL  — only outermost contours (no nested holes)
        cv2.CHAIN_APPROX_SIMPLE — compress runs to endpoints (saves memory)
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_area:
                boxes.append(cv2.boundingRect(cnt))
        return boxes
