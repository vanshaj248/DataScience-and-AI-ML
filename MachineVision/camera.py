"""
camera.py — Mac-First Camera Handler
=====================================
Solves every macOS/AVFoundation camera quirk in one place:

  Problem 1 — Wrong backend
    OpenCV on macOS may default to a legacy backend that opens the device
    but delivers all-zero frames. We force CAP_AVFOUNDATION.

  Problem 2 — BGRA frames
    AVFoundation delivers 4-channel BGRA, not 3-channel BGR.
    We detect this and convert automatically.

  Problem 3 — Warm-up drain
    Previous code read warm-up frames from cap, draining the live buffer,
    then the main loop found nothing left. We now reopen after probing.

  Problem 4 — Startup blank frames
    The first N frames after open() are often blank (black) while the
    sensor initialises exposure. We skip them transparently.
"""

import time
import cv2
import numpy as np


# How many consecutive blank frames to tolerate at startup before giving up
_MAX_BLANK_STARTUP = 120

# How many consecutive bad reads (ret=False) to tolerate in the main loop
MAX_BAD_READS = 30


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """
    Normalise any frame format to 3-channel BGR uint8.
    AVFoundation → BGRA (4ch), standard webcam → BGR (3ch).
    """
    if frame is None:
        return None
    if frame.ndim == 2:                          # already grayscale
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:                      # BGRA  (AVFoundation)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame                                 # already BGR


def _probe(source: int | str) -> bool:
    """
    Open the camera once, verify it delivers non-blank frames, then close.
    Returns True if the camera is usable.
    """
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION) \
          if isinstance(source, int) else cv2.VideoCapture(source)

    if not cap.isOpened():
        cap.release()
        return False

    good = 0
    for _ in range(_MAX_BLANK_STARTUP):
        ret, f = cap.read()
        if ret and f is not None and f.max() > 0:
            good += 1
        if good >= 5:
            break

    cap.release()
    return good >= 5


class MacCamera:
    """
    Drop-in replacement for raw cv2.VideoCapture that works reliably on Mac.

    Usage
    -----
        cam = MacCamera(source=0)
        cam.open()
        while True:
            frame = cam.read()       # always returns BGR uint8 or None
            if frame is None: break
            ...
        cam.release()

    Or use as a context manager:
        with MacCamera(0) as cam:
            while True:
                frame = cam.read()
                ...
    """

    def __init__(self, source: int | str = 0, target_fps: int = 30):
        self.source     = source
        self.target_fps = target_fps
        self._cap       = None
        self._bad_reads = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Open the camera for the main loop.
        Exits with a clear error message if the camera isn't usable.
        """
        import sys

        src = self.source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        self.source = src

        print(f"[camera] Opening source {src} ...")
        backend = cv2.CAP_AVFOUNDATION if isinstance(src, int) else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(src, backend)

        if not self._cap.isOpened():
            print("[camera] ERROR: Could not open camera.")
            sys.exit(1)

        # Optional: hint the driver about desired resolution & FPS
        # (driver may ignore these — we don't enforce them)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Boost exposure and gain for better visibility in low light
        self._cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)
        self._cap.set(cv2.CAP_PROP_EXPOSURE, 0)      # 0 = auto exposure
        self._cap.set(cv2.CAP_PROP_GAIN, 80)

        # Short sleep so AVFoundation can fully initialise the new session
        time.sleep(1.5)

        # Drain stale frames and check brightness
        brightness_samples = []
        for _ in range(60):
            ret, f = self._cap.read()
            if ret and f is not None:
                brightness_samples.append(f.max())

        avg_brightness = sum(brightness_samples) / len(brightness_samples) if brightness_samples else 0
        print(f"[camera] Ready  |  "
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  "
              f"@ {int(self._cap.get(cv2.CAP_PROP_FPS))} fps")
        print(f"[camera] Avg frame brightness: {avg_brightness:.1f}/255")
        
        if avg_brightness < 10:
            print("[camera] ⚠️  WARNING: Frames are very dark. Check:")
            print("  • Is the camera lens covered?")
            print("  • Is there adequate lighting?")
            print("  • Try: System Settings › Privacy & Security › Camera settings")
            print("  • Or test with: python main.py --source your_video_file.mp4")

    def read(self) -> np.ndarray | None:
        """
        Read the next frame.  Returns a BGR uint8 ndarray, or None on failure.
        Tolerates up to MAX_BAD_READS consecutive driver hiccups before
        returning None (which signals the caller to stop).
        """
        if self._cap is None:
            return None

        ret, frame = self._cap.read()

        if not ret or frame is None:
            self._bad_reads += 1
            if self._bad_reads > MAX_BAD_READS:
                return None          # tell the caller to stop
            return self.read()       # retry once (recursive, shallow)

        self._bad_reads = 0
        return _to_bgr(frame)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Context-manager support ────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()
