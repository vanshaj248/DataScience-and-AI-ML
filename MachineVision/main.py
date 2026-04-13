"""
main.py — Motion Detection & Tracking System
============================================
Classical computer-vision pipeline (no deep learning).

Pipeline overview
-----------------
  Video frame
      │
      ▼
  Grayscale conversion          ← reduce to 1-channel intensity image
      │
      ▼
  Gaussian blur                 ← suppress high-frequency noise
      │
      ▼
  Background subtraction        ← per-pixel difference vs rolling average
      │
      ▼
  Absolute difference + thresh  ← isolate significant motion pixels
      │
      ▼
  Morphological ops (open/dilate) ← remove noise, fill holes
      │
      ▼
  Contour detection             ← find connected moving regions
      │
      ▼
  Area filtering                ← discard tiny spurious blobs
      │
      ▼
  Bounding boxes → Tracker      ← assign / maintain unique object IDs
      │
      ▼
  Draw overlays + display

Usage
-----
  # Webcam (default)
  python main.py

  # Video file
  python main.py --source path/to/video.mp4

  # Save output, show mask window, tune sensitivity
  python main.py --source video.mp4 --save output.mp4 --show-mask --sensitivity 20

  Controls
  --------
  q  — quit
  p  — pause / resume
  m  — toggle motion-mask window
  t  — toggle trajectory trails
"""

import argparse
import sys

import cv2
import numpy as np

from tracker import CentroidTracker
from utils import (
    FPSCounter, VideoWriter,
    draw_bounding_box, draw_centroid, draw_trail,
    draw_fps, draw_object_count, show_motion_mask
)


# ─────────────────────────────────────────────────────────────────────────────
# Command-line arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classical Motion Detection & Tracking System"
    )
    parser.add_argument(
        "--source", default=0,
        help="Video source: 0 (webcam), integer device index, or path to video file."
    )
    parser.add_argument(
        "--save", default="",
        help="If set, save annotated output to this file path (e.g. output.mp4)."
    )
    parser.add_argument(
        "--show-mask", action="store_true",
        help="Open a second window showing the binary motion mask."
    )
    parser.add_argument(
        "--sensitivity", type=int, default=25,
        help="Motion threshold (0-255). Lower = more sensitive. Default: 25."
    )
    parser.add_argument(
        "--min-area", type=int, default=500,
        help="Minimum contour area (px²) to be considered an object. Default: 500."
    )
    parser.add_argument(
        "--max-disappeared", type=int, default=40,
        help="Frames before a lost object is deregistered. Default: 40."
    )
    parser.add_argument(
        "--no-trails", action="store_true",
        help="Disable trajectory trails."
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Background model helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_background_subtractor():
    """
    Use OpenCV's MOG2 background subtractor.
    It maintains a Gaussian Mixture Model per pixel, adapting to slow
    lighting changes automatically — more robust than a simple running mean.

    detectShadows=False speeds things up and avoids grey shadow regions
    being misclassified as foreground.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False
    )


def preprocess_frame(frame: np.ndarray,
                     blur_ksize: int = 21) -> np.ndarray:
    """
    Convert to grayscale and apply Gaussian blur.

    Parameters
    ----------
    blur_ksize : int — kernel size (must be odd). Larger = more blur = less noise
                       but also less sensitivity to small/slow motion.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blur: smooths out sensor noise so tiny single-pixel
    # fluctuations don't trigger false positives.
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return blurred


def compute_motion_mask(fg_mask: np.ndarray,
                        threshold: int = 25) -> np.ndarray:
    """
    Post-process the raw foreground mask from MOG2.

    Steps
    -----
    1. Binary threshold  — convert grey-level probabilities to hard 0/255.
    2. Morphological open (erosion then dilation)
         — removes small isolated noise pixels.
    3. Morphological dilate
         — expands blobs slightly so nearby fragments merge into one contour.
    """
    # 1. Hard threshold: pixels brighter than `threshold` become white (255)
    _, binary = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)

    # 2. Opening: erode first (shrinks), then dilate (expands)
    #    Net effect: tiny isolated blobs disappear, large blobs mostly survive.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # 3. Dilation: expand surviving blobs to fill small internal holes
    #    and connect nearby fragments of the same object.
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(opened, kernel_dilate, iterations=3)

    return dilated


def find_detections(mask: np.ndarray,
                    min_area: int = 500) -> list:
    """
    Find bounding-box rectangles of significant moving objects.

    Parameters
    ----------
    mask     : binary motion mask (uint8, 0 or 255)
    min_area : contours smaller than this many pixels are ignored

    Returns
    -------
    List of (x, y, w, h) tuples, one per detected object.
    """
    # cv2.RETR_EXTERNAL: only outermost contours (no nested children)
    # cv2.CHAIN_APPROX_SIMPLE: compress horizontal/vertical/diagonal
    #   runs to just their endpoints, saving memory.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Skip tiny blobs — they are usually noise or micro-vibrations
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        detections.append((x, y, w, h))

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:

    # ── Open video source ────────────────────────────────────────────────────
    source = args.source
    # Allow passing an integer string from the command line ("0", "1" …)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    print(f"[INFO] Opened source: {source}")
    print("[INFO] Controls:  q=quit  p=pause  m=toggle mask  t=toggle trails")

    # ── Initialise components ────────────────────────────────────────────────
    bg_subtractor = build_background_subtractor()

    tracker = CentroidTracker(
        max_disappeared=args.max_disappeared,
        max_distance=80,
        max_trail_len=40
    )

    fps_counter  = FPSCounter(avg_over=30)
    video_writer = VideoWriter(args.save) if args.save else None

    # Runtime toggles (can be flipped with keyboard shortcuts)
    show_mask  = args.show_mask
    show_trail = not args.no_trails
    paused     = False

    # Warm-up: skip first few frames so MOG2 can initialise its model
    WARMUP_FRAMES = 30
    warmup_count  = 0

    while True:
        # ── Keyboard input ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("[INFO] Paused" if paused else "[INFO] Resumed")
        elif key == ord('m'):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Motion Mask")
        elif key == ord('t'):
            show_trail = not show_trail

        if paused:
            continue

        # ── Read frame ───────────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video / no frame received.")
            break

        # Optionally resize large frames for speed (uncomment if needed)
        # frame = cv2.resize(frame, (960, 540))

        # ── Pre-process ───────────────────────────────────────────────────
        blurred = preprocess_frame(frame, blur_ksize=21)

        # ── Background subtraction ────────────────────────────────────────
        # MOG2 returns a grey-scale "foreground probability" mask.
        # We pass the blurred (noise-reduced) frame for better quality.
        fg_mask = bg_subtractor.apply(blurred)

        # ── Motion mask clean-up ──────────────────────────────────────────
        clean_mask = compute_motion_mask(fg_mask, threshold=args.sensitivity)

        # During warm-up MOG2 is still learning the background — skip tracking
        warmup_count += 1
        if warmup_count < WARMUP_FRAMES:
            cv2.putText(frame, "Calibrating background...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.imshow("Motion Tracker", frame)
            continue

        # ── Find detections ───────────────────────────────────────────────
        detections = find_detections(clean_mask, min_area=args.min_area)

        # ── Update tracker ────────────────────────────────────────────────
        objects = tracker.update(detections)

        # Build a quick lookup: centroid → rect (for drawing)
        # We match each tracked centroid to the closest detection bounding box.
        centroid_to_rect: dict = {}
        for rect in detections:
            rx, ry, rw, rh = rect
            rc_x = rx + rw // 2
            rc_y = ry + rh // 2
            # Find the tracked object whose centroid is nearest this detection
            best_id, best_dist = None, float("inf")
            for obj_id, (cx, cy) in objects.items():
                d = ((cx - rc_x) ** 2 + (cy - rc_y) ** 2) ** 0.5
                if d < best_dist:
                    best_dist, best_id = d, obj_id
            if best_id is not None:
                centroid_to_rect[best_id] = rect

        # ── Draw overlays ─────────────────────────────────────────────────
        for obj_id, centroid in objects.items():

            # Bounding box (if we have a detection for this object this frame)
            if obj_id in centroid_to_rect:
                draw_bounding_box(frame, centroid_to_rect[obj_id], obj_id)

            # Centroid dot
            draw_centroid(frame, centroid, obj_id)

            # Motion trail
            if show_trail and obj_id in tracker.trails:
                draw_trail(frame, tracker.trails[obj_id], obj_id)

        # ── HUD ───────────────────────────────────────────────────────────
        fps = fps_counter.tick()
        draw_fps(frame, fps)
        draw_object_count(frame, len(objects))

        # Key-binding reminder (bottom-left)
        hint = "q:quit  p:pause  m:mask  t:trails"
        cv2.putText(frame, hint, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        # ── Optional mask window ──────────────────────────────────────────
        if show_mask:
            show_motion_mask(clean_mask)

        # ── Display main window ───────────────────────────────────────────
        cv2.imshow("Motion Tracker", frame)

        # ── Save output ───────────────────────────────────────────────────
        if video_writer:
            video_writer.write(frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(parse_args())
