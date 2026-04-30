"""
main.py — Motion Detection & Tracking  (Mac Edition)
=====================================================

Usage
-----
  python main.py                        # built-in webcam
  python main.py --source 1             # second camera
  python main.py --source clip.mp4      # video file
  python main.py --show-mask            # open motion-mask debug window
  python main.py --sensitivity 15       # more sensitive (default 25)
  python main.py --save output.mp4      # save annotated video

Runtime controls
----------------
  q — quit            p — pause / resume
  m — mask on/off     t — trails on/off
"""

import argparse
import sys

import cv2

from camera   import MacCamera
from detector import MotionDetector
from tracker  import CentroidTracker
from display  import (
    FPSCounter, VideoWriter,
    draw_box, draw_centroid, draw_trail,
    draw_hud, draw_calibrating, show_mask_window,
)


CALIBRATION_FRAMES = 45


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Motion Detection & Tracking — Mac")
    p.add_argument("--source",          default=0,
                   help="Camera index (default 0) or path to video file")
    p.add_argument("--save",            default="",
                   help="Save annotated output to this .mp4 path")
    p.add_argument("--show-mask",       action="store_true",
                   help="Open a side-by-side motion-mask debug window")
    p.add_argument("--sensitivity",     type=int, default=25,
                   help="Motion threshold 0-255 (lower = more sensitive, default 25)")
    p.add_argument("--min-area",        type=int, default=800,
                   help="Minimum blob area in px² (default 800)")
    p.add_argument("--max-disappeared", type=int, default=50,
                   help="Frames before a lost object is removed (default 50)")
    p.add_argument("--no-trails",       action="store_true",
                   help="Disable trajectory trails")
    p.add_argument("--bg-alpha",        type=float, default=0.05,
                   help="Background learning rate 0-1 (default 0.05)")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:

    cam      = MacCamera(source=args.source)
    detector = MotionDetector(
        bg_alpha  = args.bg_alpha,
        threshold = args.sensitivity,
        min_area  = args.min_area,
    )
    tracker  = CentroidTracker(max_disappeared=args.max_disappeared)
    fps_ctr  = FPSCounter()
    writer   = VideoWriter(args.save) if args.save else None

    show_mask  = args.show_mask
    show_trail = not args.no_trails
    paused     = False
    frame_num  = 0

    cam.open()

    cv2.namedWindow("Motion Tracker", cv2.WINDOW_NORMAL)

    print("[main] Running — press q to quit")

    while True:

        frame = cam.read()
        if frame is None:
            print("[main] Stream ended.")
            break

        frame_num += 1

        if paused:
            cv2.imshow("Motion Tracker", frame)
            key = cv2.waitKey(30) & 0xFF
            if   key == ord('q'): break
            elif key == ord('p'):
                paused = False
                print("[main] Resumed")
            continue

        detections = detector.update(frame)

        calibrating = (frame_num <= CALIBRATION_FRAMES)

        objects = {} if calibrating else tracker.update(detections)

        id_to_rect: dict = {}
        for rect in detections:
            rx, ry, rw, rh = rect
            rc = (rx + rw // 2, ry + rh // 2)
            best_id, best_d = None, float("inf")
            for oid, (cx, cy) in objects.items():
                d = ((cx - rc[0]) ** 2 + (cy - rc[1]) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_id = d, oid
            if best_id is not None:
                id_to_rect[best_id] = rect

        for oid, centroid in objects.items():
            if oid in id_to_rect:
                draw_box(frame, id_to_rect[oid], oid)
            draw_centroid(frame, centroid, oid)
            if show_trail:
                draw_trail(frame, tracker.trails.get(oid, []), oid)

        if calibrating:
            draw_calibrating(frame)

        draw_hud(frame, fps_ctr.tick(), len(objects))

        if show_mask and detector.debug_diff is not None:
            show_mask_window(detector.debug_diff, detector.debug_mask)

        cv2.imshow("Motion Tracker", frame)
        if writer:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('p'):
            paused = not paused
            print("[main] Paused" if paused else "[main] Resumed")
        elif key == ord('m'):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Motion Mask")
        elif key == ord('t'):
            show_trail = not show_trail

    cam.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[main] Done.")


if __name__ == "__main__":
    run(parse_args())
