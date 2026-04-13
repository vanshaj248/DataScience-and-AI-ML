# 🎯 Motion Detection & Tracking System

Classical computer-vision pipeline — **no deep learning, no pre-built trackers**.

---

## 📁 Project Structure

```
motion_tracker/
├── main.py          # Entry point + full CV pipeline
├── tracker.py       # Centroid-based object tracker
├── utils.py         # Drawing helpers, FPS counter, video writer
├── requirements.txt
└── README.md
```

---

## 🧠 Pipeline

```
Frame → Greyscale → Gaussian Blur → MOG2 Background Subtraction
  → Threshold → Morphological Open/Dilate → Contour Detection
  → Area Filter → Centroid Tracker → Annotated Display
```

| Stage | Why |
|---|---|
| **Grayscale** | Reduces 3-channel BGR to 1-channel intensity — faster & simpler |
| **Gaussian blur** | Kills high-frequency sensor noise before differencing |
| **MOG2 BG subtraction** | Per-pixel Gaussian Mixture Model adapts to lighting changes |
| **Threshold** | Converts soft probability mask to hard binary (foreground/background) |
| **Morphological open** | Erode → dilate removes isolated noise pixels |
| **Morphological dilate** | Expands blobs to merge nearby fragments and fill holes |
| **Contour detection** | `cv2.findContours` finds connected moving regions |
| **Area filter** | Discard blobs smaller than `--min-area` px² |
| **Centroid tracker** | Nearest-neighbour matching assigns stable IDs across frames |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Webcam
python main.py

# Video file
python main.py --source path/to/video.mp4

# All options
python main.py --source video.mp4 \
               --save output.mp4  \
               --show-mask        \
               --sensitivity 20   \
               --min-area 400     \
               --max-disappeared 50
```

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause / Resume |
| `m` | Toggle motion-mask window |
| `t` | Toggle trajectory trails |

---

## ⚙️ CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Webcam index or video file path |
| `--save` | _(none)_ | Save annotated output to MP4 |
| `--show-mask` | off | Open motion-mask window on startup |
| `--sensitivity` | `25` | Threshold (0–255); lower = more sensitive |
| `--min-area` | `500` | Min blob area in px² |
| `--max-disappeared` | `40` | Frames before lost object is dropped |
| `--no-trails` | off | Disable trajectory trails |

---

## 🔑 Tracker Algorithm

`CentroidTracker` in `tracker.py`:

1. Compute centroid `(cx, cy)` of each detected bounding box.  
2. Build a distance matrix: existing object centroids × incoming centroids.  
3. Greedy nearest-neighbour matching (sorted by distance ascending).  
4. Reject matches where distance > `max_distance` (prevents wrong merges).  
5. Register unmatched incoming centroids as new objects.  
6. Increment `disappeared` counter for unmatched existing objects; deregister after `max_disappeared` frames.  
7. Append matched centroid to a fixed-length **trail** for trajectory drawing.
