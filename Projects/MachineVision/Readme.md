# Motion Detection & Tracking

A real-time motion detection and object tracking application built specifically for macOS, using OpenCV and Python. This project provides reliable camera handling, background subtraction-based motion detection, centroid tracking, and visual overlays for monitoring moving objects in video streams.

## Features

- **Real-Time Motion Detection**: Frame-differencing algorithm with rolling background model
- **Object Tracking**: Centroid-based tracking with unique ID assignment and trajectory trails
- **Visual Overlays**: Bounding boxes, centroids, motion trails, and HUD with FPS/object count
- **Debug Tools**: Optional motion mask visualization for tuning detection parameters
- **Video Recording**: Save annotated output to MP4 files
- **Runtime Controls**: Keyboard shortcuts for pause/resume, mask toggle, trail toggle

## Requirements

- Python 3.8+
- macOS (optimized for AVFoundation cameras)
- OpenCV 4.5+
- NumPy

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or manually:
   ```bash
   pip install opencv-python numpy
   ```

## Usage

### Basic Usage

Run with default webcam:
```bash
python main.py
```

### Command Line Options

- `--source`: Camera index (default 0) or path to video file
- `--save`: Save annotated output to MP4 file
- `--show-mask`: Enable motion mask debug window
- `--sensitivity`: Motion threshold (0-255, lower = more sensitive, default 25)
- `--min-area`: Minimum blob area in pixels² (default 800)
- `--max-disappeared`: Frames before lost object is removed (default 50)
- `--no-trails`: Disable trajectory trails
- `--bg-alpha`: Background learning rate (0-1, default 0.05)

### Examples

Use second camera:
```bash
python main.py --source 1
```

Process video file:
```bash
python main.py --source video.mp4
```

Save output with custom sensitivity:
```bash
python main.py --save output.mp4 --sensitivity 15
```

Debug motion detection:
```bash
python main.py --show-mask --sensitivity 20
```

### Runtime Controls

- `q` - Quit application
- `p` - Pause/resume processing
- `m` - Toggle motion mask debug window
- `t` - Toggle trajectory trails

## Architecture

The application is organized into modular components:

### `main.py`
Main application entry point with argument parsing, main loop, and orchestration of all components.

### `camera.py`
Mac-optimized camera handler that:
- Uses AVFoundation backend for macOS cameras
- Handles frame format normalization (BGR/BGRA conversion)
- Implements robust error handling and brightness checking
- Provides context manager support

### `detector.py`
Motion detection using frame differencing:
- Rolling weighted-average background model
- Gaussian blur for noise reduction
- Morphological operations for mask cleaning
- Contour detection and bounding box extraction

### `tracker.py`
Centroid-based object tracking:
- Nearest-neighbor matching algorithm
- Unique ID assignment and persistence
- Trajectory trail recording
- Automatic object deregistration for lost objects

### `display.py`
Visualization and overlay utilities:
- Color-coded bounding boxes and labels
- Centroid markers and fading trails
- HUD with FPS and object count
- Debug motion mask window
- Video recording wrapper

## Configuration

### Motion Detection Tuning

- **Sensitivity** (`--sensitivity`): Lower values detect more motion but increase false positives
- **Minimum Area** (`--min-area`): Filter out small noise blobs
- **Background Alpha** (`--bg-alpha`): How quickly background adapts (lower = slower adaptation)

### Tracking Parameters

- **Max Disappeared** (`--max-disappeared`): Frames to wait before removing lost objects
- **Max Distance**: Maximum pixel distance for centroid matching (configured in code)

## Troubleshooting

### Camera Issues

**Dark frames or no video:**
- Check camera permissions in System Settings → Privacy & Security → Camera
- Ensure adequate lighting
- Try different camera index: `python main.py --source 1`

**Camera won't open:**
- Close other applications using the camera
- Restart the computer
- Check camera hardware connections

### Detection Issues

**Too many false positives:**
- Increase `--sensitivity` (higher threshold)
- Increase `--min-area` (larger minimum blob size)
- Use `--show-mask` to visualize detection

**Missing real motion:**
- Decrease `--sensitivity` (lower threshold)
- Decrease `--min-area` (smaller minimum blob size)
- Check lighting conditions

### Performance Issues

**Low FPS:**
- Reduce video resolution in `camera.py` (currently 1280x720)
- Increase `--bg-alpha` for faster background updates
- Close other applications

## License

This project is open source. See individual file headers for licensing information.

## Contributing

Contributions welcome! Please ensure code works on macOS and includes appropriate documentation.