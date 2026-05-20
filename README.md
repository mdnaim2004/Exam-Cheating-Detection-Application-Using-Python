# Exam Cheating Detection Application

A real-time exam proctoring prototype built with Python, OpenCV, MediaPipe, and YOLOv8. It monitors a webcam feed and flags suspicious behavior such as looking away, a phone or book in frame, and multiple people appearing on camera.

## Features

- Head pose estimation with MediaPipe Face Mesh
- Automatic straight-face calibration before monitoring starts
- Multiple-face detection with MediaPipe Face Detection
- Phone and book detection with YOLOv8
- Configurable camera, model path, confidence threshold, and pose thresholds
- End-of-session summary with event percentages

## Setup

Use Python 3.10 or 3.11. MediaPipe does not currently work reliably on newer Python versions such as 3.13 or 3.14.

Create and activate a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the YOLOv8 model:

```bash
python scripts/download_model.py
```

## Usage

Start the proctoring session:

```bash
python main.py
```

Press `q` in the OpenCV window to stop the session and print the summary.

Useful options:

```bash
python main.py --camera 1
python main.py --confidence 0.55
python main.py --model models/yolov8n.pt
python main.py --no-display
python main.py --yaw-threshold 140 --pitch-threshold 180
```

## Notes

- The default model path is `models/yolov8n.pt`.
- If the model is missing, run `python scripts/download_model.py`.
- Detection quality depends heavily on lighting, camera placement, and calibration posture.
- This is a prototype aid, not a standalone proof of cheating. Review flagged sessions before taking action.
