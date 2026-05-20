# Exam Cheating Detection Application
---

A real-time exam proctoring prototype built with Python, OpenCV, MediaPipe, and YOLOv8. It monitors a webcam feed and flags suspicious behavior such as looking away, a phone or book in frame, and multiple people appearing on camera.

## Features

- Head pose estimation with MediaPipe Face Mesh
- Automatic straight-face calibration before monitoring starts
- Multiple-face detection with MediaPipe Face Detection
- Multi-person detection using both MediaPipe face count and YOLO person count
- No-face detection when the candidate leaves the camera
- Phone, book/paper-like book, laptop, keyboard, mouse, and remote detection with YOLOv8
- Looking left/right, looking up, and looking down detection
- Eye gaze, blink-rate, mouth-open, frequent head movement, hand movement, and hand-near-face checks
- Face-cover, seat-leaving, full-body absence, camera-block, low-light, brightness anomaly, and suspicious movement checks
- Suspicion score overlay during monitoring
- Risk level classification during monitoring
- CSV event log for review after the exam
- Optional snapshot images for suspicious events
- Optional annotated video recording
- Configurable camera, model path, confidence threshold, and pose thresholds
- End-of-session summary with event percentages

## Setup

Use Python 3.10 or 3.11. MediaPipe does not currently work reliably on newer Python versions such as 3.13 or 3.14.

If you use Conda or Miniconda:

```bash
conda activate py310
python -m pip install -r requirements.txt
```

Or create and activate a normal Python virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
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

Stable mode keeps the app lighter and focuses on camera, face direction, no-face, multiple-person, and YOLO object detection. Heavier eye/hand/pose/movement analysis is optional:

```bash
python main.py --advanced-analysis
```

Press `q` in the OpenCV window to stop the session and print the summary.

Window controls:

- Drag the window border/corner to resize it manually
- Press `+` to make the window bigger
- Press `-` to make the window smaller
- Press `f` to toggle fullscreen
- Press `q` to quit

Useful options:

```bash
python main.py --camera 1
python main.py --confidence 0.25
python main.py --model models/yolov8n.pt
python main.py --no-display
python main.py --camera-width 1280 --camera-height 720 --window-width 1280 --window-height 720
python main.py --fullscreen
python main.py --yaw-threshold 140 --pitch-threshold 180
python main.py --max-seconds 600
python main.py --no-snapshot-events
python main.py --suspicious-objects "cell phone,book,laptop,keyboard,mouse,remote"
python main.py --show-all-objects --debug-detections
python main.py --record-video
```

If phone, book, or person is not detected, test with:

```bash
python main.py --confidence 0.15 --show-all-objects --debug-detections
```

The terminal will print every YOLO detection. If `cell phone`, `book`, or `person` never appears in the debug output, the issue is usually camera angle, object size, lighting, or the default YOLO model not recognizing that object in the frame.

## Session Review

Each run creates a folder inside `sessions/` with:

- `events.csv` containing suspicious event timestamps, detected people counts, details, and score
- `snapshots/` containing images from suspicious moments when snapshots are enabled

The app flags these events:

- `looking_left`, `looking_right`, `looking_up`, `looking_down`, `looking_away`
- `no_face`: no face is visible
- `multiple_people`: more than one face or YOLO person is visible
- `eye_gaze_away`, `abnormal_blink_rate`, `mouth_open`, `frequent_head_movement`
- `hand_movement`, `hand_near_face`, `face_cover`
- `seat_leaving`, `full_body_absence`, `background_movement`, `suspicious_movement`
- `camera_block`, `low_light`, `brightness_anomaly`
- `cell_phone`, `book`, `laptop`, `keyboard`, `mouse`, `remote`, or any configured suspicious object label

## Feature Coverage

Implemented in this desktop webcam version:

- Phone detection
- Book/paper-like book detection
- Looking left/right/down/up
- Multiple person detection
- No face detection
- Face cover detection by hand-near-face approximation
- Laptop/secondary-device detection when YOLO detects known labels
- Eye gaze tracking approximation
- Eye blink-rate detection
- Mouth/talking approximation by mouth-open detection
- Frequent head movement detection
- Hand movement detection
- Hand near face detection
- Seat leaving and full-body absence approximation
- Background/suspicious movement detection
- Camera block/disconnect handling
- Low-light and brightness anomaly detection
- Suspicion score and risk level
- Real-time warning overlay
- Screenshot/snapshot capture
- Optional video recording
- Incident timeline CSV logging
- Session activity timeline through `events.csv`
- Face direction and pose monitoring
- Repeated suspicious pattern logging through event cooldown and counts
- Unauthorized object detection for configured YOLO labels

Needs custom work beyond the default YOLOv8 COCO model:

- Cheat sheet, loose paper, earphone/headset, smart watch, calculator, and hidden-phone behavior detection
- Identity spoof detection and face verification
- AI-based cheating probability/classification trained on your own exam data
- PDF report generation, database logging, cloud upload, admin dashboard, multi-camera control
- Browser tab switch, Alt+Tab, copy-paste, keyboard shortcut, mouse inactivity, and screen focus monitoring
- Microphone/voice detection and audio warning system

## Notes
---

- The default model path is `models/yolov8n.pt`.
- If the model is missing, run `python scripts/download_model.py`.
- Default YOLOv8 can only detect labels it was trained for. For cheat sheets, smart watches, calculators, or earphones, train a custom YOLO model and run it with `--model`.
- Loose paper is not a default YOLOv8 COCO class. Some paper/notebook objects may be detected as `book`, but reliable paper or cheat-sheet detection needs a custom model.
- Detection quality depends heavily on lighting, camera placement, and calibration posture.
- This is a prototype aid, not a standalone proof of cheating. Review flagged sessions before taking action.
