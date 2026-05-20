import argparse
import csv
import importlib
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path


DEFAULT_SUSPICIOUS_OBJECTS = {
    "cell phone",
    "book",
    "laptop",
    "keyboard",
    "mouse",
    "remote",
}
LABEL_ALIASES = {
    "mobile": "cell phone",
    "phone": "cell phone",
    "cellphone": "cell phone",
    "cell": "cell phone",
    "paper": "book",
    "notebook": "book",
    "person": "person",
    "people": "person",
}
EVENT_WEIGHTS = {
    "looking_away": 1,
    "looking_left": 1,
    "looking_right": 1,
    "looking_up": 1,
    "looking_down": 2,
    "eye_gaze_away": 2,
    "abnormal_blink_rate": 1,
    "mouth_open": 1,
    "frequent_head_movement": 2,
    "face_cover": 3,
    "hand_movement": 1,
    "hand_near_face": 2,
    "seat_leaving": 4,
    "full_body_absence": 3,
    "background_movement": 2,
    "suspicious_movement": 2,
    "camera_block": 5,
    "low_light": 2,
    "brightness_anomaly": 2,
    "no_face": 2,
    "book": 2,
    "cell_phone": 3,
    "laptop": 3,
    "keyboard": 2,
    "mouse": 1,
    "remote": 2,
    "multiple_people": 4,
}
MODEL_POINTS_DATA = (
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
)

cv2 = None
mp = None
np = None
YOLO = None
MODEL_POINTS = None


def load_runtime_dependencies():
    global cv2, mp, np, YOLO, MODEL_POINTS

    cache_dir = Path(".cache")
    matplotlib_cache = cache_dir / "matplotlib"
    ultralytics_cache = cache_dir / "ultralytics"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    ultralytics_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_cache.resolve()))

    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "This project depends on MediaPipe, which is not supported on your active "
            f"Python {sys.version.split()[0]}. Use Python 3.10 or 3.11, then run "
            "'pip install -r requirements.txt'."
        )

    missing = []
    modules = {}
    for module_name in ("cv2", "mediapipe", "numpy", "ultralytics"):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)

    if missing:
        packages = ", ".join(missing)
        raise RuntimeError(
            f"Missing Python package(s): {packages}. Install them with "
            f"'{sys.executable} -m pip install -r requirements.txt'. "
            f"Current Python: {sys.executable}"
        )

    cv2 = modules["cv2"]
    mp = modules["mediapipe"]
    np = modules["numpy"]
    YOLO = modules["ultralytics"].YOLO
    MODEL_POINTS = np.array(MODEL_POINTS_DATA, dtype=np.float64)


def normalize_label(label):
    normalized = label.strip().lower().replace("_", " ")
    return LABEL_ALIASES.get(normalized, normalized)


def normalize_labels(labels):
    return {normalize_label(label) for label in labels if label.strip()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time exam proctoring with head pose, face, and object detection."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index to open.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/yolov8n.pt"),
        help="Path to the YOLO model file.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum YOLO confidence for suspicious object detections.",
    )
    parser.add_argument(
        "--pre-calibration-seconds",
        type=int,
        default=5,
        help="Countdown before collecting calibration frames.",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=int,
        default=3,
        help="Number of seconds used to calibrate a straight-facing pose.",
    )
    parser.add_argument(
        "--yaw-threshold",
        type=float,
        default=148.5,
        help="Maximum allowed yaw offset from the calibrated pose.",
    )
    parser.add_argument(
        "--pitch-threshold",
        type=float,
        default=189.0,
        help="Maximum allowed pitch offset from the calibrated pose.",
    )
    parser.add_argument(
        "--direction-yaw-threshold",
        type=float,
        default=30.0,
        help="Yaw offset used for left/right direction warnings.",
    )
    parser.add_argument(
        "--direction-pitch-threshold",
        type=float,
        default=25.0,
        help="Pitch offset used for up/down direction warnings.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the live OpenCV monitoring window.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Requested webcam capture width.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Requested webcam capture height.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1280,
        help="OpenCV display window width.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=720,
        help="OpenCV display window height.",
    )
    parser.add_argument(
        "--fullscreen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the monitoring window in fullscreen mode.",
    )
    parser.add_argument(
        "--suspicious-objects",
        default="cell phone,book,laptop,keyboard,mouse,remote",
        help="Comma-separated YOLO labels to flag as suspicious objects.",
    )
    parser.add_argument(
        "--show-all-objects",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw all YOLO detections, not only suspicious labels and people.",
    )
    parser.add_argument(
        "--debug-detections",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print YOLO detections to the terminal for troubleshooting.",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=640,
        help="YOLO inference image size. Larger can help small objects but runs slower.",
    )
    parser.add_argument(
        "--advanced-analysis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable heavier hand, pose, eye, blink, mouth, and movement analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sessions"),
        help="Directory where session logs and snapshots are saved.",
    )
    parser.add_argument(
        "--snapshot-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a snapshot image when a suspicious event is first logged.",
    )
    parser.add_argument(
        "--event-cooldown",
        type=float,
        default=2.0,
        help="Minimum seconds between repeated log entries for the same event type.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=0,
        help="Automatically stop after this many monitoring seconds. Use 0 for no limit.",
    )
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save an annotated MP4 recording for the session.",
    )
    parser.add_argument(
        "--low-light-threshold",
        type=float,
        default=45.0,
        help="Average grayscale brightness below this value triggers low-light detection.",
    )
    parser.add_argument(
        "--bright-light-threshold",
        type=float,
        default=235.0,
        help="Average grayscale brightness above this value triggers brightness anomaly.",
    )
    parser.add_argument(
        "--camera-block-std-threshold",
        type=float,
        default=12.0,
        help="Very low frame contrast below this value can indicate a blocked camera.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=18.0,
        help="Frame-difference score above this value triggers movement detection.",
    )
    parser.add_argument(
        "--blink-rate-threshold",
        type=int,
        default=30,
        help="Blinks per minute above this value trigger abnormal blink-rate detection.",
    )
    parser.add_argument(
        "--face-confidence",
        type=float,
        default=0.35,
        help="Minimum MediaPipe face detection confidence.",
    )
    parser.add_argument(
        "--hand-confidence",
        type=float,
        default=0.4,
        help="Minimum MediaPipe hand detection confidence.",
    )
    parser.add_argument(
        "--pose-confidence",
        type=float,
        default=0.4,
        help="Minimum MediaPipe pose detection confidence.",
    )
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1.")
    for name in ("face_confidence", "hand_confidence", "pose_confidence"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.pre_calibration_seconds < 0 or args.calibration_seconds < 0:
        parser.error("Calibration durations must be zero or greater.")
    if args.event_cooldown < 0:
        parser.error("--event-cooldown must be zero or greater.")
    if args.max_seconds < 0:
        parser.error("--max-seconds must be zero or greater.")
    if args.direction_yaw_threshold < 0 or args.direction_pitch_threshold < 0:
        parser.error("Direction thresholds must be zero or greater.")
    if min(args.camera_width, args.camera_height, args.window_width, args.window_height) <= 0:
        parser.error("Camera and window dimensions must be greater than zero.")
    args.suspicious_objects = normalize_labels(args.suspicious_objects.split(","))
    if not args.suspicious_objects:
        args.suspicious_objects = set(DEFAULT_SUSPICIOUS_OBJECTS)
    return args


def estimate_head_pose(landmarks, width, height):
    image_points = np.array(
        [
            (landmarks[1].x * width, landmarks[1].y * height),
            (landmarks[152].x * width, landmarks[152].y * height),
            (landmarks[33].x * width, landmarks[33].y * height),
            (landmarks[263].x * width, landmarks[263].y * height),
            (landmarks[61].x * width, landmarks[61].y * height),
            (landmarks[291].x * width, landmarks[291].y * height),
        ],
        dtype=np.float64,
    )

    focal_length = width
    camera_matrix = np.array(
        [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, np.zeros((4, 1))
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    return angles


def is_looking_away(pitch, yaw, calibrated_pitch, calibrated_yaw, pitch_threshold, yaw_threshold):
    pitch_offset = abs(pitch - calibrated_pitch)
    yaw_offset = abs(yaw - calibrated_yaw)
    return pitch_offset > pitch_threshold or yaw_offset > yaw_threshold


def landmark_point(landmarks, index, width, height):
    landmark = landmarks[index]
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float64)


def point_distance(point_a, point_b):
    return float(np.linalg.norm(point_a - point_b))


def eye_aspect_ratio(landmarks, indices, width, height):
    points = [landmark_point(landmarks, index, width, height) for index in indices]
    vertical_1 = point_distance(points[1], points[5])
    vertical_2 = point_distance(points[2], points[4])
    horizontal = max(point_distance(points[0], points[3]), 1.0)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_open_ratio(landmarks, width, height):
    upper_lip = landmark_point(landmarks, 13, width, height)
    lower_lip = landmark_point(landmarks, 14, width, height)
    left_corner = landmark_point(landmarks, 61, width, height)
    right_corner = landmark_point(landmarks, 291, width, height)
    return point_distance(upper_lip, lower_lip) / max(point_distance(left_corner, right_corner), 1.0)


def iris_horizontal_ratio(landmarks, iris_index, left_corner_index, right_corner_index, width, height):
    iris = landmark_point(landmarks, iris_index, width, height)
    left_corner = landmark_point(landmarks, left_corner_index, width, height)
    right_corner = landmark_point(landmarks, right_corner_index, width, height)
    return (iris[0] - left_corner[0]) / max(right_corner[0] - left_corner[0], 1.0)


def draw_text(frame, text, x, y, color=(255, 255, 255), scale=0.7, thickness=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def risk_level(score):
    if score >= 10:
        return "HIGH"
    if score >= 5:
        return "MEDIUM"
    if score > 0:
        return "LOW"
    return "NORMAL"


def draw_status_panel(frame, stats, face_count, person_count, active_events, suspicion_score):
    height, width, _ = frame.shape
    panel_width = 330
    panel_height = 150
    x1 = max(0, width - panel_width - 12)
    y1 = height - panel_height - 12
    x2 = width - 12
    y2 = height - 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

    score_color = (0, 255, 0)
    if suspicion_score >= 5:
        score_color = (0, 165, 255)
    if suspicion_score >= 8:
        score_color = (0, 0, 255)

    lines = [
        f"Risk: {risk_level(suspicion_score)} | Score: {suspicion_score}",
        f"Faces: {face_count} | YOLO people: {person_count}",
        f"Frames: {stats['total_frames']}",
        "Active: " + (", ".join(active_events) if active_events else "normal"),
    ]
    for index, line in enumerate(lines):
        color = score_color if index == 0 else (255, 255, 255)
        draw_text(frame, line, x1 + 12, y1 + 30 + index * 30, color, 0.6, 2)


def get_face_count(face_detection_results):
    return len(face_detection_results.detections or [])


def draw_face_boxes(frame, face_detection_results):
    height, width, _ = frame.shape
    for detection in face_detection_results.detections or []:
        bbox = detection.location_data.relative_bounding_box
        x = max(0, int(bbox.xmin * width))
        y = max(0, int(bbox.ymin * height))
        w = int(bbox.width * width)
        h = int(bbox.height * height)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)


def face_bbox_from_landmarks(landmarks, width, height):
    xs = [landmark.x * width for landmark in landmarks]
    ys = [landmark.y * height for landmark in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float64)


def point_in_bbox(point, bbox, padding=0):
    x1, y1, x2, y2 = bbox
    return x1 - padding <= point[0] <= x2 + padding and y1 - padding <= point[1] <= y2 + padding


def configure_display(window_name, args):
    if not args.display:
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, args.window_width, args.window_height)


def display_frame(frame, args):
    return frame


def show_frame(window_name, frame, args, wait_ms=1):
    if not args.display:
        return None

    cv2.imshow(window_name, display_frame(frame, args))
    return cv2.waitKey(wait_ms) & 0xFF


def handle_display_key(window_name, key, args):
    if key is None:
        return False
    if key == ord("q"):
        return True
    if key in (ord("+"), ord("=")):
        args.window_width = int(args.window_width * 1.15)
        args.window_height = int(args.window_height * 1.15)
        cv2.resizeWindow(window_name, args.window_width, args.window_height)
    elif key in (ord("-"), ord("_")):
        args.window_width = max(320, int(args.window_width * 0.85))
        args.window_height = max(240, int(args.window_height * 0.85))
        cv2.resizeWindow(window_name, args.window_width, args.window_height)
    elif key == ord("f"):
        args.fullscreen = not args.fullscreen
        mode = cv2.WINDOW_FULLSCREEN if args.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)
        if not args.fullscreen:
            cv2.resizeWindow(window_name, args.window_width, args.window_height)
    return False


def configure_camera(cap, args):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)


def print_camera_settings(cap, args):
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("=== CAMERA SETTINGS ===")
    print(f"Requested camera size: {args.camera_width}x{args.camera_height}")
    print(f"Actual camera size: {actual_width}x{actual_height}")
    if args.fullscreen:
        print("Display: fullscreen")
    else:
        print(f"Display window size: {args.window_width}x{args.window_height}")


def run_countdown(cap, args, seconds):
    if seconds <= 0:
        return

    for remaining in range(seconds, 0, -1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Unable to read from camera during calibration countdown.")

        draw_text(frame, "Please face the camera directly for calibration.", 50, 100)
        draw_text(frame, f"Calibration starts in: {remaining}", 50, 150, (0, 255, 255), 1)
        key = show_frame("Proctoring System", frame, args, 1000)
        if not args.display:
            time.sleep(1)
        if handle_display_key("Proctoring System", key, args):
            raise KeyboardInterrupt


def calibrate_pose(cap, face_mesh, args, seconds):
    calibration_frames = []
    start_time = time.time()

    while time.time() - start_time < seconds:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Unable to read from camera during calibration.")

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            angles = estimate_head_pose(landmarks, width, height)
            if angles:
                pitch, yaw, _ = angles
                calibration_frames.append((pitch, yaw))

        seconds_left = max(0, seconds - int(time.time() - start_time))
        draw_text(frame, "Calibrating... Keep your face straight!", 50, 100)
        draw_text(frame, f"Seconds left: {seconds_left}", 50, 150, (0, 255, 255), 1)
        key = show_frame("Proctoring System", frame, args)
        if handle_display_key("Proctoring System", key, args):
            raise KeyboardInterrupt

    if not calibration_frames:
        print("Warning: no face was detected during calibration; using neutral pose values.")
        return 0.0, 0.0

    return (
        float(np.mean([pitch for pitch, _ in calibration_frames])),
        float(np.mean([yaw for _, yaw in calibration_frames])),
    )


class BehaviorAnalyzer:
    def __init__(self, args):
        self.args = args
        self.previous_gray = None
        self.head_history = deque(maxlen=45)
        self.blink_times = deque(maxlen=80)
        self.eye_closed = False
        self.last_hand_center = None

    def analyze_frame_quality(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        events = []

        if mean_brightness < self.args.low_light_threshold:
            events.append("low_light")
        if mean_brightness > self.args.bright_light_threshold:
            events.append("brightness_anomaly")
        if contrast < self.args.camera_block_std_threshold:
            events.append("camera_block")

        motion_score = 0.0
        if self.previous_gray is not None:
            diff = cv2.absdiff(gray, self.previous_gray)
            motion_score = float(np.mean(diff))
            if motion_score > self.args.motion_threshold:
                events.append("suspicious_movement")
        self.previous_gray = gray

        return events, mean_brightness, contrast, motion_score

    def analyze_face_landmarks(self, face_mesh_results, width, height, now):
        if not face_mesh_results.multi_face_landmarks:
            return [], None, None

        landmarks = face_mesh_results.multi_face_landmarks[0].landmark
        events = []
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [263, 387, 385, 362, 380, 373]
        left_ear = eye_aspect_ratio(landmarks, left_eye_indices, width, height)
        right_ear = eye_aspect_ratio(landmarks, right_eye_indices, width, height)
        average_ear = (left_ear + right_ear) / 2

        if average_ear < 0.19 and not self.eye_closed:
            self.eye_closed = True
            self.blink_times.append(now)
        elif average_ear >= 0.22:
            self.eye_closed = False

        while self.blink_times and now - self.blink_times[0] > 60:
            self.blink_times.popleft()
        if len(self.blink_times) > self.args.blink_rate_threshold:
            events.append("abnormal_blink_rate")

        mouth_ratio = mouth_open_ratio(landmarks, width, height)
        if mouth_ratio > 0.18:
            events.append("mouth_open")

        if len(landmarks) > 473:
            left_gaze = iris_horizontal_ratio(landmarks, 468, 33, 133, width, height)
            right_gaze = iris_horizontal_ratio(landmarks, 473, 362, 263, width, height)
            gaze = (left_gaze + right_gaze) / 2
            if gaze < 0.25 or gaze > 0.75:
                events.append("eye_gaze_away")

        face_bbox = face_bbox_from_landmarks(landmarks, width, height)
        return events, face_bbox, landmarks

    def analyze_head_motion(self, pitch, yaw, now):
        self.head_history.append((now, pitch, yaw))
        recent = [item for item in self.head_history if now - item[0] <= 4]
        if len(recent) < 6:
            return []

        pitch_values = [item[1] for item in recent]
        yaw_values = [item[2] for item in recent]
        if max(pitch_values) - min(pitch_values) > 18 or max(yaw_values) - min(yaw_values) > 22:
            return ["frequent_head_movement"]
        return []

    def analyze_hands(self, frame, hand_results, face_bbox):
        events = []
        height, width, _ = frame.shape
        hand_centers = []

        for hand_landmarks in hand_results.multi_hand_landmarks or []:
            points = [
                np.array([landmark.x * width, landmark.y * height], dtype=np.float64)
                for landmark in hand_landmarks.landmark
            ]
            center = np.mean(points, axis=0)
            hand_centers.append(center)
            x1, y1 = np.min(points, axis=0).astype(int)
            x2, y2 = np.max(points, axis=0).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 255, 0), 2)
            draw_text(frame, "hand", x1, max(20, y1 - 10), (180, 255, 0), 0.6)

            if face_bbox and any(point_in_bbox(point, face_bbox, padding=45) for point in points):
                events.append("hand_near_face")
                events.append("face_cover")

        if hand_centers:
            current_center = np.mean(hand_centers, axis=0)
            if self.last_hand_center is not None:
                movement = point_distance(current_center, self.last_hand_center)
                if movement > 45:
                    events.append("hand_movement")
            self.last_hand_center = current_center
        else:
            self.last_hand_center = None

        return events


def process_head_pose(
    frame,
    face_mesh_results,
    calibrated_pitch,
    calibrated_yaw,
    pitch_threshold,
    yaw_threshold,
    direction_pitch_threshold,
    direction_yaw_threshold,
    analyzer,
    now,
):
    if not face_mesh_results.multi_face_landmarks:
        return [], None

    height, width, _ = frame.shape
    landmarks = face_mesh_results.multi_face_landmarks[0].landmark
    angles = estimate_head_pose(landmarks, width, height)
    if not angles:
        return [], None

    pitch, yaw, _ = angles
    events = []
    if is_looking_away(
        pitch,
        yaw,
        calibrated_pitch,
        calibrated_yaw,
        pitch_threshold,
        yaw_threshold,
    ):
        events.append("looking_away")

    pitch_offset = pitch - calibrated_pitch
    yaw_offset = yaw - calibrated_yaw
    if yaw_offset > direction_yaw_threshold:
        events.append("looking_right")
        draw_text(frame, "LOOKING RIGHT!", 50, 50, (0, 0, 255), 1)
    elif yaw_offset < -direction_yaw_threshold:
        events.append("looking_left")
        draw_text(frame, "LOOKING LEFT!", 50, 50, (0, 0, 255), 1)

    if pitch_offset > direction_pitch_threshold:
        events.append("looking_down")
        draw_text(frame, "LOOKING DOWN!", 50, 80, (0, 0, 255), 1)
    elif pitch_offset < -direction_pitch_threshold:
        events.append("looking_up")
        draw_text(frame, "LOOKING UP!", 50, 80, (0, 0, 255), 1)

    events.extend(analyzer.analyze_head_motion(pitch, yaw, now))

    draw_text(frame, f"Pitch: {pitch:.1f}", max(10, width - 200), 30)
    draw_text(frame, f"Yaw: {yaw:.1f}", max(10, width - 200), 60)
    return events, (pitch, yaw)


def process_objects(frame, yolo_model, confidence, suspicious_objects, show_all_objects, debug, imgsz):
    detections = {}
    person_count = 0

    for result in yolo_model(frame, stream=True, verbose=False, conf=confidence, imgsz=imgsz):
        for box in result.boxes:
            raw_label = result.names[int(box.cls[0])]
            label = normalize_label(raw_label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])

            if debug:
                print(f"YOLO: {raw_label} -> {label} {score:.2f}")

            if label == "person":
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
                draw_text(frame, f"person {score:.2f}", x1, max(20, y1 - 10), (255, 180, 0), 0.6)
                continue

            if label not in suspicious_objects:
                if show_all_objects:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
                    draw_text(
                        frame,
                        f"{raw_label} {score:.2f}",
                        x1,
                        max(20, y1 - 10),
                        (180, 180, 180),
                        0.5,
                        1,
                    )
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            draw_text(
                frame,
                f"{label} {score:.2f}",
                x1,
                max(20, y1 - 10),
                (0, 0, 255),
                0.6,
            )
            detections[label] = detections.get(label, 0) + 1

    return detections, person_count


class EventLogger:
    def __init__(self, output_dir, snapshot_events, cooldown_seconds):
        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = output_dir / session_id
        self.snapshot_dir = self.session_dir / "snapshots"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if snapshot_events:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_events = snapshot_events
        self.cooldown_seconds = cooldown_seconds
        self.last_logged_at = {}
        self.csv_file = (self.session_dir / "events.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "timestamp",
                "elapsed_seconds",
                "event",
                "details",
                "face_count",
                "person_count",
                "suspicion_score",
                "risk_level",
                "snapshot",
            ],
        )
        self.writer.writeheader()

    def close(self):
        self.csv_file.close()

    def log(self, event, details, frame, elapsed_seconds, face_count, person_count, suspicion_score):
        now = time.time()
        if now - self.last_logged_at.get(event, 0) < self.cooldown_seconds:
            return

        self.last_logged_at[event] = now
        snapshot_path = ""
        if self.snapshot_events:
            filename = f"{int(elapsed_seconds):06d}_{event}.jpg"
            snapshot_path = str(self.snapshot_dir / filename)
            cv2.imwrite(snapshot_path, frame)

        self.writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": f"{elapsed_seconds:.2f}",
                "event": event,
                "details": details,
                "face_count": face_count,
                "person_count": person_count,
                "suspicion_score": suspicion_score,
                "risk_level": risk_level(suspicion_score),
                "snapshot": snapshot_path,
            }
        )
        self.csv_file.flush()


def calculate_suspicion_score(active_events):
    return sum(EVENT_WEIGHTS.get(event, 1) for event in active_events)


def create_video_writer(args, session_dir, cap, first_frame):
    if not args.record_video:
        return None

    height, width, _ = first_frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 20.0
    output_path = session_dir / "recording.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create video recording at {output_path}")
    return writer


def print_detector_settings(args, yolo_model):
    available_labels = ", ".join(yolo_model.names.values())
    suspicious_labels = ", ".join(sorted(args.suspicious_objects))
    print("=== DETECTOR SETTINGS ===")
    print(f"Model: {args.model}")
    print(f"YOLO confidence: {args.confidence}")
    print(f"YOLO image size: {args.yolo_imgsz}")
    print(f"Suspicious labels: {suspicious_labels}")
    print(f"Available YOLO labels: {available_labels}")


def print_summary(stats, duration, session_dir):
    total_frames = max(stats["total_frames"], 1)

    print("\n=== SESSION SUMMARY ===")
    print(f"Total Duration: {duration} seconds")
    print(f"Frames Processed: {stats['total_frames']}")
    print(f"Highest Suspicion Score: {stats['highest_suspicion_score']}")
    print(f"Highest Visible People Count: {stats['highest_visible_people']}")
    for event, count in sorted(stats["event_counts"].items()):
        print(f"{event}: {count / total_frames * 100:.2f}%")
    print(f"Session Log: {session_dir / 'events.csv'}")


def run_monitoring(args):
    load_runtime_dependencies()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Run 'python scripts/download_model.py' first."
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}.")
    configure_camera(cap, args)
    configure_display("Proctoring System", args)
    print_camera_settings(cap, args)

    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
    mp_hands = mp.solutions.hands if args.advanced_analysis else None
    mp_pose = mp.solutions.pose if args.advanced_analysis else None
    yolo_model = YOLO(str(args.model))
    print_detector_settings(args, yolo_model)
    event_logger = EventLogger(args.output_dir, args.snapshot_events, args.event_cooldown)
    analyzer = BehaviorAnalyzer(args)
    stats = {
        "total_frames": 0,
        "event_counts": {},
        "highest_suspicion_score": 0,
        "highest_visible_people": 0,
    }
    video_writer = None
    hands = None
    pose = None

    start_session_time = time.time()
    try:
        with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh, (
            mp_face_detection.FaceDetection(min_detection_confidence=args.face_confidence)
        ) as face_detection:
            if args.advanced_analysis:
                hands = mp_hands.Hands(
                    max_num_hands=2,
                    min_detection_confidence=args.hand_confidence,
                    min_tracking_confidence=args.hand_confidence,
                )
                pose = mp_pose.Pose(
                    min_detection_confidence=args.pose_confidence,
                    min_tracking_confidence=args.pose_confidence,
                )

            run_countdown(cap, args, args.pre_calibration_seconds)
            calibrated_pitch, calibrated_yaw = calibrate_pose(
                cap, face_mesh, args, args.calibration_seconds
            )

            start_session_time = time.time()
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                stats["total_frames"] += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_mesh_results = face_mesh.process(rgb_frame)
                face_detection_results = face_detection.process(rgb_frame)
                hand_results = hands.process(rgb_frame) if hands else None
                pose_results = pose.process(rgb_frame) if pose else None
                elapsed_seconds = time.time() - start_session_time
                active_events = []

                brightness = 0.0
                contrast = 0.0
                motion_score = 0.0
                if args.advanced_analysis:
                    quality_events, brightness, contrast, motion_score = analyzer.analyze_frame_quality(frame)
                    active_events.extend(quality_events)

                head_events, _ = process_head_pose(
                    frame,
                    face_mesh_results,
                    calibrated_pitch,
                    calibrated_yaw,
                    args.pitch_threshold,
                    args.yaw_threshold,
                    args.direction_pitch_threshold,
                    args.direction_yaw_threshold,
                    analyzer,
                    elapsed_seconds,
                )
                active_events.extend(head_events)

                height, width, _ = frame.shape
                face_bbox = None
                if args.advanced_analysis:
                    landmark_events, face_bbox, _ = analyzer.analyze_face_landmarks(
                        face_mesh_results, width, height, elapsed_seconds
                    )
                    active_events.extend(landmark_events)

                if args.advanced_analysis and hand_results:
                    hand_events = analyzer.analyze_hands(frame, hand_results, face_bbox)
                    active_events.extend(hand_events)

                face_count = get_face_count(face_detection_results)
                draw_face_boxes(frame, face_detection_results)
                if face_count == 0:
                    active_events.append("no_face")
                    draw_text(frame, "NO FACE DETECTED!", 50, 100, (0, 0, 255), 1)

                object_detections, person_count = process_objects(
                    frame,
                    yolo_model,
                    args.confidence,
                    args.suspicious_objects,
                    args.show_all_objects,
                    args.debug_detections,
                    args.yolo_imgsz,
                )
                visible_people = max(face_count, person_count)
                stats["highest_visible_people"] = max(stats["highest_visible_people"], visible_people)
                if visible_people > 1:
                    active_events.append("multiple_people")
                    draw_text(frame, "MULTIPLE PEOPLE DETECTED!", 50, 130, (0, 0, 255), 1)

                if face_count == 0 and person_count == 0:
                    active_events.append("seat_leaving")
                if args.advanced_analysis and pose_results and pose_results.pose_landmarks is None and face_count == 0:
                    active_events.append("full_body_absence")
                if args.advanced_analysis and motion_score > args.motion_threshold and visible_people > 1:
                    active_events.append("background_movement")

                for label, count in object_detections.items():
                    stat_key = label.replace(" ", "_")
                    active_events.append(stat_key)
                    y = 150 if label == "cell phone" else 200
                    draw_text(frame, f"{label.upper()} DETECTED!", 50, y, (0, 0, 255), 1)

                active_events = sorted(set(active_events))
                suspicion_score = calculate_suspicion_score(active_events)
                stats["highest_suspicion_score"] = max(
                    stats["highest_suspicion_score"], suspicion_score
                )
                for event in active_events:
                    stats["event_counts"][event] = stats["event_counts"].get(event, 0) + 1

                if args.advanced_analysis:
                    draw_text(
                        frame,
                        f"Brightness: {brightness:.0f} | Motion: {motion_score:.1f}",
                        10,
                        max(20, frame.shape[0] - 20),
                        (220, 220, 220),
                        0.55,
                        1,
                    )
                draw_status_panel(
                    frame,
                    stats,
                    face_count,
                    person_count,
                    active_events,
                    suspicion_score,
                )

                for event in active_events:
                    event_logger.log(
                        event=event,
                        details="; ".join(
                            [
                                f"faces={face_count}",
                                f"yolo_people={person_count}",
                                f"objects={object_detections}",
                                f"brightness={brightness:.1f}",
                                f"contrast={contrast:.1f}",
                                f"motion={motion_score:.1f}",
                            ]
                        ),
                        frame=frame,
                        elapsed_seconds=elapsed_seconds,
                        face_count=face_count,
                        person_count=person_count,
                        suspicion_score=suspicion_score,
                    )

                if video_writer is None and args.record_video:
                    video_writer = create_video_writer(args, event_logger.session_dir, cap, frame)
                if video_writer is not None:
                    video_writer.write(frame)

                key = show_frame("Proctoring System", frame, args)
                if handle_display_key("Proctoring System", key, args):
                    break
                if args.max_seconds and elapsed_seconds >= args.max_seconds:
                    break
    finally:
        if args.advanced_analysis:
            if hands is not None:
                hands.close()
            if pose is not None:
                pose.close()
        cap.release()
        if video_writer is not None:
            video_writer.release()
        event_logger.close()
        if args.display:
            cv2.destroyAllWindows()

    print_summary(stats, int(time.time() - start_session_time), event_logger.session_dir)


def main():
    args = parse_args()
    try:
        run_monitoring(args)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nSession stopped by user.")


if __name__ == "__main__":
    main()