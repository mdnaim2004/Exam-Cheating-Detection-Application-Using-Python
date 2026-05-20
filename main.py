import argparse
import importlib
import sys
import time
from pathlib import Path


SUSPICIOUS_OBJECTS = {"cell phone", "book"}
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
            "'pip install -r requirements.txt'."
        )

    cv2 = modules["cv2"]
    mp = modules["mediapipe"]
    np = modules["numpy"]
    YOLO = modules["ultralytics"].YOLO
    MODEL_POINTS = np.array(MODEL_POINTS_DATA, dtype=np.float64)


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
        default=0.45,
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
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the live OpenCV monitoring window.",
    )
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1.")
    if args.pre_calibration_seconds < 0 or args.calibration_seconds < 0:
        parser.error("Calibration durations must be zero or greater.")
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


def draw_text(frame, text, x, y, color=(255, 255, 255), scale=0.7, thickness=2):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def show_frame(window_name, frame, display, wait_ms=1):
    if not display:
        return None

    cv2.imshow(window_name, frame)
    return cv2.waitKey(wait_ms) & 0xFF


def run_countdown(cap, display, seconds):
    if seconds <= 0:
        return

    for remaining in range(seconds, 0, -1):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Unable to read from camera during calibration countdown.")

        draw_text(frame, "Please face the camera directly for calibration.", 50, 100)
        draw_text(frame, f"Calibration starts in: {remaining}", 50, 150, (0, 255, 255), 1)
        key = show_frame("Proctoring System", frame, display, 1000)
        if not display:
            time.sleep(1)
        if key == ord("q"):
            raise KeyboardInterrupt


def calibrate_pose(cap, face_mesh, display, seconds):
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
        key = show_frame("Proctoring System", frame, display)
        if key == ord("q"):
            raise KeyboardInterrupt

    if not calibration_frames:
        print("Warning: no face was detected during calibration; using neutral pose values.")
        return 0.0, 0.0

    return (
        float(np.mean([pitch for pitch, _ in calibration_frames])),
        float(np.mean([yaw for _, yaw in calibration_frames])),
    )


def process_head_pose(
    frame,
    face_mesh_results,
    calibrated_pitch,
    calibrated_yaw,
    pitch_threshold,
    yaw_threshold,
):
    if not face_mesh_results.multi_face_landmarks:
        return False

    height, width, _ = frame.shape
    landmarks = face_mesh_results.multi_face_landmarks[0].landmark
    angles = estimate_head_pose(landmarks, width, height)
    if not angles:
        return False

    pitch, yaw, _ = angles
    looking_away = is_looking_away(
        pitch,
        yaw,
        calibrated_pitch,
        calibrated_yaw,
        pitch_threshold,
        yaw_threshold,
    )
    if looking_away:
        draw_text(frame, "LOOKING AWAY!", 50, 50, (0, 0, 255), 1)

    draw_text(frame, f"Pitch: {pitch:.1f}", max(10, width - 200), 30)
    draw_text(frame, f"Yaw: {yaw:.1f}", max(10, width - 200), 60)
    return looking_away


def process_objects(frame, yolo_model, confidence):
    detections = set()

    for result in yolo_model(frame, stream=True, verbose=False, conf=confidence):
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            if label not in SUSPICIOUS_OBJECTS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            draw_text(
                frame,
                f"{label} {score:.2f}",
                x1,
                max(20, y1 - 10),
                (0, 0, 255),
                0.6,
            )
            detections.add(label)

    return detections


def print_summary(stats, duration):
    total_frames = max(stats["total_frames"], 1)

    print("\n=== SESSION SUMMARY ===")
    print(f"Total Duration: {duration} seconds")
    print(f"Frames Processed: {stats['total_frames']}")
    print(f"Looking Away: {stats['looking_away'] / total_frames * 100:.2f}%")
    print(f"Phone Detection: {stats['cell_phone'] / total_frames * 100:.2f}%")
    print(f"Book Detection: {stats['book'] / total_frames * 100:.2f}%")
    print(f"Unauthorized Person Detection: {stats['multiple_people'] / total_frames * 100:.2f}%")


def run_monitoring(args):
    load_runtime_dependencies()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Run 'python scripts/download_model.py' first."
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}.")

    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
    yolo_model = YOLO(str(args.model))
    stats = {
        "total_frames": 0,
        "looking_away": 0,
        "cell_phone": 0,
        "book": 0,
        "multiple_people": 0,
    }

    try:
        with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh, (
            mp_face_detection.FaceDetection()
        ) as face_detection:
            run_countdown(cap, args.display, args.pre_calibration_seconds)
            calibrated_pitch, calibrated_yaw = calibrate_pose(
                cap, face_mesh, args.display, args.calibration_seconds
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

                if process_head_pose(
                    frame,
                    face_mesh_results,
                    calibrated_pitch,
                    calibrated_yaw,
                    args.pitch_threshold,
                    args.yaw_threshold,
                ):
                    stats["looking_away"] += 1

                face_count = len(face_detection_results.detections or [])
                if face_count > 1:
                    stats["multiple_people"] += 1
                    draw_text(frame, "MULTIPLE PEOPLE DETECTED!", 50, 100, (0, 0, 255), 1)

                object_detections = process_objects(frame, yolo_model, args.confidence)
                if "cell phone" in object_detections:
                    stats["cell_phone"] += 1
                    draw_text(frame, "PHONE DETECTED!", 50, 150, (0, 0, 255), 1)
                if "book" in object_detections:
                    stats["book"] += 1
                    draw_text(frame, "BOOK DETECTED!", 50, 200, (0, 0, 255), 1)

                key = show_frame("Proctoring System", frame, args.display)
                if key == ord("q"):
                    break
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()

    print_summary(stats, int(time.time() - start_session_time))


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
