"""Head pose and facial analysis."""

import numpy as np
from typing import Optional, Tuple


class PoseAnalyzer:
    """Analyze head pose from facial landmarks."""

    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ], dtype=np.float64)

    def __init__(self):
        """Initialize pose analyzer."""
        import cv2
        self.cv2 = cv2

    def estimate_head_pose(self, landmarks, width: int, height: int) -> Optional[Tuple[float, float, float]]:
        """Estimate head pose angles from facial landmarks.

        Args:
            landmarks: MediaPipe face landmarks
            width: Frame width
            height: Frame height

        Returns:
            (pitch, yaw, roll) angles or None if estimation fails
        """
        image_points = np.array([
            (landmarks[1].x * width, landmarks[1].y * height),
            (landmarks[152].x * width, landmarks[152].y * height),
            (landmarks[33].x * width, landmarks[33].y * height),
            (landmarks[263].x * width, landmarks[263].y * height),
            (landmarks[61].x * width, landmarks[61].y * height),
            (landmarks[291].x * width, landmarks[291].y * height),
        ], dtype=np.float64)

        focal_length = width
        camera_matrix = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        success, rotation_vector, _ = self.cv2.solvePnP(
            self.MODEL_POINTS, image_points, camera_matrix, np.zeros((4, 1))
        )

        if not success:
            return None

        rotation_matrix, _ = self.cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = self.cv2.RQDecomp3x3(rotation_matrix)

        return tuple(angles)

    def is_looking_away(self, pitch: float, yaw: float,
                        calibrated_pitch: float, calibrated_yaw: float,
                        pitch_threshold: float, yaw_threshold: float) -> bool:
        """Check if head is looking away from camera.

        Args:
            pitch, yaw: Current head angles
            calibrated_pitch, calibrated_yaw: Baseline angles
            pitch_threshold, yaw_threshold: Detection thresholds

        Returns:
            True if head is looking away
        """
        pitch_offset = abs(pitch - calibrated_pitch)
        yaw_offset = abs(yaw - calibrated_yaw)
        return pitch_offset > pitch_threshold or yaw_offset > yaw_threshold


class BehaviorAnalyzer:
    """Analyze behavioral metrics from facial landmarks."""

    def __init__(self):
        """Initialize behavior analyzer."""
        self.blink_counter = 0
        self.prev_eye_state = None

    @staticmethod
    def landmark_point(landmarks, index: int, width: int, height: int) -> np.ndarray:
        """Get 2D point from landmark."""
        landmark = landmarks[index]
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float64)

    @staticmethod
    def point_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
        """Calculate Euclidean distance between points."""
        return float(np.linalg.norm(point_a - point_b))

    @staticmethod
    def eye_aspect_ratio(landmarks, indices, width: int, height: int) -> float:
        """Calculate eye aspect ratio (used for blink detection).

        Args:
            landmarks: Face landmarks
            indices: Eye landmark indices
            width, height: Frame dimensions

        Returns:
            Eye aspect ratio (lower = blink)
        """
        points = [BehaviorAnalyzer.landmark_point(
            landmarks, idx, width, height) for idx in indices]
        vertical_1 = BehaviorAnalyzer.point_distance(points[1], points[5])
        vertical_2 = BehaviorAnalyzer.point_distance(points[2], points[4])
        horizontal = max(BehaviorAnalyzer.point_distance(
            points[0], points[3]), 1.0)
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    @staticmethod
    def mouth_open_ratio(landmarks, width: int, height: int) -> float:
        """Calculate mouth opening ratio.

        Args:
            landmarks: Face landmarks
            width, height: Frame dimensions

        Returns:
            Mouth opening ratio
        """
        upper_lip = BehaviorAnalyzer.landmark_point(
            landmarks, 13, width, height)
        lower_lip = BehaviorAnalyzer.landmark_point(
            landmarks, 14, width, height)
        left_corner = BehaviorAnalyzer.landmark_point(
            landmarks, 61, width, height)
        right_corner = BehaviorAnalyzer.landmark_point(
            landmarks, 291, width, height)
        return BehaviorAnalyzer.point_distance(upper_lip, lower_lip) / max(
            BehaviorAnalyzer.point_distance(left_corner, right_corner), 1.0
        )

    @staticmethod
    def iris_horizontal_ratio(landmarks, iris_index: int,
                              left_corner_index: int, right_corner_index: int,
                              width: int, height: int) -> float:
        """Calculate horizontal iris position ratio.

        Args:
            landmarks: Face landmarks
            iris_index: Iris landmark index
            left_corner_index, right_corner_index: Eye corner indices
            width, height: Frame dimensions

        Returns:
            Position ratio (0.0 = far left, 0.5 = center, 1.0 = far right)
        """
        iris = BehaviorAnalyzer.landmark_point(
            landmarks, iris_index, width, height)
        left_corner = BehaviorAnalyzer.landmark_point(
            landmarks, left_corner_index, width, height)
        right_corner = BehaviorAnalyzer.landmark_point(
            landmarks, right_corner_index, width, height)
        return (iris[0] - left_corner[0]) / max(right_corner[0] - left_corner[0], 1.0)
