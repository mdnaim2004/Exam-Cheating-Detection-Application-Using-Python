"""Optimized MediaPipe face detection with caching."""

import time
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Face:
    """Detected face with landmarks and pose info."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[object] = None  # MediaPipe landmarks
    pose_angles: Optional[Tuple[float, float, float]
                          ] = None  # pitch, yaw, roll


@dataclass
class FaceDetectionCache:
    """Cache for face detection results."""
    frame_id: int
    faces: List[Face] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def is_stale(self, current_frame_id: int, max_age: int = 2) -> bool:
        """Check if cache is too old."""
        return current_frame_id - self.frame_id > max_age


class FaceDetector:
    """Wrapper for MediaPipe face detection with caching."""

    def __init__(self, face_detection, face_mesh, cache_enabled: bool = True):
        """Initialize face detector.

        Args:
            face_detection: MediaPipe face detection model
            face_mesh: MediaPipe face mesh model
            cache_enabled: Enable result caching
        """
        self.face_detection = face_detection
        self.face_mesh = face_mesh
        self.cache_enabled = cache_enabled
        self.cache: Optional[FaceDetectionCache] = None
        self.frame_counter = 0

    def detect_faces(self, frame: np.ndarray, use_cache: bool = True) -> Tuple[int, Optional[FaceDetectionCache]]:
        """Detect faces in frame.

        Args:
            frame: Input image frame (BGR)
            use_cache: Use cached results if available

        Returns:
            (face_count, cache_object)
        """
        self.frame_counter += 1

        # Check cache
        if use_cache and self.cache_enabled and self.cache and not self.cache.is_stale(self.frame_counter):
            return len(self.cache.faces), self.cache

        # Convert to RGB for MediaPipe
        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        results = self.face_detection.process(rgb_frame)
        height, width, _ = frame.shape

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox_data = detection.location_data.relative_bounding_box
                x1 = max(0, int(bbox_data.xmin * width))
                y1 = max(0, int(bbox_data.ymin * height))
                x2 = min(width, int((bbox_data.xmin + bbox_data.width) * width))
                y2 = min(height, int(
                    (bbox_data.ymin + bbox_data.height) * height))

                face = Face(
                    bbox=(x1, y1, x2, y2),
                    confidence=detection.score[0] if detection.score else 0.0
                )
                faces.append(face)

        # Cache result
        if self.cache_enabled:
            self.cache = FaceDetectionCache(
                frame_id=self.frame_counter, faces=faces)

        return len(faces), self.cache

    def get_landmarks(self, frame: np.ndarray, max_faces: int = 1):
        """Get facial landmarks for faces.

        Args:
            frame: Input image frame (BGR)
            max_faces: Maximum faces to process

        Returns:
            List of landmark sets
        """
        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[:max_faces]
        return []

    def clear_cache(self):
        """Clear detection cache."""
        self.cache = None
