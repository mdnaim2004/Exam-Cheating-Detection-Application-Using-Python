"""Optimized YOLO object detection with caching."""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DetectionResult:
    """A single YOLO detection result."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class YOLOFrameCache:
    """Cache for YOLO results to avoid redundant computation."""
    frame_id: int
    results: List[DetectionResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def is_stale(self, current_frame_id: int, max_age: int = 3) -> bool:
        """Check if cache is too old."""
        return current_frame_id - self.frame_id > max_age


class YOLODetector:
    """Wrapper for YOLO detection with optimization."""

    def __init__(self, model, confidence_threshold: float = 0.45, cache_enabled: bool = True):
        """Initialize YOLO detector.

        Args:
            model: Ultralytics YOLO model
            confidence_threshold: Minimum confidence for detections
            cache_enabled: Enable result caching
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.cache_enabled = cache_enabled
        self.cache: Optional[YOLOFrameCache] = None
        self.frame_counter = 0

    def detect(self, frame: np.ndarray, imgsz: int = 640, use_cache: bool = True) -> List[DetectionResult]:
        """Run detection on frame.

        Args:
            frame: Input image frame
            imgsz: YOLO inference size
            use_cache: Use cached results if available

        Returns:
            List of DetectionResult objects
        """
        self.frame_counter += 1

        # Check cache
        if use_cache and self.cache_enabled and self.cache and not self.cache.is_stale(self.frame_counter):
            return self.cache.results

        # Run detection
        results = self.model.predict(
            frame, conf=self.confidence_threshold, imgsz=imgsz, verbose=False)

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                if box.conf[0].item() >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = self.model.names.get(int(box.cls[0]), "unknown")
                    confidence = box.conf[0].item()

                    detections.append(DetectionResult(
                        label=label,
                        confidence=confidence,
                        bbox=(int(x1), int(y1), int(x2), int(y2))
                    ))

        # Cache result
        if self.cache_enabled:
            self.cache = YOLOFrameCache(
                frame_id=self.frame_counter, results=detections)

        return detections

    def get_detections_by_label(self, detections: List[DetectionResult], labels: set) -> List[DetectionResult]:
        """Filter detections by label set."""
        return [d for d in detections if d.label.lower() in labels]

    def count_label(self, detections: List[DetectionResult], label: str) -> int:
        """Count occurrences of a label."""
        return sum(1 for d in detections if d.label.lower() == label.lower())

    def clear_cache(self):
        """Clear detection cache."""
        self.cache = None
