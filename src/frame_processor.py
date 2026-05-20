"""Optimized frame processing pipeline with frame skipping and caching."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from src.config import ProcessConfig, FrameProcessingConfig
from src.performance_monitor import PerformanceMonitor, FrameTimer


class FrameProcessor:
    """Handles frame processing with intelligent frame skipping."""

    def __init__(self, config: ProcessConfig, monitor: PerformanceMonitor):
        """Initialize frame processor.

        Args:
            config: Process configuration
            monitor: Performance monitor
        """
        self.config = config
        self.monitor = monitor
        self.frame_id = 0

        # Frame skipping tracking
        self.yolo_skip_counter = 0
        self.mediapipe_heavy_skip_counter = 0
        self.pose_skip_counter = 0

        # Cached results
        self.last_yolo_results = None
        self.last_landmarks = None
        self.last_pose = None

    def should_run_yolo(self) -> bool:
        """Check if YOLO should run this frame."""
        self.yolo_skip_counter += 1
        if self.yolo_skip_counter >= self.config.frame_processing.yolo_frame_skip:
            self.yolo_skip_counter = 0
            return True
        return False

    def should_run_mediapipe_heavy(self) -> bool:
        """Check if heavy MediaPipe analysis should run."""
        self.mediapipe_heavy_skip_counter += 1
        if self.mediapipe_heavy_skip_counter >= self.config.frame_processing.mediapipe_heavy_skip:
            self.mediapipe_heavy_skip_counter = 0
            return True
        return False

    def should_run_pose(self) -> bool:
        """Check if pose estimation should run."""
        self.pose_skip_counter += 1
        if self.pose_skip_counter >= self.config.frame_processing.pose_skip:
            self.pose_skip_counter = 0
            return True
        return False

    def resize_for_processing(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Resize frame for efficient processing.

        Args:
            frame: Original frame

        Returns:
            (resized_frame, scale_x, scale_y) where scale factors are for upscaling
        """
        import cv2

        height, width = frame.shape[:2]
        target_width = self.config.frame_processing.process_width
        target_height = self.config.frame_processing.process_height

        # Calculate aspect ratio
        aspect = width / height
        target_aspect = target_width / target_height

        if aspect > target_aspect:
            # Width limited
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            # Height limited
            new_height = target_height
            new_width = int(target_height * aspect)

        resized = cv2.resize(frame, (new_width, new_height),
                             interpolation=cv2.INTER_LINEAR)

        scale_x = width / new_width
        scale_y = height / new_height

        return resized, scale_x, scale_y

    def scale_detections(self, detections: List, scale_x: float, scale_y: float) -> List:
        """Scale detection coordinates back to original frame size.

        Args:
            detections: Detection results
            scale_x, scale_y: Scale factors

        Returns:
            Scaled detections
        """
        scaled = []
        for det in detections:
            # Scale bbox coordinates
            x1, y1, x2, y2 = det.bbox
            det.bbox = (
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y)
            )
            scaled.append(det)
        return scaled

    def process_frame_step(self, frame: np.ndarray, step_name: str, process_fn, *args, **kwargs):
        """Process a frame with timing and monitoring.

        Args:
            frame: Input frame
            step_name: Name of processing step
            process_fn: Function to call
            *args, **kwargs: Arguments for process_fn

        Returns:
            Result from process_fn
        """
        with FrameTimer(self.monitor, step_name):
            return process_fn(frame, *args, **kwargs)

    def next_frame(self):
        """Mark next frame in sequence."""
        self.frame_id += 1
        self.monitor.record_frame()
