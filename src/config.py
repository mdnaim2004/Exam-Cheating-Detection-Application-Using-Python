"""Configuration management for the proctoring system."""

from dataclasses import dataclass, field
from typing import Set
from pathlib import Path


@dataclass
class DetectionConfig:
    """Configuration for detection thresholds and limits."""

    yolo_confidence: float = 0.45
    yolo_imgsz: int = 640
    face_confidence: float = 0.35
    hand_confidence: float = 0.4
    pose_confidence: float = 0.4

    # Head pose thresholds
    yaw_threshold: float = 140.0
    pitch_threshold: float = 180.0

    # Advanced analysis
    enable_advanced_analysis: bool = False
    blink_rate_threshold: int = 30

    # Camera & environment
    low_light_threshold: float = 45.0
    bright_light_threshold: float = 235.0
    camera_block_std_threshold: float = 12.0
    motion_threshold: float = 18.0

    # Event handling
    event_cooldown: float = 2.0

    # Suspicious objects
    suspicious_objects: Set[str] = field(default_factory=lambda: {
        "cell phone", "book", "laptop", "keyboard", "mouse", "remote"
    })


@dataclass
class FrameProcessingConfig:
    """Configuration for frame processing optimization."""

    # Frame skipping strategy
    yolo_frame_skip: int = 2  # Run YOLO every N frames
    mediapipe_heavy_skip: int = 3  # Run heavy MediaPipe analysis every N frames
    pose_skip: int = 2  # Run pose estimation every N frames

    # Processing resolution (for efficiency)
    process_width: int = 640  # Process at this width
    process_height: int = 480  # Process at this height
    camera_width: int = 1280
    camera_height: int = 720

    # Caching
    cache_yolo_results: bool = True
    cache_landmarks: bool = True

    # Threading
    use_threaded_camera: bool = True
    camera_thread_queue_size: int = 2


@dataclass
class DisplayConfig:
    """Configuration for display and output."""

    display: bool = True
    window_width: int = 1280
    window_height: int = 720
    fullscreen: bool = False

    # Output
    output_dir: Path = field(default_factory=lambda: Path("sessions"))
    snapshot_events: bool = True
    record_video: bool = False
    show_all_objects: bool = False
    debug_detections: bool = False


@dataclass
class ProcessConfig:
    """Master configuration combining all sub-configs."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    frame_processing: FrameProcessingConfig = field(
        default_factory=FrameProcessingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    model_path: str = "models/yolov8n.pt"
    max_seconds: int = 0  # 0 = no limit

    @classmethod
    def from_args(cls, args) -> "ProcessConfig":
        """Create configuration from argparse args."""
        config = cls()

        # Detection config
        config.detection.yolo_confidence = args.confidence
        config.detection.yolo_imgsz = args.yolo_imgsz
        config.detection.face_confidence = args.face_confidence
        config.detection.hand_confidence = args.hand_confidence
        config.detection.pose_confidence = args.pose_confidence
        config.detection.yaw_threshold = args.direction_yaw_threshold
        config.detection.pitch_threshold = args.direction_pitch_threshold
        config.detection.enable_advanced_analysis = args.advanced_analysis
        config.detection.event_cooldown = args.event_cooldown
        config.detection.suspicious_objects = args.suspicious_objects
        config.detection.blink_rate_threshold = args.blink_rate_threshold
        config.detection.low_light_threshold = args.low_light_threshold
        config.detection.bright_light_threshold = args.bright_light_threshold
        config.detection.camera_block_std_threshold = args.camera_block_std_threshold
        config.detection.motion_threshold = args.motion_threshold

        # Frame processing config
        config.frame_processing.camera_width = args.camera_width
        config.frame_processing.camera_height = args.camera_height
        config.frame_processing.process_width = min(args.camera_width, 640)
        config.frame_processing.process_height = min(args.camera_height, 480)

        # Display config
        config.display.display = args.display
        config.display.window_width = args.window_width
        config.display.window_height = args.window_height
        config.display.fullscreen = args.fullscreen
        config.display.output_dir = args.output_dir
        config.display.snapshot_events = args.snapshot_events
        config.display.record_video = args.record_video
        config.display.show_all_objects = args.show_all_objects
        config.display.debug_detections = args.debug_detections

        # Other
        config.model_path = args.model
        config.max_seconds = args.max_seconds

        return config
