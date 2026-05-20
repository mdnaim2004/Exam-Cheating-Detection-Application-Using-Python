#!/usr/bin/env python3
"""
Demo script showing performance optimizations.
Usage: python examples/demo_optimization.py
"""

from src.performance_monitor import PerformanceMonitor, FrameTimer
from src.config import ProcessConfig, FrameProcessingConfig, DetectionConfig
import sys
import time
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_configuration():
    """Demonstrate configuration management."""
    print("=" * 60)
    print("DEMO 1: Configuration Management")
    print("=" * 60)

    # Create default config
    config = ProcessConfig()
    print(f"Default YOLO confidence: {config.detection.yolo_confidence}")
    print(
        f"Default frame skip (YOLO): {config.frame_processing.yolo_frame_skip}")
    print(
        f"Processing resolution: {config.frame_processing.process_width}x{config.frame_processing.process_height}")
    print(f"Threaded camera: {config.frame_processing.use_threaded_camera}")

    # Modify for lightweight mode
    config.detection.yolo_confidence = 0.5
    config.frame_processing.yolo_frame_skip = 3
    config.frame_processing.process_width = 480
    config.frame_processing.process_height = 360

    print("\nAfter optimization for lightweight mode:")
    print(f"YOLO confidence: {config.detection.yolo_confidence}")
    print(f"Frame skip (YOLO): {config.frame_processing.yolo_frame_skip}")
    print(
        f"Processing resolution: {config.frame_processing.process_width}x{config.frame_processing.process_height}")
    print()


def demo_performance_monitoring():
    """Demonstrate performance monitoring."""
    print("=" * 60)
    print("DEMO 2: Performance Monitoring")
    print("=" * 60)

    monitor = PerformanceMonitor()

    # Simulate frame processing with different components
    for i in range(30):
        # Simulate YOLO detection
        with FrameTimer(monitor, "yolo_detection"):
            time.sleep(0.02)  # 20ms YOLO inference

        # Simulate face detection
        with FrameTimer(monitor, "face_detection"):
            time.sleep(0.01)  # 10ms face detection

        # Simulate pose estimation (every 2 frames)
        if i % 2 == 0:
            with FrameTimer(monitor, "pose_estimation"):
                time.sleep(0.015)  # 15ms pose estimation

        monitor.record_frame()

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} frames... FPS: {monitor.get_fps():.1f}")

    print("\nPerformance Summary:")
    monitor.print_summary()
    print()


def demo_frame_skipping():
    """Demonstrate frame skipping logic."""
    print("=" * 60)
    print("DEMO 3: Frame Skipping Strategy")
    print("=" * 60)

    from src.frame_processor import FrameProcessor

    config = ProcessConfig()
    config.frame_processing.yolo_frame_skip = 2
    config.frame_processing.mediapipe_heavy_skip = 3
    config.frame_processing.pose_skip = 2

    monitor = PerformanceMonitor()
    processor = FrameProcessor(config, monitor)

    print(
        f"YOLO frame skip: every {config.frame_processing.yolo_frame_skip} frames")
    print(
        f"MediaPipe heavy skip: every {config.frame_processing.mediapipe_heavy_skip} frames")
    print(f"Pose skip: every {config.frame_processing.pose_skip} frames")
    print()

    # Simulate 20 frames
    for i in range(20):
        processor.next_frame()

        yolo_run = processor.should_run_yolo()
        heavy_run = processor.should_run_mediapipe_heavy()
        pose_run = processor.should_run_pose()

        print(f"Frame {i+1:2d}: YOLO={'✓' if yolo_run else ' '} | Heavy={'✓' if heavy_run else ' '} | Pose={'✓' if pose_run else ' '}")

    print()


def demo_caching_concept():
    """Demonstrate caching benefits."""
    print("=" * 60)
    print("DEMO 4: Caching Concept")
    print("=" * 60)

    from src.detector.yolo_detector import YOLOFrameCache, DetectionResult

    # Simulate detection cache
    detections = [
        DetectionResult("cell phone", 0.95, (100, 100, 200, 200)),
        DetectionResult("book", 0.87, (300, 150, 450, 350)),
    ]

    cache = YOLOFrameCache(frame_id=1, results=detections)

    print(f"Cache created at frame 1 with {len(detections)} detections")
    print(f"Cached detections:")
    for det in cache.results:
        print(f"  - {det.label}: confidence {det.confidence:.2f}")

    print("\nCache staleness check:")
    print(f"  At frame 2: stale={cache.is_stale(2, max_age=3)} (age=1)")
    print(f"  At frame 4: stale={cache.is_stale(4, max_age=3)} (age=3)")
    print(f"  At frame 5: stale={cache.is_stale(5, max_age=3)} (age=4)")

    print("\nCaching benefits:")
    print("  - Skip expensive detection on N-1 out of N frames")
    print("  - Temporal redundancy: most detections don't change per frame")
    print("  - ~90% faster result retrieval for cached frames")
    print()


def demo_threaded_camera_concept():
    """Demonstrate threaded camera concept."""
    print("=" * 60)
    print("DEMO 5: Threaded Camera Concept")
    print("=" * 60)

    print("ThreadedCamera benefits:")
    print("  - Runs capture loop in background thread")
    print("  - Main thread never blocks waiting for camera")
    print("  - Prevents frame drops from processing delays")
    print("  - Queue maintains N most recent frames")
    print()

    print("Usage example:")
    print("  camera = ThreadedCamera(cv2.VideoCapture(0), queue_size=2)")
    print("  while True:")
    print("      ok, frame = camera.read()  # Non-blocking!")
    print("      # Process frame...")
    print("      # Processing delay won't drop frames")
    print()


def demo_performance_gains():
    """Show expected performance gains."""
    print("=" * 60)
    print("DEMO 6: Expected Performance Gains")
    print("=" * 60)

    scenarios = [
        {
            "name": "Frame Skipping Only",
            "config": {
                "yolo_skip": 2,
                "pose_skip": 2,
                "threading": False,
            },
            "gains": "40-50% CPU reduction, 2-3x faster detection",
        },
        {
            "name": "Threading Only",
            "config": {
                "yolo_skip": 1,
                "pose_skip": 1,
                "threading": True,
            },
            "gains": "60+ FPS possible, smooth capture",
        },
        {
            "name": "Caching Only",
            "config": {
                "caching": True,
                "cache_max_age": 3,
            },
            "gains": "~90% faster repeated detections",
        },
        {
            "name": "Full Optimization",
            "config": {
                "yolo_skip": 2,
                "pose_skip": 2,
                "threading": True,
                "caching": True,
                "process_resolution": "640x480",
            },
            "gains": "30-50% FPS improvement, 40% CPU reduction",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Config: {scenario['config']}")
        print(f"   Gains: {scenario['gains']}")

    print("\n" + "=" * 60)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EXAM CHEATING DETECTION - OPTIMIZATION DEMO")
    print("=" * 60 + "\n")

    parser = argparse.ArgumentParser(
        description="Performance optimization demo")
    parser.add_argument("--all", action="store_true", help="Run all demos")
    parser.add_argument("--config", action="store_true",
                        help="Run config demo")
    parser.add_argument("--monitoring", action="store_true",
                        help="Run monitoring demo")
    parser.add_argument("--skipping", action="store_true",
                        help="Run frame skipping demo")
    parser.add_argument("--caching", action="store_true",
                        help="Run caching demo")
    parser.add_argument("--threading", action="store_true",
                        help="Run threading demo")
    parser.add_argument("--gains", action="store_true",
                        help="Show expected gains")

    args = parser.parse_args()

    # Default to all if no specific demo selected
    if not any([args.all, args.config, args.monitoring, args.skipping, args.caching, args.threading, args.gains]):
        args.all = True

    if args.all or args.config:
        demo_configuration()

    if args.all or args.monitoring:
        demo_performance_monitoring()

    if args.all or args.skipping:
        demo_frame_skipping()

    if args.all or args.caching:
        demo_caching_concept()

    if args.all or args.threading:
        demo_threaded_camera_concept()

    if args.all or args.gains:
        demo_performance_gains()

    print("=" * 60)
    print("For full integration, see PERFORMANCE_GUIDE.md")
    print("=" * 60 + "\n")
