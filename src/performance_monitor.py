"""Performance monitoring and metrics tracking."""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PerformanceMetrics:
    """Track performance metrics for different components."""

    component_name: str
    timings: deque = field(default_factory=lambda: deque(maxlen=100))
    last_timestamp: float = field(default_factory=time.time)

    def record(self, duration: float):
        """Record a timing."""
        self.timings.append(duration)
        self.last_timestamp = time.time()

    def average_ms(self) -> float:
        """Get average timing in milliseconds."""
        if not self.timings:
            return 0.0
        return sum(self.timings) / len(self.timings) * 1000

    def max_ms(self) -> float:
        """Get max timing in milliseconds."""
        if not self.timings:
            return 0.0
        return max(self.timings) * 1000


class PerformanceMonitor:
    """Monitor overall system performance."""

    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_window = deque(maxlen=30)
        self.last_fps_update = self.start_time

    def record_component(self, component: str, duration: float):
        """Record timing for a component."""
        if component not in self.metrics:
            self.metrics[component] = PerformanceMetrics(component)
        self.metrics[component].record(duration)

    def record_frame(self):
        """Mark a frame processed."""
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_fps_update

        if elapsed > 0:
            self.fps_window.append(1.0 / elapsed)
            self.last_fps_update = now

    def get_fps(self) -> float:
        """Get current FPS."""
        if not self.fps_window:
            return 0.0
        return sum(self.fps_window) / len(self.fps_window)

    def get_runtime(self) -> float:
        """Get total runtime in seconds."""
        return time.time() - self.start_time

    def get_summary(self) -> Dict:
        """Get performance summary."""
        return {
            "total_frames": self.frame_count,
            "runtime_seconds": self.get_runtime(),
            "average_fps": self.get_fps(),
            "components": {
                name: {
                    "average_ms": metric.average_ms(),
                    "max_ms": metric.max_ms(),
                    "samples": len(metric.timings),
                }
                for name, metric in self.metrics.items()
            },
        }

    def print_summary(self):
        """Print performance summary."""
        summary = self.get_summary()
        print("\n=== PERFORMANCE SUMMARY ===")
        print(f"Total frames: {summary['total_frames']}")
        print(f"Runtime: {summary['runtime_seconds']:.1f}s")
        print(f"Average FPS: {summary['average_fps']:.1f}")
        print("\nComponent timings (ms):")
        for name, stats in summary['components'].items():
            print(
                f"  {name}: {stats['average_ms']:.2f}ms (max: {stats['max_ms']:.2f}ms)")


class FrameTimer:
    """Context manager for timing frame processing."""

    def __init__(self, monitor: PerformanceMonitor, component: str):
        self.monitor = monitor
        self.component = component
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.monitor.record_component(self.component, duration)
