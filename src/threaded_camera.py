"""Threaded camera capture for non-blocking frame reads."""

import threading
import queue
import time
import numpy as np
from typing import Optional, Tuple


class ThreadedCamera:
    """Non-blocking camera capture using background thread."""

    def __init__(self, cap, queue_size: int = 2):
        """Initialize threaded camera.

        Args:
            cap: OpenCV VideoCapture object
            queue_size: Maximum frames to buffer
        """
        self.cap = cap
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.stop_event = threading.Event()
        self.is_open = True
        self.thread.start()

    def _capture_loop(self):
        """Background thread that continuously captures frames."""
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.is_open = False
                break

            # Replace oldest frame if queue is full
            try:
                self.queue.put((ok, frame), block=False)
            except queue.Full:
                try:
                    self.queue.get_nowait()  # Remove oldest
                    self.queue.put((ok, frame), block=False)
                except queue.Empty:
                    pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the most recent frame without blocking.

        Returns:
            (success, frame) tuple, or (False, None) if no frame available
        """
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return False, None

    def get_property(self, prop: int):
        """Get camera property."""
        return self.cap.get(prop)

    def set_property(self, prop: int, value):
        """Set camera property."""
        self.cap.set(prop, value)

    def release(self):
        """Stop capture and release resources."""
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.cap.release()

    def is_opened(self) -> bool:
        """Check if camera is still open."""
        return self.is_open and not self.stop_event.is_set()
