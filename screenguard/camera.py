"""
ScreenGuard Camera Module

Handles webcam access, frame capture, and camera lifecycle management.
Runs in its own thread to avoid blocking the UI or detection pipeline.
"""

import cv2
import time
import logging
import threading
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """
    Thread-safe webcam wrapper with automatic reconnection.
    
    Usage:
        camera = Camera(camera_index=0)
        camera.start()
        
        frame = camera.read()
        if frame is not None:
            # process frame
            ...
        
        camera.stop()
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self._index = camera_index
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def fps(self) -> float:
        return self._fps

    def start(self) -> bool:
        """Open the camera and begin capturing frames in a background thread."""
        if self._running:
            logger.warning("Camera is already running.")
            return True

        success = self._open()
        if not success:
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraThread")
        self._thread.start()
        logger.info("Camera started (index=%d, %dx%d)", self._index, self._width, self._height)
        return True

    def stop(self):
        """Stop capturing and release the camera."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._close()
        logger.info("Camera stopped.")

    def read(self) -> Optional[np.ndarray]:
        """Return the most recent frame (thread-safe). Returns None if no frame available."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def _open(self) -> bool:
        """Open the camera device."""
        try:
            self._cap = cv2.VideoCapture(self._index)
            if not self._cap.isOpened():
                logger.error("Cannot open camera index %d", self._index)
                self._connected = False
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

            self._connected = True
            logger.info("Camera opened successfully (index=%d)", self._index)
            return True
        except Exception as e:
            logger.error("Failed to open camera: %s", e)
            self._connected = False
            return False

    def _close(self):
        """Release the camera device."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None
        self._connected = False

    def _capture_loop(self):
        """Continuously read frames from the camera."""
        reconnect_delay = 2.0
        while self._running:
            if not self._connected or self._cap is None or not self._cap.isOpened():
                logger.warning("Camera disconnected. Attempting reconnect in %.1fs...", reconnect_delay)
                time.sleep(reconnect_delay)
                self._close()
                if self._open():
                    logger.info("Camera reconnected.")
                continue

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera.")
                self._connected = False
                continue

            with self._lock:
                self._frame = frame

            # FPS calculation
            self._frame_count += 1
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._last_fps_time = now

            # Small sleep to limit CPU usage  (~60 fps max capture rate)
            time.sleep(0.008)

    def test_connection(self) -> bool:
        """Quick test to verify camera access without starting the capture thread."""
        try:
            cap = cv2.VideoCapture(self._index)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    logger.info("Camera test passed (index=%d)", self._index)
                    return True
            cap.release()
            logger.error("Camera test failed (index=%d)", self._index)
            return False
        except Exception as e:
            logger.error("Camera test error: %s", e)
            return False
