"""
ScreenGuard Monitor

Background monitoring thread that ties together:
Camera → Face Detector → Face Recognizer → Detection Engine → Alerts/Lock

This runs independently of the GUI and communicates via Qt signals.
"""

import time
import logging
import cv2
import numpy as np
from typing import Optional

from PySide6.QtCore import QThread, Signal, QMutex

from screenguard.camera import Camera
from screenguard.face_detector import FaceDetector, DetectionResult
from screenguard.face_recognizer import FaceRecognizer, IdentifiedFace, FaceIdentity
from screenguard.detection_engine import DetectionEngine, PrivacyState
from screenguard.alert_system import AlertSystem
from screenguard.security_manager import SecurityManager
from screenguard.event_logger import EventLogger
from screenguard.config import Config

logger = logging.getLogger(__name__)


class MonitorThread(QThread):
    """
    Background thread that runs the full privacy monitoring pipeline.
    
    Signals:
        frame_ready(np.ndarray, list): Emits processed frame and identified faces
        status_updated(dict): Emits current status for the GUI
        alert_triggered(str, str): Emits alert title and message
        state_changed(str, str): Emits old state and new state
        registration_progress(int, str): Emits sample count and message
        error_occurred(str): Emits error message
    """

    frame_ready = Signal(np.ndarray, list)
    status_updated = Signal(dict)
    alert_triggered = Signal(str, str)
    state_changed = Signal(str, str)
    registration_progress = Signal(int, str)
    error_occurred = Signal(str)

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self._config = config
        self._running = False
        self._paused = False
        self._mutex = QMutex()

        # Components
        self._camera: Optional[Camera] = None
        self._detector: Optional[FaceDetector] = None
        self._recognizer: Optional[FaceRecognizer] = None
        self._engine: Optional[DetectionEngine] = None
        self._alert_system: Optional[AlertSystem] = None
        self._security_manager: Optional[SecurityManager] = None
        self._event_logger: Optional[EventLogger] = None

        # Registration state
        self._registration_mode = False

    @property
    def is_monitoring(self) -> bool:
        return self._running and not self._paused

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def recognizer(self) -> Optional[FaceRecognizer]:
        return self._recognizer

    @property
    def event_logger(self) -> Optional[EventLogger]:
        return self._event_logger

    @property
    def security_manager(self) -> Optional[SecurityManager]:
        return self._security_manager

    @property
    def engine(self) -> Optional[DetectionEngine]:
        return self._engine

    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Camera
            self._camera = Camera(
                camera_index=self._config["camera_index"],
                width=self._config["frame_width"],
                height=self._config["frame_height"],
            )

            # Face detector
            self._detector = FaceDetector(
                confidence_threshold=self._config["face_confidence_threshold"],
                min_face_size_ratio=self._config["min_face_size_ratio"],
            )
            if not self._detector.initialize():
                self.error_occurred.emit("Failed to initialize face detector.")
                return False

            # Face recognizer
            self._recognizer = FaceRecognizer()

            # Alert system
            self._alert_system = AlertSystem(
                cooldown_seconds=self._config["alert_cooldown_seconds"],
                sound_enabled=self._config["alert_sound_enabled"],
                notification_enabled=self._config["desktop_notification_enabled"],
                visual_enabled=self._config["visual_warning_enabled"],
                on_visual_alert=self._on_visual_alert,
            )

            # Security manager
            self._security_manager = SecurityManager(
                dry_run=self._config["dry_run"],
                max_locks_per_minute=self._config["max_locks_per_minute"],
            )

            # Event logger
            if self._config["event_logging_enabled"]:
                self._event_logger = EventLogger()
            else:
                self._event_logger = None

            # Detection engine
            self._engine = DetectionEngine(
                confirmation_seconds=self._config["confirmation_seconds"],
                lock_delay_seconds=self._config["lock_delay_seconds"],
                consecutive_frames_required=self._config["consecutive_frames_required"],
                on_state_change=self._on_state_change,
                on_threat_confirmed=self._on_threat_confirmed,
            )

            logger.info("Monitor initialized successfully.")
            return True

        except Exception as e:
            logger.error("Monitor initialization failed: %s", e)
            self.error_occurred.emit(f"Initialization failed: {e}")
            return False

    def run(self):
        """Main monitoring loop (runs in background thread)."""
        if not self._camera:
            self.error_occurred.emit("Camera not initialized.")
            return

        if not self._camera.start():
            self.error_occurred.emit("Failed to start camera.")
            return

        self._running = True
        target_interval = 1.0 / max(1, self._config["detection_fps"])

        if self._event_logger:
            self._event_logger.log_event("STARTED", details="Protection monitoring started.")

        logger.info("Monitoring started. Target FPS: %d", self._config["detection_fps"])

        while self._running:
            loop_start = time.time()

            if self._paused:
                time.sleep(0.1)
                continue

            # Read frame
            frame = self._camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            try:
                if self._registration_mode:
                    self._process_registration_frame(frame)
                else:
                    self._process_monitoring_frame(frame)
            except Exception as e:
                logger.error("Frame processing error: %s", e)

            # Throttle to target FPS
            elapsed = time.time() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup
        self._camera.stop()
        if self._detector:
            self._detector.release()
        if self._event_logger:
            self._event_logger.log_event("STOPPED", details="Protection monitoring stopped.")
        logger.info("Monitoring stopped.")

    def _process_monitoring_frame(self, frame: np.ndarray):
        """Full monitoring pipeline for a single frame."""
        timestamp = time.time()

        # 1. Detect faces
        detection = self._detector.detect(frame, timestamp)

        # 2. Identify faces
        bboxes = [face.bbox for face in detection.faces]
        identified = self._recognizer.identify_faces(frame, bboxes)

        # 3. Update detection engine
        self._engine.update(identified)

        # 4. Emit frame and status to GUI
        status = self._engine.status
        self.frame_ready.emit(frame, identified)
        self.status_updated.emit({
            "state": status.state.value,
            "primary_user": status.primary_user_detected,
            "unknown_count": status.unknown_person_count,
            "total_faces": status.total_faces,
            "threat_level": status.threat_level,
            "verification_progress": status.verification_progress,
            "fps": self._camera.fps,
            "camera_connected": self._camera.is_connected,
        })

    def _process_registration_frame(self, frame: np.ndarray):
        """Process a frame during registration mode — just show detection."""
        detection = self._detector.detect(frame)
        identified = []
        for face in detection.faces:
            identified.append(IdentifiedFace(
                bbox=face.bbox,
                confidence=face.confidence,
                identity=FaceIdentity.UNREGISTERED,
                similarity=0.0,
            ))
        self.frame_ready.emit(frame, identified)

    def _on_state_change(self, old_state: PrivacyState, new_state: PrivacyState):
        """Callback when detection engine state changes."""
        self.state_changed.emit(old_state.value, new_state.value)

        if self._event_logger:
            status = self._engine.status
            self._event_logger.log_event(
                event_type=new_state.value,
                face_count=status.total_faces,
                primary_user_detected=status.primary_user_detected,
                unknown_person_detected=status.unknown_person_count > 0,
                details=f"Transition: {old_state.value} → {new_state.value}",
            )

    def _on_threat_confirmed(self):
        """Callback when a privacy threat is confirmed by the engine."""
        mode = self._config["mode"]

        # Always alert
        if mode in ("alert", "alert_and_lock"):
            self._alert_system.trigger_alert()
            if self._event_logger:
                self._event_logger.log_event("ALERT", action="alert_triggered")

        # Lock if configured
        if mode in ("lock", "alert_and_lock"):
            success = self._security_manager.lock_computer("Privacy threat confirmed")
            if self._event_logger:
                action = "lock_executed" if success else "lock_failed"
                self._event_logger.log_event("LOCK", action=action)

    def _on_visual_alert(self, title: str, message: str):
        """Forward visual alerts to the GUI via signal."""
        self.alert_triggered.emit(title, message)

    # ── Registration ───────────────────────────────────────────────

    def start_registration(self):
        """Enter registration mode."""
        self._registration_mode = True
        if self._recognizer:
            self._recognizer.start_registration()
        logger.info("Registration mode started.")

    def add_registration_sample(self, frame: np.ndarray) -> tuple[bool, str]:
        """Add a registration sample from the current frame."""
        if not self._recognizer:
            return False, "Recognizer not initialized."
        return self._recognizer.add_registration_sample(frame)

    def finish_registration(self) -> tuple[bool, str]:
        """Complete registration and return to monitoring mode."""
        if not self._recognizer:
            return False, "Recognizer not initialized."
        success, message = self._recognizer.finish_registration()
        self._registration_mode = False
        if success and self._event_logger:
            self._event_logger.log_event("REGISTRATION", action="user_registered")
        return success, message

    def cancel_registration(self):
        """Cancel ongoing registration."""
        self._registration_mode = False
        logger.info("Registration cancelled.")

    # ── Controls ───────────────────────────────────────────────────

    def pause(self):
        self._paused = True
        logger.info("Monitoring paused.")

    def resume(self):
        self._paused = False
        if self._engine:
            self._engine.reset()
        logger.info("Monitoring resumed.")

    def stop_monitoring(self):
        self._running = False
        logger.info("Monitoring stop requested.")
