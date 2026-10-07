"""
ScreenGuard Test Suite

Comprehensive tests for all stages of the application.
Run with: python -m pytest tests/ -v
"""

import sys
import os
import time
import json
import tempfile
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from screenguard.config import Config


# ── Stage 0: Project Setup Tests ────────────────────────────────────

class TestProjectSetup:
    """Verify project structure and basic imports."""

    def test_imports(self):
        """All core modules should be importable."""
        from screenguard import __version__, __app_name__
        assert __version__ == "1.0.0"
        assert __app_name__ == "ScreenGuard"

    def test_config_module(self):
        from screenguard.config import Config, DEFAULTS
        assert isinstance(DEFAULTS, dict)
        assert "protection_enabled" in DEFAULTS

    def test_camera_module(self):
        from screenguard.camera import Camera
        camera = Camera(camera_index=99)  # Non-existent camera
        assert not camera.is_connected

    def test_detector_module(self):
        from screenguard.face_detector import FaceDetector, DetectedFace, DetectionResult
        assert FaceDetector is not None

    def test_recognizer_module(self):
        from screenguard.face_recognizer import FaceRecognizer, FaceIdentity
        assert FaceIdentity.PRIMARY_USER.value == "PRIMARY_USER"
        assert FaceIdentity.UNKNOWN_PERSON.value == "UNKNOWN_PERSON"

    def test_engine_module(self):
        from screenguard.detection_engine import DetectionEngine, PrivacyState
        assert PrivacyState.SAFE.value == "SAFE"
        assert PrivacyState.PRIVACY_THREAT.value == "PRIVACY_THREAT"


# ── Stage 1: Configuration Tests ────────────────────────────────────

class TestConfig:
    """Test configuration management."""

    def test_default_values(self):
        from screenguard.config import Config
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(os.path.join(tmpdir, "test_config.json"))
            assert config["protection_enabled"] is True
            assert config["dry_run"] is False
            assert config["confirmation_seconds"] == 2.0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_config.json")
            config = Config(path)
            config["confirmation_seconds"] = 5.0
            config.save()

            config2 = Config(path)
            assert config2["confirmation_seconds"] == 5.0

    def test_get_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(os.path.join(tmpdir, "test_config.json"))
            config.set("test_key", "test_value")
            assert config.get("test_key") == "test_value"

    def test_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(os.path.join(tmpdir, "test_config.json"))
            config["confirmation_seconds"] = 99.0
            config.reset()
            assert config["confirmation_seconds"] == 2.0


# ── Stage 2: Face Detector Tests ────────────────────────────────────

class TestFaceDetector:
    """Test face detection functionality."""

    def test_initialize(self):
        from screenguard.face_detector import FaceDetector
        detector = FaceDetector()
        assert detector.initialize()
        assert detector.is_initialized
        detector.release()
        assert not detector.is_initialized

    def test_detect_empty_frame(self):
        from screenguard.face_detector import FaceDetector
        detector = FaceDetector()
        detector.initialize()

        # Black frame — no faces
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result.face_count == 0
        assert not result.has_faces
        assert result.frame_shape == (480, 640)

        detector.release()

    def test_detect_without_init(self):
        from screenguard.face_detector import FaceDetector
        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)
        assert result.face_count == 0

    def test_detected_face_dataclass(self):
        from screenguard.face_detector import DetectedFace
        face = DetectedFace(bbox=(100, 100, 50, 60), confidence=0.95)
        assert face.center == (125, 130)
        assert face.area == 3000


# ── Stage 3: Face Recognizer Tests ──────────────────────────────────

class TestFaceRecognizer:
    """Test face recognition and registration."""

    def test_unregistered_state(self):
        from screenguard.face_recognizer import FaceRecognizer, FaceIdentity
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = FaceRecognizer(data_dir=tmpdir)
            assert not recognizer.is_registered

            # All faces should be UNREGISTERED
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            bboxes = [(100, 100, 50, 50)]
            results = recognizer.identify_faces(frame, bboxes)
            assert len(results) == 1
            assert results[0].identity == FaceIdentity.UNREGISTERED

    def test_registration_flow(self):
        from screenguard.face_recognizer import FaceRecognizer
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = FaceRecognizer(data_dir=tmpdir)
            recognizer.start_registration()
            assert recognizer.is_registering

    def test_delete_profile(self):
        from screenguard.face_recognizer import FaceRecognizer
        with tempfile.TemporaryDirectory() as tmpdir:
            recognizer = FaceRecognizer(data_dir=tmpdir)
            recognizer.delete_profile()
            assert not recognizer.is_registered


# ── Stage 4: Detection Engine Tests ─────────────────────────────────

class TestDetectionEngine:
    """Test the privacy detection state machine."""

    def test_initial_state(self):
        from screenguard.detection_engine import DetectionEngine, PrivacyState
        engine = DetectionEngine()
        assert engine.state == PrivacyState.SAFE

    def test_safe_with_no_faces(self):
        from screenguard.detection_engine import DetectionEngine, PrivacyState
        engine = DetectionEngine()
        engine.update([])
        assert engine.state == PrivacyState.SAFE

    def test_safe_with_primary_user(self):
        from screenguard.detection_engine import DetectionEngine, PrivacyState
        from screenguard.face_recognizer import IdentifiedFace, FaceIdentity

        engine = DetectionEngine(consecutive_frames_required=2)

        primary = IdentifiedFace(
            bbox=(100, 100, 50, 50),
            confidence=0.95,
            identity=FaceIdentity.PRIMARY_USER,
            similarity=0.9,
        )

        for _ in range(10):
            engine.update([primary])

        assert engine.state == PrivacyState.SAFE
        assert engine.status.primary_user_detected

    def test_unknown_person_detection(self):
        from screenguard.detection_engine import DetectionEngine, PrivacyState
        from screenguard.face_recognizer import IdentifiedFace, FaceIdentity

        engine = DetectionEngine(consecutive_frames_required=3)

        unknown = IdentifiedFace(
            bbox=(200, 100, 50, 50),
            confidence=0.9,
            identity=FaceIdentity.UNKNOWN_PERSON,
            similarity=0.1,
        )

        # Feed unknown person for enough frames
        for _ in range(5):
            engine.update([unknown])

        assert engine.state in (
            PrivacyState.UNKNOWN_PERSON_DETECTED,
            PrivacyState.VERIFYING,
        )

    def test_threat_requires_time(self):
        from screenguard.detection_engine import DetectionEngine, PrivacyState
        from screenguard.face_recognizer import IdentifiedFace, FaceIdentity

        threat_triggered = [False]

        def on_threat():
            threat_triggered[0] = True

        engine = DetectionEngine(
            confirmation_seconds=0.1,
            lock_delay_seconds=0.1,
            consecutive_frames_required=2,
            on_threat_confirmed=on_threat,
        )

        unknown = IdentifiedFace(
            bbox=(200, 100, 50, 50),
            confidence=0.9,
            identity=FaceIdentity.UNKNOWN_PERSON,
            similarity=0.1,
        )

        # Feed frames over time
        start = time.time()
        while time.time() - start < 1.0:
            engine.update([unknown])
            time.sleep(0.02)

        assert threat_triggered[0], "Threat should have been triggered"
        assert engine.state == PrivacyState.PRIVACY_THREAT

    def test_unknown_leaves_returns_safe(self):
        from screenguard.detection_engine import DetectionEngine, PrivacyState
        from screenguard.face_recognizer import IdentifiedFace, FaceIdentity

        engine = DetectionEngine(consecutive_frames_required=2)

        unknown = IdentifiedFace(
            bbox=(200, 100, 50, 50),
            confidence=0.9,
            identity=FaceIdentity.UNKNOWN_PERSON,
            similarity=0.1,
        )

        primary = IdentifiedFace(
            bbox=(100, 100, 50, 50),
            confidence=0.95,
            identity=FaceIdentity.PRIMARY_USER,
            similarity=0.9,
        )

        # Unknown appears
        for _ in range(5):
            engine.update([primary, unknown])

        assert engine.state != PrivacyState.SAFE

        # Unknown leaves
        for _ in range(10):
            engine.update([primary])

        assert engine.state == PrivacyState.SAFE

    def test_brief_appearance_no_threat(self):
        from screenguard.detection_engine import DetectionEngine, PrivacyState
        from screenguard.face_recognizer import IdentifiedFace, FaceIdentity

        threat_triggered = [False]
        def on_threat():
            threat_triggered[0] = True

        engine = DetectionEngine(
            confirmation_seconds=5.0,  # Long confirmation
            lock_delay_seconds=5.0,
            consecutive_frames_required=3,
            on_threat_confirmed=on_threat,
        )

        unknown = IdentifiedFace(
            bbox=(200, 100, 50, 50),
            confidence=0.9,
            identity=FaceIdentity.UNKNOWN_PERSON,
        )
        primary = IdentifiedFace(
            bbox=(100, 100, 50, 50),
            confidence=0.95,
            identity=FaceIdentity.PRIMARY_USER,
        )

        # Brief appearance (below threshold)
        engine.update([unknown])
        engine.update([unknown])
        # Then they leave
        for _ in range(10):
            engine.update([primary])

        assert not threat_triggered[0]

    def test_reset(self):
        from screenguard.detection_engine import DetectionEngine, PrivacyState
        engine = DetectionEngine()
        engine.reset()
        assert engine.state == PrivacyState.SAFE


# ── Stage 5: Alert System Tests ─────────────────────────────────────

class TestAlertSystem:
    """Test alert system with cooldowns."""

    def test_trigger_alert(self):
        from screenguard.alert_system import AlertSystem
        alerts = AlertSystem(
            cooldown_seconds=1.0,
            sound_enabled=False,
            notification_enabled=False,
            visual_enabled=False,
        )
        result = alerts.trigger_alert("Test alert")
        assert result is True
        assert alerts.alert_count == 1

    def test_cooldown(self):
        from screenguard.alert_system import AlertSystem
        alerts = AlertSystem(
            cooldown_seconds=60.0,
            sound_enabled=False,
            notification_enabled=False,
            visual_enabled=False,
        )
        alerts.trigger_alert("First")
        result = alerts.trigger_alert("Second")
        assert result is False  # Cooldown should suppress
        assert alerts.alert_count == 1

    def test_cooldown_reset(self):
        from screenguard.alert_system import AlertSystem
        alerts = AlertSystem(
            cooldown_seconds=60.0,
            sound_enabled=False,
            notification_enabled=False,
            visual_enabled=False,
        )
        alerts.trigger_alert("First")
        alerts.reset_cooldown()
        result = alerts.trigger_alert("Second")
        assert result is True
        assert alerts.alert_count == 2


# ── Stage 6: Security Manager Tests ────────────────────────────────

class TestSecurityManager:
    """Test security manager with dry-run and rate limiting."""

    def test_dry_run(self):
        from screenguard.security_manager import SecurityManager
        sm = SecurityManager(dry_run=True)
        assert sm.dry_run
        result = sm.lock_computer("Test")
        assert result is True  # Dry run "succeeds"
        assert sm.total_locks == 1

    def test_rate_limiting(self):
        from screenguard.security_manager import SecurityManager
        sm = SecurityManager(dry_run=True, max_locks_per_minute=2, lock_cooldown_seconds=0)
        sm.lock_computer("Lock 1")
        sm.lock_computer("Lock 2")
        result = sm.lock_computer("Lock 3")
        assert result is False  # Rate limited
        assert sm.total_locks == 2


# ── Stage 7: Event Logger Tests ─────────────────────────────────────

class TestEventLogger:
    """Test SQLite event logging."""

    def test_log_and_retrieve(self):
        from screenguard.event_logger import EventLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            logger = EventLogger(db_path=db_path)

            logger.log_event("TEST_EVENT", face_count=2, action="test")
            events = logger.get_recent_events(10)

            assert len(events) == 1
            assert events[0]["event_type"] == "TEST_EVENT"
            assert events[0]["face_count"] == 2

    def test_event_stats(self):
        from screenguard.event_logger import EventLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            el = EventLogger(db_path=db_path)

            el.log_event("PRIVACY_THREAT")
            el.log_event("ALERT")
            el.log_event("LOCK")
            el.log_event("SAFE")

            stats = el.get_event_stats()
            assert stats["total_events"] == 4
            assert stats["privacy_threats"] == 1
            assert stats["alerts_triggered"] == 1
            assert stats["locks_triggered"] == 1

    def test_clear_all(self):
        from screenguard.event_logger import EventLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            el = EventLogger(db_path=db_path)
            el.log_event("TEST")
            el.clear_all()
            events = el.get_recent_events()
            assert len(events) == 0


# ── Stage 8: Camera Tests ──────────────────────────────────────────

class TestCamera:
    """Test camera module (without requiring actual hardware)."""

    def test_camera_init(self):
        from screenguard.camera import Camera
        cam = Camera(camera_index=0, width=640, height=480)
        assert not cam.is_connected
        assert cam.fps == 0.0

    def test_read_without_start(self):
        from screenguard.camera import Camera
        cam = Camera(camera_index=99)
        frame = cam.read()
        assert frame is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
