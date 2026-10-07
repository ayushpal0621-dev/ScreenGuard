#!/usr/bin/env python3
"""
ScreenGuard — AI Screen Privacy Protection System

Main entry point. Supports both GUI and CLI modes.

Usage:
    python main.py              # Launch the GUI
    python main.py --test-cam   # Test webcam access only
    python main.py --cli        # Run in CLI mode (no GUI)
    python main.py --debug      # Launch GUI with debug window
"""

import sys
import os
import argparse
import logging
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(level: int = logging.INFO) -> None:
    """Configure application logging."""
    log_format = "%(asctime)s │ %(levelname)-8s │ %(name)-25s │ %(message)s"
    date_format = "%H:%M:%S"

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Suppress noisy third-party logs
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def test_camera(camera_index: int = 0) -> bool:
    """Quick camera test — verifies webcam access."""
    import cv2

    logger = logging.getLogger("test_camera")
    logger.info("Testing camera access (index=%d)...", camera_index)

    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("❌ Cannot open camera index %d", camera_index)
            return False

        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("❌ Cannot read frame from camera")
            cap.release()
            return False

        h, w = frame.shape[:2]
        logger.info("✅ Camera opened successfully!")
        logger.info("   Resolution: %dx%d", w, h)
        logger.info("   Frame shape: %s", frame.shape)

        cap.release()
        logger.info("✅ Camera released cleanly.")
        return True

    except Exception as e:
        logger.error("❌ Camera test failed: %s", e)
        return False


def run_cli(config):
    """Run ScreenGuard in CLI mode (no GUI)."""
    import cv2
    import numpy as np
    from screenguard.camera import Camera
    from screenguard.face_detector import FaceDetector
    from screenguard.face_recognizer import FaceRecognizer, FaceIdentity
    from screenguard.detection_engine import DetectionEngine, PrivacyState
    from screenguard.alert_system import AlertSystem
    from screenguard.security_manager import SecurityManager
    from screenguard.event_logger import EventLogger

    logger = logging.getLogger("cli")
    logger.info("Starting ScreenGuard in CLI mode...")

    # Initialize components
    camera = Camera(
        camera_index=config["camera_index"],
        width=config["frame_width"],
        height=config["frame_height"],
    )

    detector = FaceDetector(
        confidence_threshold=config["face_confidence_threshold"],
        min_face_size_ratio=config["min_face_size_ratio"],
    )

    recognizer = FaceRecognizer()
    alert_system = AlertSystem(
        cooldown_seconds=config["alert_cooldown_seconds"],
        sound_enabled=config["alert_sound_enabled"],
        notification_enabled=config["desktop_notification_enabled"],
    )
    security_manager = SecurityManager(
        dry_run=config["dry_run"],
        max_locks_per_minute=config["max_locks_per_minute"],
    )
    event_logger = EventLogger()

    def on_state_change(old, new):
        logger.info("State: %s → %s", old.value, new.value)
        event_logger.log_event(
            event_type=new.value,
            details=f"{old.value} → {new.value}",
        )

    def on_threat():
        mode = config["mode"]
        if mode in ("alert", "alert_and_lock"):
            alert_system.trigger_alert()
            event_logger.log_event("ALERT", action="alert_triggered")
        if mode in ("lock", "alert_and_lock"):
            success = security_manager.lock_computer()
            event_logger.log_event("LOCK", action="lock_executed" if success else "lock_failed")

    engine = DetectionEngine(
        confirmation_seconds=config["confirmation_seconds"],
        lock_delay_seconds=config["lock_delay_seconds"],
        consecutive_frames_required=config["consecutive_frames_required"],
        on_state_change=on_state_change,
        on_threat_confirmed=on_threat,
    )

    if not detector.initialize():
        logger.error("Failed to initialize face detector.")
        return

    if not camera.start():
        logger.error("Failed to start camera.")
        return

    show_debug = config.get("show_debug_window", False)
    logger.info("Monitoring started. Press Ctrl+C to stop.")
    if show_debug:
        logger.info("Debug window enabled. Press 'q' in the debug window to quit.")

    event_logger.log_event("STARTED", details="CLI mode started")

    try:
        target_interval = 1.0 / max(1, config["detection_fps"])
        while True:
            loop_start = time.time()

            frame = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            # Detect
            result = detector.detect(frame, time.time())

            # Identify
            bboxes = [f.bbox for f in result.faces]
            identified = recognizer.identify_faces(frame, bboxes)

            # Update engine
            engine.update(identified)
            status = engine.status

            # Debug output
            state_str = status.state.value
            primary = "✓" if status.primary_user_detected else "✗"
            unknown = status.unknown_person_count
            threat = f"{status.threat_level:.0%}"

            sys.stdout.write(
                f"\r  Faces: {status.total_faces} │ Primary: {primary} │ "
                f"Unknown: {unknown} │ State: {state_str:<25} │ "
                f"Threat: {threat} │ FPS: {camera.fps:.0f}  "
            )
            sys.stdout.flush()

            # Debug window
            if show_debug:
                display = frame.copy()
                for face in identified:
                    x, y, w, h = face.bbox
                    if face.identity == FaceIdentity.PRIMARY_USER:
                        color = (0, 200, 100)
                        label = "You"
                    elif face.identity == FaceIdentity.UNKNOWN_PERSON:
                        color = (0, 0, 255)
                        label = "Unknown"
                    else:
                        color = (128, 128, 128)
                        label = "Unregistered"

                    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display, label, (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Status overlay
                cv2.putText(display, f"State: {state_str}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(display, f"Faces: {status.total_faces}  FPS: {camera.fps:.0f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("ScreenGuard Debug", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Throttle
            elapsed = time.time() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print()  # Newline after \r output
        logger.info("Stopping...")

    finally:
        camera.stop()
        detector.release()
        if show_debug:
            cv2.destroyAllWindows()
        event_logger.log_event("STOPPED", details="CLI mode stopped")
        logger.info("ScreenGuard stopped cleanly.")


def run_gui(config):
    """Launch the PySide6 GUI application."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont
    from screenguard.gui import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("ScreenGuard")
    app.setOrganizationName("ScreenGuard")

    # Try to set a nice font
    font = QFont("Inter", 13)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    window = MainWindow(config)
    window.show()

    sys.exit(app.exec())


def main():
    parser = argparse.ArgumentParser(
        description="ScreenGuard — AI Screen Privacy Protection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test-cam", action="store_true",
                        help="Test webcam access and exit")
    parser.add_argument("--cli", action="store_true",
                        help="Run in CLI mode (no GUI)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug window with camera feed")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera index to use (default: 0)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Enable dry-run mode (simulate lock)")
    parser.add_argument("--no-dry-run", action="store_true", default=False,
                        help="Disable dry-run mode (actually lock screen)")
    parser.add_argument("--mode", choices=["alert", "lock", "alert_and_lock"], default=None,
                        help="Protection mode: alert, lock, or alert_and_lock")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose/debug logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    logger = logging.getLogger("main")
    logger.info("╔════════════════════════════════════════════════╗")
    logger.info("║   ScreenGuard — AI Screen Privacy Protection  ║")
    logger.info("║              Version 1.0.0                    ║")
    logger.info("╚════════════════════════════════════════════════╝")

    # Camera test mode
    if args.test_cam:
        cam_index = args.camera if args.camera is not None else 0
        success = test_camera(cam_index)
        sys.exit(0 if success else 1)

    # Load configuration
    from screenguard.config import Config
    config = Config()

    # Apply CLI overrides
    if args.camera is not None:
        config["camera_index"] = args.camera
    if args.dry_run:
        config["dry_run"] = True
    elif args.no_dry_run:
        config["dry_run"] = False
    if args.mode:
        config["mode"] = args.mode
    if args.debug:
        config["show_debug_window"] = True

    # Run in selected mode
    if args.cli:
        run_cli(config)
    else:
        run_gui(config)


if __name__ == "__main__":
    main()
