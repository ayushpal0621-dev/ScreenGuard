"""
ScreenGuard Configuration Manager

Handles loading, saving, and providing default configuration values.
Configuration is stored as a local JSON file.
"""

import json
import os
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULTS = {
    # Protection
    "protection_enabled": True,
    "mode": "alert_and_lock",  # "alert", "lock", "alert_and_lock"

    # Detection timing
    "confirmation_seconds": 2.0,       # Consecutive seconds before confirming threat
    "lock_delay_seconds": 3.0,         # Additional seconds before locking
    "alert_cooldown_seconds": 30.0,    # Minimum seconds between repeated alerts

    # Camera
    "camera_index": 0,
    "detection_fps": 15,
    "frame_width": 640,
    "frame_height": 480,

    # Face detection
    "face_confidence_threshold": 0.5,
    "min_face_size_ratio": 0.02,       # Minimum face size as ratio of frame area
    "consecutive_frames_required": 5,   # Frames before confirming detection

    # Alert
    "alert_sound_enabled": True,
    "desktop_notification_enabled": True,
    "visual_warning_enabled": True,

    # Lock
    "dry_run": False,  # Screen locking active by default
    "max_locks_per_minute": 2,  # Prevent lock spam

    # Privacy
    "store_webcam_images": False,
    "event_logging_enabled": True,

    # Application
    "start_minimized": False,
    "minimize_to_tray": True,
    "start_on_login": False,
    "show_debug_window": False,  # Show CV debug window with bounding boxes
}


class Config:
    """Thread-safe configuration manager with file persistence."""

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            # Store config next to the application
            app_dir = Path(__file__).parent.parent
            self._path = app_dir / "config.json"
        else:
            self._path = Path(config_path)

        self._data: dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load configuration from disk, falling back to defaults."""
        self._data = dict(DEFAULTS)

        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    user_data = json.load(f)
                self._data.update(user_data)
                logger.info("Configuration loaded from %s", self._path)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load config from %s: %s. Using defaults.", self._path, e)
        else:
            logger.info("No config file found. Using defaults.")

    def save(self):
        """Persist current configuration to disk."""
        try:
            os.makedirs(self._path.parent, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=4)
            logger.info("Configuration saved to %s", self._path)
        except IOError as e:
            logger.error("Failed to save config: %s", e)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any):
        self._data[key] = value

    def reset(self):
        """Reset all configuration to defaults."""
        self._data = dict(DEFAULTS)

    def to_dict(self) -> dict:
        return dict(self._data)
