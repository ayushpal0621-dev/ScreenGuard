"""
ScreenGuard Alert System

Handles all user-facing privacy alerts:
- Desktop notifications
- Alert sounds
- Visual warnings (via callback to GUI)
- Console/log events

Includes cooldown logic to prevent alert spam.
"""

import time
import logging
import threading
import subprocess
import platform
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Manages privacy alerts with cooldown and multiple output channels.
    
    Thread-safe: can be triggered from the detection thread.
    """

    def __init__(
        self,
        cooldown_seconds: float = 30.0,
        sound_enabled: bool = True,
        notification_enabled: bool = True,
        visual_enabled: bool = True,
        on_visual_alert: Optional[Callable[[str, str], None]] = None,
    ):
        self._cooldown_seconds = cooldown_seconds
        self._sound_enabled = sound_enabled
        self._notification_enabled = notification_enabled
        self._visual_enabled = visual_enabled
        self._on_visual_alert = on_visual_alert

        self._last_alert_time: float = 0.0
        self._alert_count: int = 0
        self._lock = threading.Lock()

    @property
    def alert_count(self) -> int:
        return self._alert_count

    def trigger_alert(self, message: str = "Another person has been detected near your screen."):
        """
        Trigger a privacy alert if cooldown has elapsed.
        
        Returns True if the alert was triggered, False if suppressed by cooldown.
        """
        with self._lock:
            now = time.time()
            if now - self._last_alert_time < self._cooldown_seconds:
                remaining = self._cooldown_seconds - (now - self._last_alert_time)
                logger.debug("Alert suppressed by cooldown (%.1fs remaining)", remaining)
                return False

            self._last_alert_time = now
            self._alert_count += 1

        logger.warning("⚠ PRIVACY ALERT: %s (alert #%d)", message, self._alert_count)

        # Fire all channels in separate threads to avoid blocking
        if self._notification_enabled:
            threading.Thread(
                target=self._send_notification,
                args=("⚠ ScreenGuard Privacy Alert", message),
                daemon=True,
            ).start()

        if self._sound_enabled:
            threading.Thread(
                target=self._play_alert_sound,
                daemon=True,
            ).start()

        if self._visual_enabled and self._on_visual_alert:
            try:
                self._on_visual_alert("⚠ PRIVACY ALERT", message)
            except Exception as e:
                logger.error("Visual alert callback failed: %s", e)

        return True

    def _send_notification(self, title: str, message: str):
        """Send a desktop notification using OS-native methods."""
        system = platform.system()
        try:
            if system == "Darwin":  # macOS
                subprocess.run(
                    [
                        "osascript", "-e",
                        f'display notification "{message}" with title "{title}" sound name "Sosumi"'
                    ],
                    capture_output=True,
                    timeout=5,
                )
                logger.debug("macOS notification sent.")

            elif system == "Linux":
                subprocess.run(
                    ["notify-send", "-u", "critical", title, message],
                    capture_output=True,
                    timeout=5,
                )
                logger.debug("Linux notification sent.")

            elif system == "Windows":
                try:
                    from plyer import notification
                    notification.notify(
                        title=title,
                        message=message,
                        app_name="ScreenGuard",
                        timeout=10,
                    )
                    logger.debug("Windows notification sent.")
                except ImportError:
                    # Fallback to PowerShell toast
                    ps_script = (
                        f'[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, '
                        f'ContentType = WindowsRuntime] > $null; '
                        f'$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(0); '
                        f'$template.GetElementsByTagName("text")[0].AppendChild($template.CreateTextNode("{title}")); '
                        f'$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("ScreenGuard"); '
                        f'$notifier.Show([Windows.UI.Notifications.ToastNotification]::new($template))'
                    )
                    subprocess.run(
                        ["powershell", "-Command", ps_script],
                        capture_output=True,
                        timeout=5,
                    )

        except Exception as e:
            logger.error("Failed to send notification: %s", e)

    def _play_alert_sound(self):
        """Play an alert sound using OS-native methods."""
        system = platform.system()
        try:
            if system == "Darwin":
                # Use macOS system sounds
                subprocess.run(
                    ["afplay", "/System/Library/Sounds/Sosumi.aiff"],
                    capture_output=True,
                    timeout=5,
                )
            elif system == "Linux":
                subprocess.run(
                    ["paplay", "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"],
                    capture_output=True,
                    timeout=5,
                )
            elif system == "Windows":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)

            logger.debug("Alert sound played.")
        except Exception as e:
            logger.error("Failed to play alert sound: %s", e)

    def reset_cooldown(self):
        """Reset the cooldown timer, allowing an immediate next alert."""
        with self._lock:
            self._last_alert_time = 0.0

    def set_visual_callback(self, callback: Optional[Callable[[str, str], None]]):
        """Set or update the visual alert callback (used by GUI)."""
        self._on_visual_alert = callback

    def update_settings(
        self,
        cooldown_seconds: Optional[float] = None,
        sound_enabled: Optional[bool] = None,
        notification_enabled: Optional[bool] = None,
        visual_enabled: Optional[bool] = None,
    ):
        """Update alert settings at runtime."""
        if cooldown_seconds is not None:
            self._cooldown_seconds = cooldown_seconds
        if sound_enabled is not None:
            self._sound_enabled = sound_enabled
        if notification_enabled is not None:
            self._notification_enabled = notification_enabled
        if visual_enabled is not None:
            self._visual_enabled = visual_enabled
