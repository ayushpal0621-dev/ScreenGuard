"""
ScreenGuard Security Manager

Handles the OS-level computer lock action with safety controls:
- Dry-run mode for testing
- Rate limiting to prevent lock spam
- OS-specific lock implementations
"""

import time
import logging
import subprocess
import platform
from typing import Optional

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Manages computer lock actions with safety controls.
    
    Safety features:
        - dry_run mode: logs the action instead of executing it
        - Rate limiting: max N locks per minute
        - Cooldown: minimum time between locks
    """

    def __init__(
        self,
        dry_run: bool = True,
        max_locks_per_minute: int = 2,
        lock_cooldown_seconds: float = 30.0,
    ):
        self._dry_run = dry_run
        self._max_locks_per_minute = max_locks_per_minute
        self._lock_cooldown_seconds = lock_cooldown_seconds

        self._lock_timestamps: list[float] = []
        self._last_lock_time: float = 0.0
        self._total_locks: int = 0

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @dry_run.setter
    def dry_run(self, value: bool):
        self._dry_run = value
        logger.info("Security manager dry_run set to %s", value)

    @property
    def total_locks(self) -> int:
        return self._total_locks

    def lock_computer(self, reason: str = "Privacy threat detected") -> bool:
        """
        Lock the computer screen.
        
        Returns True if lock was executed (or would be in dry-run mode).
        Returns False if rate-limited or failed.
        """
        now = time.time()

        # Rate limiting
        if not self._check_rate_limit(now):
            logger.warning("Lock rate-limited. Skipping lock action.")
            return False

        # Cooldown check
        if now - self._last_lock_time < self._lock_cooldown_seconds:
            logger.warning("Lock cooldown active (%.1fs remaining). Skipping.",
                           self._lock_cooldown_seconds - (now - self._last_lock_time))
            return False

        self._last_lock_time = now
        self._lock_timestamps.append(now)
        self._total_locks += 1

        if self._dry_run:
            logger.info("🔒 DRY RUN: Would lock computer now. Reason: %s", reason)
            return True

        # Actually lock the computer
        return self._execute_lock(reason)

    def _check_rate_limit(self, now: float) -> bool:
        """Check if we've exceeded the max locks per minute."""
        # Remove timestamps older than 60 seconds
        self._lock_timestamps = [t for t in self._lock_timestamps if now - t < 60.0]

        if len(self._lock_timestamps) >= self._max_locks_per_minute:
            return False
        return True

    def _execute_lock(self, reason: str) -> bool:
        """Execute the OS-specific screen lock command."""
        system = platform.system()
        logger.warning("🔒 LOCKING COMPUTER. Reason: %s", reason)

        try:
            if system == "Darwin":  # macOS
                return self._lock_macos()

            elif system == "Linux":
                # Try multiple lock commands
                for cmd in [
                    ["loginctl", "lock-session"],
                    ["xdg-screensaver", "lock"],
                    ["gnome-screensaver-command", "-l"],
                    ["xscreensaver-command", "-lock"],
                ]:
                    try:
                        result = subprocess.run(cmd, capture_output=True, timeout=5)
                        if result.returncode == 0:
                            return True
                    except FileNotFoundError:
                        continue
                logger.error("No suitable lock command found on Linux.")
                return False

            elif system == "Windows":
                subprocess.run(
                    ["rundll32.exe", "user32.dll,LockWorkStation"],
                    capture_output=True,
                    timeout=5,
                )
                return True

            else:
                logger.error("Unsupported OS for lock: %s", system)
                return False

        except Exception as e:
            logger.error("Failed to lock computer: %s", e)
            return False

    def _lock_macos(self) -> bool:
        """Lock macOS using native private frameworks and fallback commands."""
        import ctypes

        # 1. Native macOS login.framework SACLockScreenImmediate (direct, instant, no permissions needed)
        try:
            login_framework = ctypes.cdll.LoadLibrary(
                "/System/Library/PrivateFrameworks/login.framework/Versions/Current/login"
            )
            if hasattr(login_framework, "SACLockScreenImmediate"):
                login_framework.SACLockScreenImmediate()
                logger.info("macOS screen locked via native SACLockScreenImmediate")
                return True
        except Exception as e:
            logger.debug("SACLockScreenImmediate fallback triggered: %s", e)

        # 2. AppleScript keystroke
        try:
            res = subprocess.run(
                [
                    "osascript", "-e",
                    'tell application "System Events" to keystroke "q" using {control down, command down}'
                ],
                capture_output=True,
                timeout=5,
            )
            if res.returncode == 0:
                logger.info("macOS screen locked via System Events shortcut")
                return True
        except Exception as e:
            logger.debug("System Events keystroke failed: %s", e)

        # 3. pmset displaysleepnow
        try:
            res = subprocess.run(["pmset", "displaysleepnow"], capture_output=True, timeout=5)
            if res.returncode == 0:
                logger.info("macOS screen locked via pmset displaysleepnow")
                return True
        except Exception as e:
            logger.debug("pmset displaysleepnow failed: %s", e)

        return False

    def reset_rate_limit(self):
        """Clear rate limiting counters."""
        self._lock_timestamps.clear()
        self._last_lock_time = 0.0

