"""
ScreenGuard Detection Engine

State machine that manages the privacy-detection lifecycle.

States:
    SAFE                    → Only primary user or no one visible
    UNKNOWN_PERSON_DETECTED → Unknown face first appeared
    VERIFYING               → Confirming the unknown face persists
    PRIVACY_THREAT          → Confirmed threat, trigger action

Transitions require consecutive-frame confirmation and configurable delays
to prevent false positives from single-frame glitches.
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable

from screenguard.face_recognizer import FaceIdentity, IdentifiedFace

logger = logging.getLogger(__name__)


class PrivacyState(Enum):
    SAFE = "SAFE"
    UNKNOWN_PERSON_DETECTED = "UNKNOWN_PERSON_DETECTED"
    VERIFYING = "VERIFYING"
    PRIVACY_THREAT = "PRIVACY_THREAT"
    UNATTENDED = "UNATTENDED"  # No primary user visible


@dataclass
class EngineStatus:
    """Snapshot of the current detection engine state."""
    state: PrivacyState
    primary_user_detected: bool
    unknown_person_count: int
    total_faces: int
    verification_progress: float  # 0.0 to 1.0
    time_in_state: float
    threat_level: float  # 0.0 to 1.0


class DetectionEngine:
    """
    Core state machine for privacy threat detection.
    
    Feed it IdentifiedFace results each frame, and it manages state transitions
    with configurable timing and consecutive-frame requirements.
    """

    def __init__(
        self,
        confirmation_seconds: float = 2.0,
        lock_delay_seconds: float = 3.0,
        consecutive_frames_required: int = 5,
        on_state_change: Optional[Callable[[PrivacyState, PrivacyState], None]] = None,
        on_threat_confirmed: Optional[Callable[[], None]] = None,
    ):
        self._confirmation_seconds = confirmation_seconds
        self._lock_delay_seconds = lock_delay_seconds
        self._consecutive_frames_required = consecutive_frames_required
        self._on_state_change = on_state_change
        self._on_threat_confirmed = on_threat_confirmed

        # State
        self._state = PrivacyState.SAFE
        self._state_entered_at = time.time()

        # Consecutive frame tracking
        self._unknown_consecutive_frames = 0
        self._safe_consecutive_frames = 0

        # Verification timing
        self._verification_start_time: Optional[float] = None

        # Face tracking
        self._primary_user_detected = False
        self._unknown_person_count = 0
        self._total_faces = 0

        # Threat tracking
        self._threat_triggered = False

    @property
    def state(self) -> PrivacyState:
        return self._state

    @property
    def status(self) -> EngineStatus:
        now = time.time()
        verification_progress = 0.0
        if self._state == PrivacyState.VERIFYING and self._verification_start_time:
            elapsed = now - self._verification_start_time
            total_time = self._confirmation_seconds + self._lock_delay_seconds
            verification_progress = min(1.0, elapsed / total_time)

        return EngineStatus(
            state=self._state,
            primary_user_detected=self._primary_user_detected,
            unknown_person_count=self._unknown_person_count,
            total_faces=self._total_faces,
            verification_progress=verification_progress,
            time_in_state=now - self._state_entered_at,
            threat_level=self._compute_threat_level(),
        )

    def update(self, identified_faces: list[IdentifiedFace]):
        """
        Process a new frame's identification results and update the state machine.
        
        Call this once per processed frame.
        """
        # Classify faces
        self._primary_user_detected = any(
            f.identity == FaceIdentity.PRIMARY_USER for f in identified_faces
        )
        unknown_faces = [f for f in identified_faces if f.identity == FaceIdentity.UNKNOWN_PERSON]
        self._unknown_person_count = len(unknown_faces)
        self._total_faces = len(identified_faces)

        has_unknown = self._unknown_person_count > 0
        now = time.time()

        # Update consecutive frame counters
        if has_unknown:
            self._unknown_consecutive_frames += 1
            self._safe_consecutive_frames = 0
        else:
            self._safe_consecutive_frames += 1
            self._unknown_consecutive_frames = 0

        # State machine transitions
        old_state = self._state

        if self._state == PrivacyState.SAFE:
            self._handle_safe_state(has_unknown, now)

        elif self._state == PrivacyState.UNKNOWN_PERSON_DETECTED:
            self._handle_detected_state(has_unknown, now)

        elif self._state == PrivacyState.VERIFYING:
            self._handle_verifying_state(has_unknown, now)

        elif self._state == PrivacyState.PRIVACY_THREAT:
            self._handle_threat_state(has_unknown, now)

        elif self._state == PrivacyState.UNATTENDED:
            self._handle_unattended_state(has_unknown, now)

        # Notify on state change
        if self._state != old_state:
            logger.info("State transition: %s → %s", old_state.value, self._state.value)
            self._state_entered_at = now
            if self._on_state_change:
                self._on_state_change(old_state, self._state)

    def _handle_safe_state(self, has_unknown: bool, now: float):
        if has_unknown and self._unknown_consecutive_frames >= self._consecutive_frames_required:
            self._transition_to(PrivacyState.UNKNOWN_PERSON_DETECTED)

        if not self._primary_user_detected and self._total_faces == 0:
            # No one visible — could be unattended
            if self._safe_consecutive_frames > self._consecutive_frames_required * 3:
                self._transition_to(PrivacyState.UNATTENDED)

    def _handle_detected_state(self, has_unknown: bool, now: float):
        if not has_unknown:
            # Unknown person left — back to safe
            if self._safe_consecutive_frames >= self._consecutive_frames_required:
                self._transition_to(PrivacyState.SAFE)
            return

        # Unknown person persists — start verification timer
        self._verification_start_time = now
        self._transition_to(PrivacyState.VERIFYING)

    def _handle_verifying_state(self, has_unknown: bool, now: float):
        if not has_unknown:
            # Unknown person left during verification — cancel
            if self._safe_consecutive_frames >= self._consecutive_frames_required:
                logger.info("Unknown person left during verification. Returning to SAFE.")
                self._verification_start_time = None
                self._transition_to(PrivacyState.SAFE)
            return

        # Check if verification period has elapsed
        if self._verification_start_time:
            elapsed = now - self._verification_start_time
            total_required = self._confirmation_seconds + self._lock_delay_seconds
            if elapsed >= total_required:
                self._transition_to(PrivacyState.PRIVACY_THREAT)
                self._threat_triggered = True
                if self._on_threat_confirmed:
                    self._on_threat_confirmed()

    def _handle_threat_state(self, has_unknown: bool, now: float):
        if not has_unknown:
            # Threat cleared
            if self._safe_consecutive_frames >= self._consecutive_frames_required * 2:
                self._threat_triggered = False
                self._verification_start_time = None
                self._transition_to(PrivacyState.SAFE)

    def _handle_unattended_state(self, has_unknown: bool, now: float):
        if self._primary_user_detected:
            self._transition_to(PrivacyState.SAFE)
        elif has_unknown and not self._primary_user_detected:
            # Unknown person at unattended computer
            self._transition_to(PrivacyState.UNKNOWN_PERSON_DETECTED)

    def _transition_to(self, new_state: PrivacyState):
        self._state = new_state

    def _compute_threat_level(self) -> float:
        """Compute a 0-1 threat level based on current state and timing."""
        if self._state == PrivacyState.SAFE:
            return 0.0
        elif self._state == PrivacyState.UNKNOWN_PERSON_DETECTED:
            return 0.3
        elif self._state == PrivacyState.VERIFYING:
            # Ramp up during verification
            if self._verification_start_time:
                elapsed = time.time() - self._verification_start_time
                total = self._confirmation_seconds + self._lock_delay_seconds
                return 0.3 + 0.5 * min(1.0, elapsed / total)
            return 0.5
        elif self._state == PrivacyState.PRIVACY_THREAT:
            return 1.0
        elif self._state == PrivacyState.UNATTENDED:
            return 0.1
        return 0.0

    def reset(self):
        """Reset the engine to SAFE state."""
        self._state = PrivacyState.SAFE
        self._state_entered_at = time.time()
        self._unknown_consecutive_frames = 0
        self._safe_consecutive_frames = 0
        self._verification_start_time = None
        self._threat_triggered = False
        self._primary_user_detected = False
        self._unknown_person_count = 0
        self._total_faces = 0
        logger.info("Detection engine reset to SAFE.")
