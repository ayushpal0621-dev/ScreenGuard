"""
ScreenGuard Face Recognizer

Handles primary user registration, face embedding generation,
and comparison of detected faces against the registered user.

Uses the `face_recognition` library (dlib-based) for robust face embeddings.
Falls back to a simple "first face = primary user" heuristic if the library
is unavailable or the user hasn't registered.
"""

import os
import json
import logging
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)

# Try importing face_recognition; fall back gracefully
try:
    import face_recognition as fr
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition library not available. Using simple mode.")


class FaceIdentity(Enum):
    PRIMARY_USER = "PRIMARY_USER"
    UNKNOWN_PERSON = "UNKNOWN_PERSON"
    UNREGISTERED = "UNREGISTERED"  # No primary user registered yet


@dataclass
class IdentifiedFace:
    """A detected face with an identity classification."""
    bbox: tuple[int, int, int, int]
    confidence: float
    identity: FaceIdentity
    similarity: float = 0.0  # 0-1, how similar to primary user (1 = identical)


class FaceRecognizer:
    """
    Manages primary user registration and face identification.
    
    Registration flow:
        1. Call start_registration() to begin
        2. Call add_registration_sample(frame) multiple times with different angles
        3. Call finish_registration() to compute and store the embedding
    
    Identification:
        - Call identify(frame, face_locations) to classify detected faces
    """

    PROFILE_DIR = "data/face_profiles"
    PROFILE_FILE = "primary_user.json"
    RECOGNITION_TOLERANCE = 0.5  # Lower = stricter matching
    MIN_REGISTRATION_SAMPLES = 3
    MAX_REGISTRATION_SAMPLES = 10

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir:
            self._profile_dir = Path(data_dir) / "face_profiles"
        else:
            self._profile_dir = Path(__file__).parent.parent / self.PROFILE_DIR

        self._primary_embedding: Optional[np.ndarray] = None
        self._is_registered = False
        self._registration_samples: list[np.ndarray] = []
        self._registering = False

        self._load_profile()

    @property
    def is_registered(self) -> bool:
        return self._is_registered

    @property
    def is_registering(self) -> bool:
        return self._registering

    @property
    def registration_sample_count(self) -> int:
        return len(self._registration_samples)

    def _load_profile(self):
        """Load the stored primary user embedding from disk."""
        profile_path = self._profile_dir / self.PROFILE_FILE
        if profile_path.exists():
            try:
                with open(profile_path, "r") as f:
                    data = json.load(f)
                self._primary_embedding = np.array(data["embedding"])
                self._is_registered = True
                logger.info("Primary user profile loaded from %s", profile_path)
            except Exception as e:
                logger.error("Failed to load profile: %s", e)
                self._is_registered = False

    def _save_profile(self):
        """Save the primary user embedding to disk."""
        if self._primary_embedding is None:
            return

        os.makedirs(self._profile_dir, exist_ok=True)
        profile_path = self._profile_dir / self.PROFILE_FILE
        data = {
            "embedding": self._primary_embedding.tolist(),
            "created_at": time.time(),
            "num_samples": len(self._registration_samples),
        }
        try:
            with open(profile_path, "w") as f:
                json.dump(data, f)
            logger.info("Primary user profile saved to %s", profile_path)
        except Exception as e:
            logger.error("Failed to save profile: %s", e)

    def delete_profile(self):
        """Delete the stored primary user profile."""
        profile_path = self._profile_dir / self.PROFILE_FILE
        if profile_path.exists():
            os.remove(profile_path)
            logger.info("Primary user profile deleted.")
        self._primary_embedding = None
        self._is_registered = False

    # ── Registration Flow ──────────────────────────────────────────

    def start_registration(self):
        """Begin the registration process."""
        self._registration_samples = []
        self._registering = True
        logger.info("Registration started. Provide %d-%d face samples.",
                     self.MIN_REGISTRATION_SAMPLES, self.MAX_REGISTRATION_SAMPLES)

    def add_registration_sample(self, frame: np.ndarray) -> tuple[bool, str]:
        """
        Add a frame as a registration sample.
        
        Returns:
            (success, message) tuple
        """
        if not self._registering:
            return False, "Registration not started."

        if not FACE_RECOGNITION_AVAILABLE:
            # Simple mode: just mark as registered after 1 sample
            self._registering = False
            self._is_registered = True
            self._primary_embedding = np.zeros(128)  # Placeholder
            logger.info("Simple registration complete (face_recognition not available).")
            return True, "Registered in simple mode."

        # Detect faces in the frame using face_recognition
        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = fr.face_locations(rgb_frame, model="hog")

        if len(face_locations) == 0:
            return False, "No face detected. Please look at the camera."

        if len(face_locations) > 1:
            return False, "Multiple faces detected. Please ensure only you are visible."

        # Compute embedding
        encodings = fr.face_encodings(rgb_frame, face_locations)
        if not encodings:
            return False, "Could not compute face encoding. Try again."

        self._registration_samples.append(encodings[0])
        count = len(self._registration_samples)
        logger.info("Registration sample %d added.", count)

        return True, f"Sample {count} captured. Need {max(0, self.MIN_REGISTRATION_SAMPLES - count)} more."

    def finish_registration(self) -> tuple[bool, str]:
        """
        Complete registration by averaging the collected embeddings.
        
        Returns:
            (success, message) tuple
        """
        if not self._registering:
            return False, "Registration not started."

        if len(self._registration_samples) < self.MIN_REGISTRATION_SAMPLES:
            if not FACE_RECOGNITION_AVAILABLE and len(self._registration_samples) >= 1:
                pass  # Simple mode allows 1 sample
            else:
                return False, (
                    f"Need at least {self.MIN_REGISTRATION_SAMPLES} samples. "
                    f"Currently have {len(self._registration_samples)}."
                )

        if FACE_RECOGNITION_AVAILABLE and self._registration_samples:
            # Average all embeddings for a robust profile
            self._primary_embedding = np.mean(self._registration_samples, axis=0)
        else:
            self._primary_embedding = np.zeros(128)

        self._is_registered = True
        self._registering = False
        self._save_profile()

        logger.info("Registration complete with %d samples.", len(self._registration_samples))
        return True, f"Registration complete! Used {len(self._registration_samples)} samples."

    # ── Identification ─────────────────────────────────────────────

    def identify_faces(self, frame: np.ndarray,
                       face_bboxes: list[tuple[int, int, int, int]]) -> list[IdentifiedFace]:
        """
        Identify each detected face as PRIMARY_USER or UNKNOWN_PERSON.
        
        Args:
            frame: BGR frame from OpenCV
            face_bboxes: List of (x, y, w, h) bounding boxes from the face detector
            
        Returns:
            List of IdentifiedFace objects
        """
        if not self._is_registered:
            # No primary user registered → all faces are UNREGISTERED
            return [
                IdentifiedFace(
                    bbox=bbox,
                    confidence=0.0,
                    identity=FaceIdentity.UNREGISTERED,
                    similarity=0.0,
                )
                for bbox in face_bboxes
            ]

        if not FACE_RECOGNITION_AVAILABLE or self._primary_embedding is None:
            # Simple mode: first/largest face = primary user, rest = unknown
            return self._identify_simple(face_bboxes)

        return self._identify_with_recognition(frame, face_bboxes)

    def _identify_simple(self, face_bboxes: list[tuple[int, int, int, int]]) -> list[IdentifiedFace]:
        """Simple identification: largest face = primary user."""
        if not face_bboxes:
            return []

        # Sort by area (largest first)
        sorted_bboxes = sorted(face_bboxes, key=lambda b: b[2] * b[3], reverse=True)

        results = []
        for i, bbox in enumerate(sorted_bboxes):
            identity = FaceIdentity.PRIMARY_USER if i == 0 else FaceIdentity.UNKNOWN_PERSON
            results.append(IdentifiedFace(
                bbox=bbox,
                confidence=0.8 if i == 0 else 0.5,
                identity=identity,
                similarity=0.9 if i == 0 else 0.1,
            ))
        return results

    def _identify_with_recognition(self, frame: np.ndarray,
                                    face_bboxes: list[tuple[int, int, int, int]]) -> list[IdentifiedFace]:
        """Use face_recognition library for proper identification."""
        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert (x, y, w, h) to face_recognition format (top, right, bottom, left)
        fr_locations = []
        for (x, y, w, h) in face_bboxes:
            fr_locations.append((y, x + w, y + h, x))

        try:
            encodings = fr.face_encodings(rgb_frame, fr_locations)
        except Exception as e:
            logger.warning("Face encoding failed: %s", e)
            return self._identify_simple(face_bboxes)

        results = []
        primary_found = False

        for bbox, encoding in zip(face_bboxes, encodings):
            # Compute distance to primary user
            distance = fr.face_distance([self._primary_embedding], encoding)[0]
            similarity = max(0.0, 1.0 - distance)
            is_match = distance <= self.RECOGNITION_TOLERANCE

            if is_match and not primary_found:
                identity = FaceIdentity.PRIMARY_USER
                primary_found = True
            else:
                identity = FaceIdentity.UNKNOWN_PERSON

            results.append(IdentifiedFace(
                bbox=bbox,
                confidence=similarity,
                identity=identity,
                similarity=similarity,
            ))

        # Handle faces without encodings
        for bbox in face_bboxes[len(encodings):]:
            results.append(IdentifiedFace(
                bbox=bbox,
                confidence=0.0,
                identity=FaceIdentity.UNKNOWN_PERSON,
                similarity=0.0,
            ))

        return results
