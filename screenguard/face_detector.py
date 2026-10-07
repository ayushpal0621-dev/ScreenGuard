"""
ScreenGuard Face Detector

Uses high-performance face detection (OpenCV YuNet DNN with Haar Cascade & MediaPipe fallbacks).
Returns face locations, confidence scores, and bounding boxes.
"""

import os
import cv2
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Base directory for bundled models
MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass
class DetectedFace:
    """Represents a single detected face in a frame."""
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in pixels
    confidence: float                 # Detection confidence 0-1
    center: tuple[int, int] = field(init=False)  # Center point
    area: int = field(init=False)                 # Bounding box area in pixels

    def __post_init__(self):
        x, y, w, h = self.bbox
        self.center = (x + w // 2, y + h // 2)
        self.area = w * h


@dataclass
class DetectionResult:
    """Result from a single frame's face detection."""
    faces: list[DetectedFace]
    frame_shape: tuple[int, int]  # (height, width)
    timestamp: float = 0.0

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def has_faces(self) -> bool:
        return len(self.faces) > 0


class FaceDetector:
    """
    Real-time face detector supporting OpenCV YuNet DNN and Haar Cascade backends.
    
    Optimized for real-time detection with configurable confidence threshold
    and minimum face size filtering.
    """

    def __init__(self, confidence_threshold: float = 0.5, min_face_size_ratio: float = 0.02):
        """
        Args:
            confidence_threshold: Minimum confidence score to consider a detection valid.
            min_face_size_ratio: Minimum face area as a ratio of frame area.
                                 Filters out tiny/distant faces.
        """
        self._confidence_threshold = confidence_threshold
        self._min_face_size_ratio = min_face_size_ratio
        self._backend: Optional[str] = None  # "yunet", "mediapipe", or "cascade"
        self._detector: Any = None
        self._current_input_size: Optional[tuple[int, int]] = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize face detector using the best available backend."""
        # Backend 1: OpenCV YuNet (ONNX DNN) - modern, accurate, fast
        yunet_path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
        if yunet_path.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    model=str(yunet_path),
                    config="",
                    input_size=(640, 480),
                    score_threshold=self._confidence_threshold,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                self._current_input_size = (640, 480)
                self._backend = "yunet"
                self._initialized = True
                logger.info("Face detector initialized with YuNet DNN backend (confidence=%.2f)",
                            self._confidence_threshold)
                return True
            except Exception as e:
                logger.warning("Failed to initialize YuNet detector: %s. Trying fallback...", e)

        # Backend 2: MediaPipe solutions (if available on system)
        try:
            import mediapipe as mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                self._detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=self._confidence_threshold,
                )
                self._backend = "mediapipe"
                self._initialized = True
                logger.info("Face detector initialized with MediaPipe backend (confidence=%.2f)",
                            self._confidence_threshold)
                return True
        except Exception:
            pass

        # Backend 3: OpenCV Haar Cascade - universal fallback
        cascade_path = MODELS_DIR / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists() and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

        if cascade_path.exists():
            try:
                cascade = cv2.CascadeClassifier(str(cascade_path))
                if not cascade.empty():
                    self._detector = cascade
                    self._backend = "cascade"
                    self._initialized = True
                    logger.info("Face detector initialized with Haar Cascade backend")
                    return True
            except Exception as e:
                logger.error("Failed to load Haar Cascade: %s", e)

        logger.error("No face detection backend could be initialized.")
        return False

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> DetectionResult:
        """
        Detect faces in a BGR frame.
        
        Args:
            frame: BGR image from OpenCV.
            timestamp: Frame timestamp for tracking.
            
        Returns:
            DetectionResult with all detected faces.
        """
        if not self._initialized or self._detector is None or frame is None or frame.size == 0:
            h = frame.shape[0] if frame is not None and frame.ndim >= 2 else 0
            w = frame.shape[1] if frame is not None and frame.ndim >= 2 else 0
            return DetectionResult(faces=[], frame_shape=(h, w), timestamp=timestamp)

        h, w = frame.shape[:2]
        frame_area = h * w
        min_face_area = frame_area * self._min_face_size_ratio
        faces: list[DetectedFace] = []

        if self._backend == "yunet":
            if self._current_input_size != (w, h):
                self._detector.setInputSize((w, h))
                self._current_input_size = (w, h)

            _, raw_detections = self._detector.detect(frame)
            if raw_detections is not None:
                for det in raw_detections:
                    confidence = float(det[-1])
                    if confidence < self._confidence_threshold:
                        continue

                    x = max(0, int(det[0]))
                    y = max(0, int(det[1]))
                    bw = max(0, min(int(det[2]), w - x))
                    bh = max(0, min(int(det[3]), h - y))

                    face = DetectedFace(bbox=(x, y, bw, bh), confidence=confidence)
                    if face.area >= min_face_area:
                        faces.append(face)

        elif self._backend == "mediapipe":
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._detector.process(rgb_frame)
            if results and results.detections:
                for detection in results.detections:
                    bbox_rel = detection.location_data.relative_bounding_box
                    x = max(0, int(bbox_rel.xmin * w))
                    y = max(0, int(bbox_rel.ymin * h))
                    bw = min(int(bbox_rel.width * w), w - x)
                    bh = min(int(bbox_rel.height * h), h - y)
                    conf = detection.score[0] if detection.score else 0.0

                    if conf >= self._confidence_threshold:
                        face = DetectedFace(bbox=(x, y, bw, bh), confidence=conf)
                        if face.area >= min_face_area:
                            faces.append(face)

        elif self._backend == "cascade":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw_faces = self._detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            for (x, y, bw, bh) in raw_faces:
                face = DetectedFace(bbox=(int(x), int(y), int(bw), int(bh)), confidence=0.90)
                if face.area >= min_face_area:
                    faces.append(face)

        return DetectionResult(
            faces=faces,
            frame_shape=(h, w),
            timestamp=timestamp,
        )

    def release(self):
        """Release detector resources."""
        if self._backend == "mediapipe" and self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
        self._detector = None
        self._backend = None
        self._current_input_size = None
        self._initialized = False
        logger.info("Face detector released.")

    @property
    def is_initialized(self) -> bool:
        return self._initialized
