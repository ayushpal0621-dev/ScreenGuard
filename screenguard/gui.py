"""
ScreenGuard — PySide6 Desktop GUI

A polished, premium desktop application for the ScreenGuard privacy
protection system. Features a dark theme with glassmorphism effects,
smooth animations, and a professional dashboard layout.
"""

import sys
import time
import logging
import numpy as np
import cv2
from typing import Optional
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QStackedWidget, QFrame, QProgressBar, QScrollArea,
    QCheckBox, QTextEdit, QSystemTrayIcon, QMenu, QMessageBox,
    QSplitter, QTabWidget, QSlider, QSizePolicy
)
from PySide6.QtCore import (
    Qt, QTimer, Signal, Slot, QSize, QPropertyAnimation,
    QEasingCurve, Property, QPoint, QParallelAnimationGroup
)
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush,
    QLinearGradient, QRadialGradient, QIcon, QPalette, QAction
)

from screenguard.config import Config
from screenguard.monitor import MonitorThread
from screenguard.face_recognizer import FaceIdentity, IdentifiedFace
from screenguard.detection_engine import PrivacyState

logger = logging.getLogger(__name__)

# ── Color Palette ──────────────────────────────────────────────────────

COLORS = {
    "bg_primary": "#0a0e17",
    "bg_secondary": "#111827",
    "bg_card": "#1a1f2e",
    "bg_card_hover": "#232a3b",
    "bg_input": "#151b2b",
    "border": "#2a3142",
    "border_focus": "#6366f1",

    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",

    "accent": "#6366f1",       # Indigo
    "accent_light": "#818cf8",
    "accent_dark": "#4f46e5",

    "safe": "#10b981",         # Emerald green
    "safe_bg": "#064e3b",
    "warning": "#f59e0b",      # Amber
    "warning_bg": "#78350f",
    "danger": "#ef4444",       # Red
    "danger_bg": "#7f1d1d",

    "gradient_start": "#6366f1",
    "gradient_end": "#a855f7",
}


def get_stylesheet() -> str:
    """Generate the complete application stylesheet."""
    return f"""
    /* ── Global ──────────────────────────────────────────── */
    QMainWindow, QWidget {{
        background-color: {COLORS["bg_primary"]};
        color: {COLORS["text_primary"]};
        font-family: 'SF Pro Display', 'Inter', 'Segoe UI', system-ui, sans-serif;
    }}

    /* ── Cards / Group Boxes ────────────────────────────── */
    QGroupBox {{
        background-color: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 12px;
        padding: 20px;
        padding-top: 35px;
        margin-top: 10px;
        font-size: 13px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 4px 12px;
        color: {COLORS["text_secondary"]};
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    /* ── Buttons ─────────────────────────────────────────── */
    QPushButton {{
        background-color: {COLORS["accent"]};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 13px;
        font-weight: 600;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background-color: {COLORS["accent_light"]};
    }}
    QPushButton:pressed {{
        background-color: {COLORS["accent_dark"]};
    }}
    QPushButton:disabled {{
        background-color: {COLORS["bg_card"]};
        color: {COLORS["text_muted"]};
    }}
    QPushButton#dangerBtn {{
        background-color: {COLORS["danger"]};
    }}
    QPushButton#dangerBtn:hover {{
        background-color: #dc2626;
    }}
    QPushButton#secondaryBtn {{
        background-color: {COLORS["bg_card"]};
        border: 1px solid {COLORS["border"]};
        color: {COLORS["text_secondary"]};
    }}
    QPushButton#secondaryBtn:hover {{
        background-color: {COLORS["bg_card_hover"]};
        border-color: {COLORS["accent"]};
        color: {COLORS["text_primary"]};
    }}

    /* ── Inputs ──────────────────────────────────────────── */
    QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {COLORS["bg_input"]};
        color: {COLORS["text_primary"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 13px;
        min-height: 20px;
    }}
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {COLORS["accent"]};
    }}
    QComboBox::drop-down {{
        border: none;
        padding-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLORS["bg_card"]};
        color: {COLORS["text_primary"]};
        border: 1px solid {COLORS["border"]};
        selection-background-color: {COLORS["accent"]};
    }}

    /* ── Progress Bar ────────────────────────────────────── */
    QProgressBar {{
        background-color: {COLORS["bg_input"]};
        border: none;
        border-radius: 6px;
        min-height: 8px;
        max-height: 8px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {COLORS["gradient_start"]}, stop:1 {COLORS["gradient_end"]});
        border-radius: 6px;
    }}

    /* ── CheckBox ────────────────────────────────────────── */
    QCheckBox {{
        color: {COLORS["text_secondary"]};
        font-size: 13px;
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid {COLORS["border"]};
        background-color: {COLORS["bg_input"]};
    }}
    QCheckBox::indicator:checked {{
        background-color: {COLORS["accent"]};
        border-color: {COLORS["accent"]};
    }}

    /* ── Tabs ────────────────────────────────────────────── */
    QTabWidget::pane {{
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        background-color: {COLORS["bg_secondary"]};
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: {COLORS["bg_card"]};
        color: {COLORS["text_muted"]};
        border: 1px solid {COLORS["border"]};
        border-bottom: none;
        padding: 10px 20px;
        font-size: 13px;
        font-weight: 500;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: {COLORS["bg_secondary"]};
        color: {COLORS["accent_light"]};
        border-bottom: 2px solid {COLORS["accent"]};
    }}
    QTabBar::tab:hover:!selected {{
        color: {COLORS["text_secondary"]};
    }}

    /* ── Scroll Area ─────────────────────────────────────── */
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QScrollBar:vertical {{
        background-color: {COLORS["bg_primary"]};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background-color: {COLORS["border"]};
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {COLORS["text_muted"]};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* ── Text Edit (Log) ─────────────────────────────────── */
    QTextEdit {{
        background-color: {COLORS["bg_input"]};
        color: {COLORS["text_secondary"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 8px;
        font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 12px;
    }}

    /* ── Labels ──────────────────────────────────────────── */
    QLabel {{
        color: {COLORS["text_secondary"]};
        font-size: 13px;
    }}
    QLabel#headerLabel {{
        color: {COLORS["text_primary"]};
        font-size: 28px;
        font-weight: 700;
    }}
    QLabel#subHeaderLabel {{
        color: {COLORS["text_muted"]};
        font-size: 14px;
        font-weight: 400;
    }}
    QLabel#statusSafe {{
        color: {COLORS["safe"]};
        font-size: 15px;
        font-weight: 700;
    }}
    QLabel#statusWarning {{
        color: {COLORS["warning"]};
        font-size: 15px;
        font-weight: 700;
    }}
    QLabel#statusDanger {{
        color: {COLORS["danger"]};
        font-size: 15px;
        font-weight: 700;
    }}
    QLabel#bigNumber {{
        font-size: 36px;
        font-weight: 700;
        color: {COLORS["text_primary"]};
    }}
    QLabel#metricLabel {{
        font-size: 11px;
        color: {COLORS["text_muted"]};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    """


# ── Status Indicator Widget ──────────────────────────────────────────

class StatusIndicator(QWidget):
    """Animated colored dot indicator."""

    def __init__(self, size: int = 12, parent=None):
        super().__init__(parent)
        self._size = size
        self._color = QColor(COLORS["safe"])
        self._pulse = 0.0
        self.setFixedSize(size + 8, size + 8)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(50)

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def _animate(self):
        self._pulse = (self._pulse + 0.05) % (2 * 3.14159)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        import math
        glow_alpha = int(40 + 30 * abs(math.sin(self._pulse)))
        center_x = self.width() // 2
        center_y = self.height() // 2

        # Glow
        glow_color = QColor(self._color)
        glow_color.setAlpha(glow_alpha)
        painter.setBrush(QBrush(glow_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            center_x - self._size // 2 - 3,
            center_y - self._size // 2 - 3,
            self._size + 6,
            self._size + 6,
        )

        # Core dot
        painter.setBrush(QBrush(self._color))
        painter.drawEllipse(
            center_x - self._size // 2,
            center_y - self._size // 2,
            self._size,
            self._size,
        )
        painter.end()


# ── Camera Feed Widget ───────────────────────────────────────────────

class CameraFeedWidget(QLabel):
    """Displays the camera feed with face annotations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet(f"""
            background-color: {COLORS["bg_input"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
        """)
        self._show_placeholder()

    def _show_placeholder(self):
        self.setText("📷  Camera feed will appear here")
        self.setStyleSheet(f"""
            background-color: {COLORS["bg_input"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
            color: {COLORS["text_muted"]};
            font-size: 14px;
        """)

    def update_frame(self, frame: np.ndarray, faces: list):
        """Update displayed frame with face annotations."""
        display = frame.copy()

        for face in faces:
            x, y, w, h = face.bbox
            identity = face.identity

            # Color by identity
            if identity == FaceIdentity.PRIMARY_USER:
                color = (16, 185, 129)  # Safe green (BGR)
                label = "You"
            elif identity == FaceIdentity.UNKNOWN_PERSON:
                color = (68, 68, 239)   # Danger red (BGR)
                label = "Unknown"
            else:
                color = (139, 163, 148) # Muted (BGR)
                label = "Unregistered"

            # Draw rounded rect border
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

            # Label background
            label_h = 22
            cv2.rectangle(display, (x, y - label_h), (x + len(label) * 10 + 16, y), color, -1)
            cv2.putText(display, label, (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Confidence
            conf_text = f"{face.confidence:.0%}"
            cv2.putText(display, conf_text, (x + w - 35, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Convert BGR → RGB → QImage → QPixmap
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to widget size maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


# ── Metric Card Widget ───────────────────────────────────────────────

class MetricCard(QFrame):
    """Small card displaying a single metric with label."""

    def __init__(self, label: str, value: str = "—", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS["bg_card"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 10px;
                padding: 12px;
            }}
            QFrame:hover {{
                border-color: {COLORS["accent"]};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        self._value_label = QLabel(value)
        self._value_label.setObjectName("bigNumber")
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._metric_label = QLabel(label.upper())
        self._metric_label.setObjectName("metricLabel")
        self._metric_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._value_label)
        layout.addWidget(self._metric_label)

    def set_value(self, value: str):
        self._value_label.setText(value)

    def set_color(self, color: str):
        self._value_label.setStyleSheet(f"color: {color}; font-size: 36px; font-weight: 700;")


# ── Alert Banner Widget ──────────────────────────────────────────────

class AlertBanner(QFrame):
    """Animated alert banner that slides in from the top."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(0)
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS["danger_bg"]}, stop:1 #991b1b);
                border-bottom: 2px solid {COLORS["danger"]};
                padding: 0px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 8, 20, 8)

        self._icon = QLabel("⚠")
        self._icon.setStyleSheet("font-size: 20px; color: #fbbf24;")
        self._title = QLabel("")
        self._title.setStyleSheet(f"color: {COLORS['danger']}; font-weight: 700; font-size: 14px;")
        self._message = QLabel("")
        self._message.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px;")
        self._dismiss_btn = QPushButton("✕")
        self._dismiss_btn.setFixedSize(28, 28)
        self._dismiss_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {COLORS["text_muted"]};
                font-size: 16px;
                border: none;
                border-radius: 14px;
            }}
            QPushButton:hover {{ color: white; background: rgba(255,255,255,0.1); }}
        """)
        self._dismiss_btn.clicked.connect(self.hide_alert)

        layout.addWidget(self._icon)
        layout.addWidget(self._title)
        layout.addWidget(self._message, 1)
        layout.addWidget(self._dismiss_btn)

        self._visible = False

    def show_alert(self, title: str, message: str):
        if self._visible:
            return
        self._title.setText(title)
        self._message.setText(message)
        self._visible = True

        self._anim = QPropertyAnimation(self, b"maximumHeight")
        self._anim.setDuration(300)
        self._anim.setStartValue(0)
        self._anim.setEndValue(50)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()

        # Auto-dismiss after 10 seconds
        QTimer.singleShot(10000, self.hide_alert)

    def hide_alert(self):
        if not self._visible:
            return
        self._visible = False
        self._anim = QPropertyAnimation(self, b"maximumHeight")
        self._anim.setDuration(200)
        self._anim.setStartValue(50)
        self._anim.setEndValue(0)
        self._anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._anim.start()


# ── Main Window ──────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """ScreenGuard main application window."""

    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        self._monitor: Optional[MonitorThread] = None
        self._is_protecting = False
        self._current_frame: Optional[np.ndarray] = None

        self.setWindowTitle("ScreenGuard — AI Screen Privacy Protection")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        self.setStyleSheet(get_stylesheet())

        self._setup_ui()
        self._setup_tray()
        self._setup_status_timer()

    def _setup_ui(self):
        """Build the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Alert banner (initially hidden)
        self._alert_banner = AlertBanner()
        main_layout.addWidget(self._alert_banner)

        # Content area with padding
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 16, 24, 16)
        content_layout.setSpacing(16)

        # Header
        header = self._build_header()
        content_layout.addLayout(header)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_dashboard_tab(), "🛡️  Dashboard")
        self._tabs.addTab(self._build_settings_tab(), "⚙️  Settings")
        self._tabs.addTab(self._build_log_tab(), "📋  Event Log")
        self._tabs.addTab(self._build_privacy_tab(), "🔒  Privacy")
        content_layout.addWidget(self._tabs, 1)

        main_layout.addWidget(content, 1)

    def _build_header(self) -> QHBoxLayout:
        """Build the top header bar."""
        layout = QHBoxLayout()

        # Left side: title
        left = QVBoxLayout()
        title = QLabel("ScreenGuard")
        title.setObjectName("headerLabel")
        subtitle = QLabel("AI-Powered Screen Privacy Protection")
        subtitle.setObjectName("subHeaderLabel")
        left.addWidget(title)
        left.addWidget(subtitle)
        left.setSpacing(2)

        # Right side: status indicator + main action button
        right = QHBoxLayout()
        right.setSpacing(12)

        self._status_indicator = StatusIndicator(size=12)
        self._status_label = QLabel("Protection Inactive")
        self._status_label.setObjectName("statusSafe")

        self._main_action_btn = QPushButton("▶  Start Protection")
        self._main_action_btn.setFixedWidth(180)
        self._main_action_btn.clicked.connect(self._toggle_protection)

        right.addStretch()
        right.addWidget(self._status_indicator)
        right.addWidget(self._status_label)
        right.addWidget(self._main_action_btn)

        layout.addLayout(left)
        layout.addLayout(right)
        return layout

    def _build_dashboard_tab(self) -> QWidget:
        """Build the main monitoring dashboard."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(16)

        # Left column: camera + controls
        left = QVBoxLayout()
        left.setSpacing(12)

        # Camera feed
        self._camera_feed = CameraFeedWidget()
        self._camera_feed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left.addWidget(self._camera_feed, 3)

        # Registration controls
        reg_group = QGroupBox("User Registration")
        reg_layout = QHBoxLayout(reg_group)
        self._reg_status_label = QLabel("Not registered")
        self._reg_status_label.setStyleSheet(f"color: {COLORS['warning']};")
        self._register_btn = QPushButton("Register Face")
        self._register_btn.setObjectName("secondaryBtn")
        self._register_btn.clicked.connect(self._start_registration)
        self._capture_btn = QPushButton("📸 Capture Sample")
        self._capture_btn.setObjectName("secondaryBtn")
        self._capture_btn.setVisible(False)
        self._capture_btn.clicked.connect(self._capture_registration_sample)
        self._finish_reg_btn = QPushButton("✓ Finish")
        self._finish_reg_btn.setVisible(False)
        self._finish_reg_btn.clicked.connect(self._finish_registration)

        reg_layout.addWidget(self._reg_status_label, 1)
        reg_layout.addWidget(self._register_btn)
        reg_layout.addWidget(self._capture_btn)
        reg_layout.addWidget(self._finish_reg_btn)
        left.addWidget(reg_group)

        # Right column: metrics + status
        right = QVBoxLayout()
        right.setSpacing(12)

        # Metrics row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(8)
        self._faces_card = MetricCard("Faces Detected", "0")
        self._unknown_card = MetricCard("Unknown People", "0")
        self._fps_card = MetricCard("FPS", "—")
        self._alerts_card = MetricCard("Total Alerts", "0")
        metrics_layout.addWidget(self._faces_card)
        metrics_layout.addWidget(self._unknown_card)
        metrics_layout.addWidget(self._fps_card)
        metrics_layout.addWidget(self._alerts_card)
        right.addLayout(metrics_layout)

        # Status card
        status_group = QGroupBox("Protection Status")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(12)

        # State display
        state_row = QHBoxLayout()
        state_label = QLabel("Current State:")
        self._state_value = QLabel("INACTIVE")
        self._state_value.setObjectName("statusSafe")
        self._state_value.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 15px; font-weight: 700;")
        state_row.addWidget(state_label)
        state_row.addStretch()
        state_row.addWidget(self._state_value)
        status_layout.addLayout(state_row)

        # Primary user row
        user_row = QHBoxLayout()
        user_label = QLabel("Primary User:")
        self._user_value = QLabel("—")
        user_row.addWidget(user_label)
        user_row.addStretch()
        user_row.addWidget(self._user_value)
        status_layout.addLayout(user_row)

        # Camera row
        cam_row = QHBoxLayout()
        cam_label = QLabel("Camera:")
        self._cam_value = QLabel("Disconnected")
        cam_row.addWidget(cam_label)
        cam_row.addStretch()
        cam_row.addWidget(self._cam_value)
        status_layout.addLayout(cam_row)

        # Verification progress
        verify_label = QLabel("Verification Progress")
        verify_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self._verify_progress = QProgressBar()
        self._verify_progress.setRange(0, 100)
        self._verify_progress.setValue(0)
        self._verify_progress.setTextVisible(False)
        status_layout.addWidget(verify_label)
        status_layout.addWidget(self._verify_progress)

        # Threat level
        threat_label = QLabel("Threat Level")
        threat_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self._threat_progress = QProgressBar()
        self._threat_progress.setRange(0, 100)
        self._threat_progress.setValue(0)
        self._threat_progress.setTextVisible(False)
        status_layout.addWidget(threat_label)
        status_layout.addWidget(self._threat_progress)

        right.addWidget(status_group)

        # Mode selection
        mode_group = QGroupBox("Protection Mode")
        mode_layout = QVBoxLayout(mode_group)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Alert Only", "Lock Only", "Alert + Lock"])
        mode_map = {"alert": 0, "lock": 1, "alert_and_lock": 2}
        self._mode_combo.setCurrentIndex(mode_map.get(self._config["mode"], 2))
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo)

        # Quick dry run toggle
        self._quick_dry_run_check = QCheckBox("Dry Run (Simulate Lock)")
        self._quick_dry_run_check.setChecked(self._config["dry_run"])
        self._quick_dry_run_check.toggled.connect(self._on_quick_dry_run_toggled)
        mode_layout.addWidget(self._quick_dry_run_check)

        # Status badge for lock behavior
        self._dry_run_label = QLabel()
        self._update_dry_run_badge(self._config["dry_run"])
        mode_layout.addWidget(self._dry_run_label)

        right.addWidget(mode_group)
        right.addStretch()

        # Assemble columns
        layout.addLayout(left, 3)
        layout.addLayout(right, 2)
        return widget

    def _build_settings_tab(self) -> QWidget:
        """Build the settings configuration panel."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)
        layout.setContentsMargins(8, 8, 8, 8)

        # Detection settings
        det_group = QGroupBox("Detection Settings")
        det_layout = QVBoxLayout(det_group)

        # Confirmation duration
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Confirmation Duration (seconds):"))
        self._confirm_spin = QDoubleSpinBox()
        self._confirm_spin.setRange(0.5, 10.0)
        self._confirm_spin.setSingleStep(0.5)
        self._confirm_spin.setValue(self._config["confirmation_seconds"])
        row1.addWidget(self._confirm_spin)
        det_layout.addLayout(row1)

        # Lock delay
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Lock Delay (seconds):"))
        self._lock_delay_spin = QDoubleSpinBox()
        self._lock_delay_spin.setRange(1.0, 15.0)
        self._lock_delay_spin.setSingleStep(0.5)
        self._lock_delay_spin.setValue(self._config["lock_delay_seconds"])
        row2.addWidget(self._lock_delay_spin)
        det_layout.addLayout(row2)

        # Confidence
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Face Confidence Threshold:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.1, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(self._config["face_confidence_threshold"])
        row3.addWidget(self._conf_spin)
        det_layout.addLayout(row3)

        # Consecutive frames
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Consecutive Frames Required:"))
        self._consec_spin = QSpinBox()
        self._consec_spin.setRange(1, 30)
        self._consec_spin.setValue(self._config["consecutive_frames_required"])
        row4.addWidget(self._consec_spin)
        det_layout.addLayout(row4)

        # Detection FPS
        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Detection FPS:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 30)
        self._fps_spin.setValue(self._config["detection_fps"])
        row5.addWidget(self._fps_spin)
        det_layout.addLayout(row5)

        layout.addWidget(det_group)

        # Alert settings
        alert_group = QGroupBox("Alert Settings")
        alert_layout = QVBoxLayout(alert_group)

        self._sound_check = QCheckBox("Enable alert sound")
        self._sound_check.setChecked(self._config["alert_sound_enabled"])
        alert_layout.addWidget(self._sound_check)

        self._notif_check = QCheckBox("Enable desktop notifications")
        self._notif_check.setChecked(self._config["desktop_notification_enabled"])
        alert_layout.addWidget(self._notif_check)

        self._visual_check = QCheckBox("Enable visual warning")
        self._visual_check.setChecked(self._config["visual_warning_enabled"])
        alert_layout.addWidget(self._visual_check)

        row_cool = QHBoxLayout()
        row_cool.addWidget(QLabel("Alert Cooldown (seconds):"))
        self._cooldown_spin = QDoubleSpinBox()
        self._cooldown_spin.setRange(5.0, 120.0)
        self._cooldown_spin.setSingleStep(5.0)
        self._cooldown_spin.setValue(self._config["alert_cooldown_seconds"])
        row_cool.addWidget(self._cooldown_spin)
        alert_layout.addLayout(row_cool)

        layout.addWidget(alert_group)

        # Security settings
        sec_group = QGroupBox("Security Settings")
        sec_layout = QVBoxLayout(sec_group)

        self._dry_run_check = QCheckBox("Dry Run Mode (simulate lock instead of locking)")
        self._dry_run_check.setChecked(self._config["dry_run"])
        sec_layout.addWidget(self._dry_run_check)

        layout.addWidget(sec_group)

        # Camera settings
        cam_group = QGroupBox("Camera Settings")
        cam_layout = QVBoxLayout(cam_group)

        row_cam = QHBoxLayout()
        row_cam.addWidget(QLabel("Camera Index:"))
        self._cam_spin = QSpinBox()
        self._cam_spin.setRange(0, 10)
        self._cam_spin.setValue(self._config["camera_index"])
        row_cam.addWidget(self._cam_spin)
        cam_layout.addLayout(row_cam)

        layout.addWidget(cam_group)

        # Save button
        save_btn = QPushButton("💾  Save Settings")
        save_btn.clicked.connect(self._save_settings)
        layout.addWidget(save_btn)

        layout.addStretch()
        scroll.setWidget(widget)
        return scroll

    def _build_log_tab(self) -> QWidget:
        """Build the event log viewer."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        # Controls
        controls = QHBoxLayout()
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setObjectName("secondaryBtn")
        refresh_btn.clicked.connect(self._refresh_log)
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.setObjectName("dangerBtn")
        clear_btn.clicked.connect(self._clear_log)
        controls.addStretch()
        controls.addWidget(refresh_btn)
        controls.addWidget(clear_btn)
        layout.addLayout(controls)

        # Log display
        self._log_display = QTextEdit()
        self._log_display.setReadOnly(True)
        self._log_display.setPlaceholderText("Events will appear here when protection is active...")
        layout.addWidget(self._log_display)

        return widget

    def _build_privacy_tab(self) -> QWidget:
        """Build the privacy information page."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(16)
        layout.setContentsMargins(8, 8, 8, 8)

        title = QLabel("🔒 Privacy & Data Protection")
        title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 20px; font-weight: 700;")
        layout.addWidget(title)

        # Privacy info
        info_group = QGroupBox("What ScreenGuard Does")
        info_layout = QVBoxLayout(info_group)
        info_text = QLabel(
            "• All face detection and recognition is performed <b>locally on your device</b>.\n\n"
            "• <b>No images or video</b> are sent to any cloud service.\n\n"
            "• <b>No webcam footage is recorded</b> unless you explicitly enable it.\n\n"
            "• Face recognition uses a mathematical embedding (a list of numbers), "
            "not your actual photos.\n\n"
            "• The face embedding is stored locally and can be deleted at any time.\n\n"
            "• Event logs record only timestamps and event types — no images."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; line-height: 1.6;")
        info_layout.addWidget(info_text)
        layout.addWidget(info_group)

        # Data management
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout(data_group)

        delete_face_btn = QPushButton("🗑️ Delete Face Profile")
        delete_face_btn.setObjectName("dangerBtn")
        delete_face_btn.clicked.connect(self._delete_face_profile)
        data_layout.addWidget(delete_face_btn)

        clear_events_btn = QPushButton("🗑️ Clear Event History")
        clear_events_btn.setObjectName("dangerBtn")
        clear_events_btn.clicked.connect(self._clear_log)
        data_layout.addWidget(clear_events_btn)

        layout.addWidget(data_group)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _setup_tray(self):
        """Set up system tray icon and menu."""
        self._tray = QSystemTrayIcon(self)
        # Use a simple icon approach
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(COLORS["accent"]))
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "S")
        painter.end()
        self._tray.setIcon(QIcon(pixmap))
        self._tray.setToolTip("ScreenGuard — Privacy Protection")

        tray_menu = QMenu()
        show_action = QAction("Show Window", self)
        show_action.triggered.connect(self.show)
        toggle_action = QAction("Toggle Protection", self)
        toggle_action.triggered.connect(self._toggle_protection)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self._quit_app)

        tray_menu.addAction(show_action)
        tray_menu.addAction(toggle_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)

        self._tray.setContextMenu(tray_menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _setup_status_timer(self):
        """Timer to periodically update UI elements."""
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._update_ui_periodic)
        self._ui_timer.start(1000)  # Every second

    # ── Actions ──────────────────────────────────────────────────

    def _toggle_protection(self):
        """Start or stop privacy protection."""
        if self._is_protecting:
            self._stop_protection()
        else:
            self._start_protection()

    def _start_protection(self):
        """Initialize and start the monitoring thread."""
        if self._monitor and self._monitor.isRunning():
            return

        self._monitor = MonitorThread(self._config)
        if not self._monitor.initialize():
            QMessageBox.critical(self, "Error", "Failed to initialize monitoring system.")
            return

        # Connect signals
        self._monitor.frame_ready.connect(self._on_frame_ready)
        self._monitor.status_updated.connect(self._on_status_updated)
        self._monitor.alert_triggered.connect(self._on_alert_triggered)
        self._monitor.state_changed.connect(self._on_state_changed)
        self._monitor.error_occurred.connect(self._on_error)

        # Update registration status
        if self._monitor.recognizer and self._monitor.recognizer.is_registered:
            self._reg_status_label.setText("✓ Face registered")
            self._reg_status_label.setStyleSheet(f"color: {COLORS['safe']};")

        self._monitor.start()
        self._is_protecting = True

        self._main_action_btn.setText("⏹  Stop Protection")
        self._main_action_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS["danger"]};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background-color: #dc2626; }}
        """)
        self._status_label.setText("Protection Active")
        self._status_label.setObjectName("statusSafe")
        self._status_indicator.set_color(COLORS["safe"])

        logger.info("Protection started from GUI.")

    def _stop_protection(self):
        """Stop the monitoring thread cleanly."""
        if self._monitor:
            self._monitor.stop_monitoring()
            if not self._monitor.wait(3000):
                self._monitor.terminate()
                self._monitor.wait(1000)
            self._monitor = None

        self._is_protecting = False

        self._main_action_btn.setText("▶  Start Protection")
        self._main_action_btn.setStyleSheet("")  # Reset to default
        self._status_label.setText("Protection Inactive")
        self._status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 15px; font-weight: 700;")
        self._status_indicator.set_color(COLORS["text_muted"])
        self._state_value.setText("INACTIVE")
        self._verify_progress.setValue(0)
        self._threat_progress.setValue(0)

        logger.info("Protection stopped from GUI.")

    def _update_dry_run_badge(self, is_dry_run: bool):
        """Update visual badge describing lock behavior."""
        if is_dry_run:
            self._dry_run_label.setText("🧪 DRY RUN ACTIVE — Lock is simulated")
            self._dry_run_label.setStyleSheet(f"""
                color: {COLORS['warning']};
                background-color: {COLORS['warning_bg']};
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 11px;
                font-weight: 600;
            """)
        else:
            self._dry_run_label.setText("🔒 LIVE PROTECTION — Screen will lock on threat")
            self._dry_run_label.setStyleSheet(f"""
                color: {COLORS['safe']};
                background-color: {COLORS['safe_bg']};
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 11px;
                font-weight: 600;
            """)

    def _on_quick_dry_run_toggled(self, checked: bool):
        """Handle toggling dry run mode from the main dashboard."""
        self._config["dry_run"] = checked
        self._config.save()
        self._update_dry_run_badge(checked)

        if hasattr(self, "_dry_run_check"):
            self._dry_run_check.blockSignals(True)
            self._dry_run_check.setChecked(checked)
            self._dry_run_check.blockSignals(False)

        if self._monitor and self._monitor.security_manager:
            self._monitor.security_manager.dry_run = checked

        logger.info("Dry run mode set to: %s", checked)

    def _start_registration(self):
        """Enter registration mode."""
        if not self._is_protecting:
            QMessageBox.information(
                self, "Start Protection First",
                "Please start protection first so the camera is active, then register your face."
            )
            return

        if self._monitor:
            self._monitor.start_registration()
            self._register_btn.setVisible(False)
            self._capture_btn.setVisible(True)
            self._finish_reg_btn.setVisible(True)
            self._reg_status_label.setText("Look at the camera and capture samples")
            self._reg_status_label.setStyleSheet(f"color: {COLORS['accent_light']};")

    def _capture_registration_sample(self):
        """Capture a registration sample from current frame."""
        if self._monitor and self._current_frame is not None:
            success, message = self._monitor.add_registration_sample(self._current_frame)
            if success:
                count = self._monitor.recognizer.registration_sample_count if self._monitor.recognizer else 0
                self._reg_status_label.setText(f"✓ Sample {count} captured — {message}")
                self._reg_status_label.setStyleSheet(f"color: {COLORS['safe']};")
            else:
                self._reg_status_label.setText(f"✗ {message}")
                self._reg_status_label.setStyleSheet(f"color: {COLORS['danger']};")

    def _finish_registration(self):
        """Complete registration."""
        if self._monitor:
            success, message = self._monitor.finish_registration()
            self._register_btn.setVisible(True)
            self._capture_btn.setVisible(False)
            self._finish_reg_btn.setVisible(False)

            if success:
                self._reg_status_label.setText(f"✓ {message}")
                self._reg_status_label.setStyleSheet(f"color: {COLORS['safe']};")
                self._register_btn.setText("Re-register Face")
            else:
                self._reg_status_label.setText(f"✗ {message}")
                self._reg_status_label.setStyleSheet(f"color: {COLORS['danger']};")

    def _save_settings(self):
        """Save current settings from the UI."""
        self._config["confirmation_seconds"] = self._confirm_spin.value()
        self._config["lock_delay_seconds"] = self._lock_delay_spin.value()
        self._config["face_confidence_threshold"] = self._conf_spin.value()
        self._config["consecutive_frames_required"] = self._consec_spin.value()
        self._config["detection_fps"] = self._fps_spin.value()
        self._config["alert_sound_enabled"] = self._sound_check.isChecked()
        self._config["desktop_notification_enabled"] = self._notif_check.isChecked()
        self._config["visual_warning_enabled"] = self._visual_check.isChecked()
        self._config["alert_cooldown_seconds"] = self._cooldown_spin.value()
        self._config["dry_run"] = self._dry_run_check.isChecked()
        self._config["camera_index"] = self._cam_spin.value()

        self._config.save()

        if hasattr(self, "_quick_dry_run_check"):
            self._quick_dry_run_check.blockSignals(True)
            self._quick_dry_run_check.setChecked(self._config["dry_run"])
            self._quick_dry_run_check.blockSignals(False)
        self._update_dry_run_badge(self._config["dry_run"])

        if self._monitor and self._monitor.security_manager:
            self._monitor.security_manager.dry_run = self._config["dry_run"]

        QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully.")

    def _on_mode_changed(self, index: int):
        modes = ["alert", "lock", "alert_and_lock"]
        if 0 <= index < len(modes):
            self._config["mode"] = modes[index]
            self._config.save()
            logger.info("Protection mode changed to: %s", modes[index])

    def _delete_face_profile(self):
        """Delete the stored face profile."""
        reply = QMessageBox.question(
            self, "Delete Face Profile",
            "Are you sure you want to delete your face profile?\n"
            "You will need to re-register your face.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self._monitor and self._monitor.recognizer:
                self._monitor.recognizer.delete_profile()
            self._reg_status_label.setText("Not registered")
            self._reg_status_label.setStyleSheet(f"color: {COLORS['warning']};")
            self._register_btn.setText("Register Face")

    def _refresh_log(self):
        """Refresh the event log display."""
        if self._monitor and self._monitor.event_logger:
            events = self._monitor.event_logger.get_recent_events(100)
            lines = []
            for event in events:
                ts = datetime.fromtimestamp(event["timestamp"]).strftime("%H:%M:%S")
                etype = event["event_type"]
                faces = event.get("face_count", 0)
                action = event.get("action", "")
                details = event.get("details", "")
                line = f"{ts} │ {etype:<22} │ {faces} faces │ {action or '—'}"
                if details:
                    line += f" │ {details}"
                lines.append(line)
            self._log_display.setPlainText("\n".join(lines) if lines else "No events recorded.")
        else:
            self._log_display.setPlainText("Event logging not active.")

    def _clear_log(self):
        """Clear all event logs."""
        reply = QMessageBox.question(
            self, "Clear Events",
            "Are you sure you want to delete all event history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self._monitor and self._monitor.event_logger:
                self._monitor.event_logger.clear_all()
            self._log_display.clear()

    # ── Signal Handlers ──────────────────────────────────────────

    @Slot(np.ndarray, list)
    def _on_frame_ready(self, frame: np.ndarray, faces: list):
        self._current_frame = frame
        self._camera_feed.update_frame(frame, faces)

    @Slot(dict)
    def _on_status_updated(self, status: dict):
        state = status.get("state", "UNKNOWN")
        primary = status.get("primary_user", False)
        unknown_count = status.get("unknown_count", 0)
        total_faces = status.get("total_faces", 0)
        threat_level = status.get("threat_level", 0.0)
        verify_progress = status.get("verification_progress", 0.0)
        fps = status.get("fps", 0)
        cam = status.get("camera_connected", False)

        # Update metric cards
        self._faces_card.set_value(str(total_faces))
        self._unknown_card.set_value(str(unknown_count))
        self._fps_card.set_value(f"{fps:.0f}")

        if unknown_count > 0:
            self._unknown_card.set_color(COLORS["danger"])
        else:
            self._unknown_card.set_color(COLORS["text_primary"])

        # Update state display
        self._state_value.setText(state)
        if state == "SAFE":
            self._state_value.setStyleSheet(f"color: {COLORS['safe']}; font-size: 15px; font-weight: 700;")
            self._status_indicator.set_color(COLORS["safe"])
        elif state in ("UNKNOWN_PERSON_DETECTED", "VERIFYING"):
            self._state_value.setStyleSheet(f"color: {COLORS['warning']}; font-size: 15px; font-weight: 700;")
            self._status_indicator.set_color(COLORS["warning"])
        elif state == "PRIVACY_THREAT":
            self._state_value.setStyleSheet(f"color: {COLORS['danger']}; font-size: 15px; font-weight: 700;")
            self._status_indicator.set_color(COLORS["danger"])
        else:
            self._state_value.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 15px; font-weight: 700;")

        # User status
        self._user_value.setText("✓ Detected" if primary else "Not detected")
        self._user_value.setStyleSheet(
            f"color: {COLORS['safe']};" if primary else f"color: {COLORS['text_muted']};"
        )

        # Camera status
        self._cam_value.setText("✓ Connected" if cam else "✗ Disconnected")
        self._cam_value.setStyleSheet(
            f"color: {COLORS['safe']};" if cam else f"color: {COLORS['danger']};"
        )

        # Progress bars
        self._verify_progress.setValue(int(verify_progress * 100))
        self._threat_progress.setValue(int(threat_level * 100))

        # Update threat progress bar color
        if threat_level > 0.7:
            self._threat_progress.setStyleSheet(f"""
                QProgressBar {{ background-color: {COLORS["bg_input"]}; border: none; border-radius: 6px; min-height: 8px; max-height: 8px; }}
                QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ef4444, stop:1 #dc2626); border-radius: 6px; }}
            """)
        elif threat_level > 0.3:
            self._threat_progress.setStyleSheet(f"""
                QProgressBar {{ background-color: {COLORS["bg_input"]}; border: none; border-radius: 6px; min-height: 8px; max-height: 8px; }}
                QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f59e0b, stop:1 #d97706); border-radius: 6px; }}
            """)
        else:
            self._threat_progress.setStyleSheet("")  # Default style

    @Slot(str, str)
    def _on_alert_triggered(self, title: str, message: str):
        self._alert_banner.show_alert(title, message)
        if self._monitor and self._monitor.event_logger:
            self._alerts_card.set_value(str(self._monitor.event_logger.get_event_stats().get("alerts_triggered", 0)))

    @Slot(str, str)
    def _on_state_changed(self, old_state: str, new_state: str):
        logger.info("GUI state change: %s → %s", old_state, new_state)

    @Slot(str)
    def _on_error(self, message: str):
        logger.error("Monitor error: %s", message)
        QMessageBox.warning(self, "ScreenGuard Error", message)

    def _update_ui_periodic(self):
        """Periodic UI updates (less frequent than frame updates)."""
        pass  # Could add uptime, stats refresh, etc.

    # ── System Tray ──────────────────────────────────────────────

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()

    def closeEvent(self, event):
        """Override close to minimize to tray if configured."""
        if self._config["minimize_to_tray"] and self._is_protecting:
            event.ignore()
            self.hide()
            self._tray.showMessage(
                "ScreenGuard",
                "Privacy protection continues in the background.",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
        else:
            self._quit_app()
            event.accept()

    def _quit_app(self):
        """Clean shutdown."""
        self._stop_protection()
        if self._monitor:
            self._monitor.stop_monitoring()
            if not self._monitor.wait(2000):
                self._monitor.terminate()
                self._monitor.wait(500)
            self._monitor = None
        self._tray.hide()
        QApplication.quit()
