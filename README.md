# ScreenGuard — AI Screen Privacy Protection System

A desktop security application that uses your computer's webcam to detect if another person is looking at your screen, and takes protective action (alerts, screen lock) when a privacy threat is detected.

## Features

- **Real-time face detection** via MediaPipe
- **Primary user recognition** — distinguishes you from others
- **Privacy threat state machine** — with configurable confirmation delays to prevent false positives
- **Desktop alerts** — native OS notifications + sound
- **Automatic screen lock** — with dry-run safety mode
- **Modern PySide6 GUI** — dark theme with glassmorphism
- **System tray** — runs in background
- **Event logging** — SQLite-based, no images stored
- **100% local processing** — no cloud, no recording

## Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test camera access
python main.py --test-cam

# Launch the GUI
python main.py

# Or run in CLI mode with debug window
python main.py --cli --debug
```

## Usage Modes

| Command | Description |
|---------|-------------|
| `python main.py` | Launch the full GUI application |
| `python main.py --cli` | Run in CLI mode (terminal output) |
| `python main.py --cli --debug` | CLI mode + OpenCV debug window |
| `python main.py --test-cam` | Quick camera test |
| `python main.py --dry-run` | Force dry-run mode (simulate lock) |
| `python main.py --camera 1` | Use a specific camera index |
| `python main.py -v` | Verbose/debug logging |

## How It Works

```
Webcam → Frame Capture → Face Detection (MediaPipe)
                              ↓
                     Face Recognition (dlib)
                              ↓
                    Identity Classification
                   ┌──────────┴──────────┐
              PRIMARY_USER         UNKNOWN_PERSON
                   │                     │
                   └──────────┬──────────┘
                              ↓
                    Detection Engine (State Machine)
                              │
          ┌───────────────────┼───────────────────┐
          ↓                   ↓                   ↓
        SAFE            VERIFYING          PRIVACY_THREAT
                              ↓                   ↓
                     (confirmation timer)    Alert + Lock
```

### State Machine

| State | Meaning |
|-------|---------|
| `SAFE` | Only primary user (or nobody) visible |
| `UNKNOWN_PERSON_DETECTED` | Unknown face first appeared |
| `VERIFYING` | Confirming the unknown face persists |
| `PRIVACY_THREAT` | Confirmed threat — taking action |
| `UNATTENDED` | No one visible at the computer |

### Safety Features

- **Dry-run mode** (default): simulates lock instead of actually locking
- **Consecutive-frame confirmation**: requires multiple frames before detecting
- **Configurable delays**: 2s confirmation + 3s lock delay by default
- **Rate limiting**: max 2 locks per minute
- **Cooldown**: 30s between repeated alerts

## Configuration

Settings are stored in `config.json` (auto-generated). Key settings:

```json
{
    "protection_enabled": true,
    "mode": "alert",
    "confirmation_seconds": 2.0,
    "lock_delay_seconds": 3.0,
    "dry_run": true,
    "face_confidence_threshold": 0.5,
    "alert_sound_enabled": true,
    "detection_fps": 15
}
```

## Privacy

- **All processing is local** — no cloud APIs
- **No video recording** by default
- **No image storage** — face recognition uses mathematical embeddings
- **Deletable profile** — remove your face data at any time
- **Event logs** store only timestamps and event types

## Project Structure

```
Privacy/
├── main.py                      # Entry point (GUI/CLI/test modes)
├── requirements.txt             # Python dependencies
├── config.json                  # User configuration (auto-generated)
├── screenguard/
│   ├── __init__.py              # Package metadata
│   ├── config.py                # Configuration manager
│   ├── camera.py                # Webcam capture thread
│   ├── face_detector.py         # MediaPipe face detection
│   ├── face_recognizer.py       # Face registration & identification
│   ├── detection_engine.py      # Privacy threat state machine
│   ├── alert_system.py          # Notifications & sounds
│   ├── security_manager.py      # OS screen lock
│   ├── event_logger.py          # SQLite event logging
│   ├── monitor.py               # Background monitoring thread
│   └── gui.py                   # PySide6 desktop GUI
├── tests/
│   └── test_screenguard.py      # Test suite
└── data/                        # Local data (auto-created)
    ├── face_profiles/           # Stored face embeddings
    └── screenguard_events.db    # Event log database
```

## Running Tests

```bash
source venv/bin/activate
pip install pytest
python -m pytest tests/ -v
```

## License

Private — All rights reserved.
