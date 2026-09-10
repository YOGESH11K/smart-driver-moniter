# Smart Driver Monitoring & Drowsiness Detection System

A real-time computer-vision driver monitoring system that detects drowsiness and distraction using **multiple independent signals** rather than relying on a single metric.

> **IMPORTANT DISCLAIMER:** This project is a **student/research prototype**. It is **NOT a certified automotive safety system**, does not meet automotive safety standards, and must not be relied upon to prevent accidents in a real vehicle. Always drive with full attention and obey traffic laws.

---

## Features

- **Face Detection** - 478 facial landmarks via MediaPipe Face Landmarker
- **Eye Aspect Ratio (EAR)** - real-time blinking & prolonged eye closure detection
- **Mouth Aspect Ratio (MAR)** - yawn detection with a temporal counter
- **Head Pose Estimation** - pitch/yaw/roll via OpenCV `solvePnP`; classifies FORWARD / LEFT / RIGHT / UP / DOWN
- **Hand-Down Detection** - MediaPipe Hand Landmarker with a configurable safe region and temporal debounce (missing hand is never treated as "down")
- **Mobile Phone Detection** - optional YOLOv8n (COCO class 67) running on a **background thread** so it never blocks the video loop
- **Risk Engine** - weighted multi-signal scoring with a 5-state state machine and hysteresis
- **Alert System** - visual + optional sound alerts with cooldown/debounce
- **Real-Time Alarm** - a dedicated `AlarmManager` plays a loud, non-blocking audible alarm (background thread) for eye closure, "eyes off road", yawning, and phone use; merges simultaneous conditions into one sound and auto-stops when the danger clears
- **Modern GUI** - CustomTkinter dashboard with live webcam, landmark overlays, metrics, and session stats
- **Session Statistics + SQLite** - persistent session/event storage and an end-of-session summary
- **Calibration** - a short wizard that adapts EAR/MAR thresholds to the user
- **Centralized YAML Configuration** - every threshold is adjustable without touching code

---

## Architecture

```
Webcam ──▶ FaceLandmarker ──▶ EAR / MAR / HeadPose (one inference)
   │
   ├──▶ HandLandmarker ──▶ hand position/safe-region check
   │
   └──▶ YOLOv8n (thread) ──▶ phone detection (cached result)
                 │
                 ▼
            Risk Engine ──▶ State + Safety Score ──▶ Alerts
                        │
                        ▼
                  GUI Dashboard  +  SQLite
```

```
smart-driver-monitor/
├── main.py                    # Entry point
├── setup_models.py            # Downloads .task / .pt model files
├── requirements.txt
├── config/
│   └── config.yaml            # All thresholds & settings
├── models/                    # face_landmarker.task, hand_landmarker.task, yolov8n.pt
├── assets/sounds/             # alert_*.wav (place your files here)
├── src/
│   ├── app.py                 # Orchestrates all components
│   ├── camera/webcam.py       # Threaded webcam capture
│   ├── detection/
│   │   ├── face_detector.py   # MediaPipe FaceLandmarker (Tasks API)
│   │   ├── eye_analyzer.py    # EAR + blink + closure
│   │   ├── mouth_analyzer.py  # MAR + yawn counting
│   │   ├── head_pose.py       # solvePnP pitch/yaw/roll + direction
│   │   ├── hand_detector.py   # MediaPipe HandLandmarker + safe region
│   │   └── phone_detector.py  # YOLOv8n (background thread)
│   ├── monitoring/
│   │   ├── risk_engine.py     # Weighted scoring + state machine
│   │   ├── alert_manager.py   # Cooldown + severity + sound
│   │   └── alarm_manager.py   # Real-time audible alarm (non-blocking)
│   ├── gui/                   # CustomTkinter dashboard
│   ├── storage/               # SQLite session/event storage
│   └── utils/                 # config loader, math helpers, logging
└── tests/                     # Unit tests
```

---

## Technologies

| Component | Library |
|---|---|
| Face/Hand landmarks | `mediapipe` (Tasks API, `mp.tasks.*`) |
| Frame handling / geometry | `opencv-python`, `numpy` |
| Phone detection | `ultralytics` (YOLOv8n, COCO class 67) |
| GUI | `customtkinter` + `pillow` |
| Config | `pyyaml` |
| Sound | `pygame` |
| Storage | SQLite (stdlib `sqlite3`) |
| Python | 3.10+ (tested on 3.14) |

---

## Installation

### 1. Requirements

- Python 3.10 or newer
- A webcam
- ~15 MB free disk for models

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download detection models

```bash
python setup_models.py
```

This downloads:
- `models/face_landmarker.task` (~3.6 MB)
- `models/hand_landmarker.task` (~7.5 MB)
- `models/yolov8n.pt` (~6.2 MB, via Ultralytics)

If a download fails (network restrictions, etc.), fetch the `.task` files manually:

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 4. (Optional) Sound alerts

Place `alert_low.wav`, `alert_medium.wav`, `alert_high.wav` in `assets/sounds/`.
If no files are present, the app falls back to visual-only alerts automatically.

### 5. (Optional) Alarm sound

A default `assets/sounds/alarm.wav` is included. Replace it with your own **loud**
`alarm.wav` if you prefer (the real-time alarm feature reads this file). The alarm
falls back to a Windows built-in beep (`winsound`) if pygame is unavailable.

---

## How to Run

```bash
python main.py
```

Run without phone detection (lower CPU usage):

```bash
python main.py --no-phone
```

In the dashboard:
1. Click **Start Monitoring** to begin.
2. **Calibrate** runs a short wizard to adapt thresholds to your face.
3. **Config** opens `config/config.yaml` in your default editor.

Press the window close button to stop and view the session summary.

---

## Configuration

All tunable parameters live in `config/config.yaml`:

- `detection.eyes.ear_threshold` - EAR below this = eye closed (default 0.21)
- `detection.eyes.ear_consecutive_frames` - frames closed before "prolonged closure" (default 45 ≈ 1.5 s at 30 fps)
- `detection.eyes.blink_min_frames` / `blink_max_frames` - normal blink duration window
- `detection.mouth.mar_threshold` - MAR above this = mouth open (default 0.50)
- `detection.mouth.yawn_consecutive_frames` - frames of open mouth before a yawn counts (default 30)
- `detection.head_pose.pitch_threshold` / `yaw_threshold` - away thresholds in degrees
- `detection.hand.safe_region_y_max` - normalized Y below which hands are considered "down"
- `detection.phone.*` - enable/disable YOLO, confidence, check interval
- `risk_engine.*` - signal weights, state thresholds, hysteresis
- `alerts.*` - cooldown, sound on/off, sound path
- `alarm.*` - real-time alarm: `cooldown_seconds` (min gap between sounds), `alarm_duration_seconds` (how long each sound plays), `yawn_cooldown_seconds` (same-yawn debounce), `eyes_off_threshold_seconds` (face-missing time before "EYES OFF THE ROAD"), `sound_file`, `sound_enabled`
- `camera.*` - index, resolution, target FPS

---

## Calibration

Different users have different eye sizes, mouth shapes, and camera positions. Use the **Calibrate** button:

1. Sit in a normal driving posture, look straight ahead.
2. Keep eyes open for several seconds (baseline EAR is recorded).
3. Blink normally.
4. Open your mouth naturally.

The wizard then adjusts `ear_threshold` (≈ 70% of your baseline) and `mar_threshold` automatically.

---

## Real-Time Alarm (AlarmManager)

`src/monitoring/alarm_manager.py` adds a loud, **non-blocking** audible alarm on top of the
existing alerts. It is fully separate from the detection logic:

- **Inputs (per frame):** `eye_closed` (`EyeAnalyzer.closure_detected`, already confirms ~1.5 s of closure),
  `eyes_off` (face missing for `eyes_off_threshold_seconds`), `yawn` (`MouthAnalyzer.yawn_detected`),
  `phone` (`PhoneDetector.detected`).
- **Trigger:** a danger must persist for 2 consecutive update calls (anti single-frame noise, on top of the
  detectors' own consecutive-frame filters). The sound plays once per event.
- **Cooldown (`alarm.cooldown_seconds`):** suppresses rapid re-triggering after an event.
- **Yawn debounce (`alarm.yawn_cooldown_seconds`):** the same yawn never re-triggers.
- **Merging:** simultaneous conditions play **one** alarm sound; **all** reasons are shown in the UI.
- **Auto-stop:** the sound stops immediately when no dangerous condition remains.
- **Non-blocking:** audio runs on a background thread (`_AlarmPlayer`); the video loop never waits on sound.

### Status indicator

| Status | Color |
|---|---|
| NORMAL | green |
| DROWSINESS DETECTED (EYES CLOSED / EYES OFF) | red |
| YAWNING DETECTED | orange/red |
| PHONE DETECTED | red |
| ALARM ACTIVE (sound playing) | **flashing red** |

The UI also shows the reason, e.g. `⚠ ALERT: EYES CLOSED` or `⚠ ALERT: EYES CLOSED, PHONE DETECTED`.

---

## Scoring & State Machine

Each signal contributes weighted risk points. The **safety score** = 100 − normalized risk.

| Signal | Weight (max) |
|---|---|
| Prolonged eye closure | 35 |
| Head looking away | 25 |
| Phone in face region | 30 |
| Yawning | 20 |
| Hand down | 10 |

States with hysteresis (require N consecutive frames to transition, so brief movements don't false-alarm):

- **SAFE** (score ≥ 80) - green
- **CAUTION** (60–79) - yellow, "Please stay attentive."
- **ATTENTION REQUIRED** (40–59) - orange
- **DROWSY / DISTRACTED** (20–39) - red, "Please take a break."
- **DANGER** (< 20) - dark red, critical alarm

Recovery is deliberately slower than escalation to avoid alert flip-flopping.

---

## Performance

Measured values on a student laptop (CPU-only, 640×480):

| Path | Measured |
|---|---|
| Face landmarker (all facial signals) | ~12 ms/frame |
| Hand landmarker | ~13 ms/frame |
| Main video loop (face + hand) | ~20 FPS |
| YOLOv8n phone inference | ~450 ms/frame (runs on a background thread, only every N frames) |

Because YOLO runs asynchronously, normal monitoring stays smooth; phone status updates as fast as the background worker allows.

Suggested if your machine is slower:
- Reduce `camera.width/height` to 480×360.
- Run with `--no-phone`.
- Increase `detection.phone.check_interval`.

---

## Session Summary & Storage

When you stop monitoring, a summary dialog shows:

```
================================
       SESSION SUMMARY
================================

Duration:                    32:15
Yawns:                       7
Eye Closure Events:          4
Looking Away:                3
Hand Down:                   2
Phone Events:                1

Average Safety Score:        82/100
Maximum Risk:                ATTENTION REQUIRED
================================
```

Sessions and events are stored in SQLite at `~/.smart_driver_monitor/sessions.db`
(enable/disable via `storage.enabled` in the config). Stored data contains **no personal information**.

---

## Testing

Unit tests (no webcam needed):

```bash
python tests/test_math.py
python tests/test_risk_engine.py
python tests/test_config.py
python tests/test_alarm_manager.py
```

Manual test scenarios to try with a real webcam:

1. Normal driving posture - system should stay **SAFE**.
2. Slow/quick blinking - a blink should **not** trigger a warning.
3. Close eyes ~2 seconds - **DROWSY** warning.
4. Yawn for ~1.5 seconds - yawn counted, **CAUTION/ATTENTION**.
5. Look left/right for ~3 seconds - **looking-away** event.
6. Look down frequently - head direction shows **DOWN**.
7. Move a hand below the safe region - **hand down** status.
8. Hold a phone in front of your face - **PHONE** alert (if enabled).
9. Cover your face - shows "NO FACE", no crash.
10. Poor lighting - degrades gracefully.
11. Close eyes ~2 s - red **ALARM ACTIVE** + sound, reason `EYES CLOSED`.
12. Yawn once - one alarm, reason `YAWNING DETECTED`; the same yawn does not repeat.
13. Hold a phone in front of your face - alarm, reason `PHONE DETECTED`.
14. Trigger two dangers at once - only one sound plays, both reasons shown.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Face landmarker model not found` | Run `python setup_models.py` |
| `mediapipe not installed` | `pip install mediapipe` (0.10.30+ required for `mp.tasks`) |
| Camera won't open | Check `camera.index` in config; try 0, 1, 2 |
| No phone detection | Install ultralytics; verify `models/yolov8n.pt` exists |
| Audio not playing | Place WAVs in `assets/sounds/`; check device volume; set `alerts.sound_enabled: false` / `alarm.sound_enabled: false` to silence |
| Alarm silent but UI shows ALARM ACTIVE | Check `assets/sounds/alarm.wav` exists; the manager falls back to `winsound` only on Windows; if pygame is missing install it with `pip install pygame` |
| Low FPS | Lower camera resolution, use `--no-phone`, raise `check_interval` |
| Session dialog at exit | Expected - session summary + DB save on stop |
| MediaPipe "landmark_projection_calculator" warning | Harmless log noise from MediaPipe |

---

## Known Limitations

- Not a safety-rated device; research/demo only.
- Head pose uses a generic 3D face model (assumes average anthropometry).
- Hands below the camera's field of view report **NOT DETECTED** (by design, this is not treated as "down").
- YOLO phone detection cannot distinguish the driver's phone from a phone beside them; position heuristics only.
- Very poor lighting will degrade all detection quality.

---

## Future Improvements

- ONNX/OpenVINO export of YOLO for faster CPU inference
- Iris tracking for gaze direction estimation
- Personalized per-user profiles stored in the database
- Session history viewer/graphs
- Voice alerts via TTS
- Raspberry-Pi-friendly build (TFLite versions of the models)