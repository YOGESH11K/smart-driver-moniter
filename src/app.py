import os
import time
import logging
import threading

import cv2
import numpy as np

from src.camera.webcam import WebcamCapture
from src.detection.face_detector import FaceDetector
from src.detection.eye_analyzer import EyeAnalyzer
from src.detection.mouth_analyzer import MouthAnalyzer
from src.detection.head_pose import HeadPoseDetector
from src.detection.hand_detector import HandDetector
from src.detection.phone_detector import PhoneDetector
from src.monitoring.risk_engine import RiskEngine, STATES
from src.monitoring.alert_manager import AlertManager
from src.monitoring.alarm_manager import AlarmManager
from src.storage.database import Database
from src.storage.models import SessionSummary, SessionEvent
from src.utils.config import Config

logger = logging.getLogger(__name__)

FACE_CONNECTION_LINES = [
    (33, 133), (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (362, 263), (362, 382), (382, 381), (381, 380),
    (380, 374), (374, 373), (373, 390), (390, 249), (249, 263),
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405),
    (405, 321), (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
]

EYE_LANDMARK_GROUPS = {
    "right": [33, 160, 158, 133, 153, 144],
    "left": [362, 385, 387, 263, 373, 380],
}

MOUTH_LANDMARKS = [61, 291, 13, 14, 0, 185, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                   40, 37, 267, 269, 270, 409, 78, 191, 308, 324, 318, 402, 317, 14, 87]

SAFE_HAND_COLOR = (0, 255, 0)
DOWN_HAND_COLOR = (0, 0, 255)
PHONE_COLOR = (0, 0, 255)


class DriverMonitorApp:
    def __init__(self, config_path=None):
        self._config = Config(config_path)
        self._monitoring = False
        self._thread = None
        self._stop_event = threading.Event()
        self._session_start = 0.0
        self._session_elapsed = 0.0
        self._score_history = []

        self._camera = None
        self._face_detector = None
        self._eye_analyzer = None
        self._mouth_analyzer = None
        self._head_pose = None
        self._hand_detector = None
        self._phone_detector = None
        self._risk_engine = RiskEngine(
            weights=self._config.get("risk_engine", "weights"),
            state_thresholds=self._config.get("risk_engine", "state_thresholds"),
            transition_frames=self._config.get("risk_engine", "transition_frames"),
        )
        self._alert_manager = AlertManager(
            cooldown_seconds=self._config.get("alerts", "cooldown_seconds", default=5),
            sound_enabled=self._config.get("alerts", "sound_enabled", default=True),
            sound_path=self._resolve_asset_path("assets/sounds/"),
            visual_duration=self._config.get("alerts", "visual_duration_seconds", default=3),
        )
        self._alarm_manager = AlarmManager(
            sound_enabled=self._config.get("alarm", "sound_enabled", default=True),
            sound_path=self._resolve_asset_path(
                self._config.get("alarm", "sound_path", default="assets/sounds/")),
            sound_file=self._config.get("alarm", "sound_file", default="alarm.wav"),
            cooldown_seconds=self._config.get("alarm", "cooldown_seconds", default=10),
            alarm_duration_seconds=self._config.get("alarm", "alarm_duration_seconds", default=8),
            yawn_cooldown_seconds=self._config.get("alarm", "yawn_cooldown_seconds", default=8),
            eyes_off_threshold_seconds=self._config.get(
                "alarm", "eyes_off_threshold_seconds", default=1.5),
        )
        self._db = None
        if self._config.get("storage", "enabled", default=True):
            self._db = Database(self._config.get("storage", "db_path", default="~/.smart_driver_monitor/sessions.db"))

        self._annotated_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._latest_metrics = {}
        self._window = None
        self._show_landmarks = self._config.get("gui", "show_landmarks", default=True)
        self._show_head_arrow = self._config.get("gui", "show_head_arrow", default=True)

    def _resolve_asset_path(self, path):
        if os.path.exists(path):
            return path
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full = os.path.join(base, path)
        return full if os.path.exists(full) else path

    def set_window(self, window):
        self._window = window

    def _init_detectors(self):
        try:
            self._face_detector = FaceDetector(
                model_path=self._config.get("detection", "face", "model_path"),
                num_faces=self._config.get("detection", "face", "num_faces", default=1),
                min_detection_confidence=self._config.get("detection", "face", "min_detection_confidence", default=0.5),
                min_tracking_confidence=self._config.get("detection", "face", "min_tracking_confidence", default=0.5),
            )
        except ImportError:
            logger.warning("mediapipe not installed. Face-based detection disabled.")
            self._face_detector = None
        except FileNotFoundError as e:
            logger.warning("%s Run setup_models.py first.", e)
            self._face_detector = None

        try:
            self._hand_detector = HandDetector(
                model_path=self._config.get("detection", "hand", "model_path"),
                max_hands=self._config.get("detection", "hand", "max_hands", default=2),
                min_detection_confidence=self._config.get("detection", "hand", "min_detection_confidence", default=0.5),
                min_tracking_confidence=self._config.get("detection", "hand", "min_tracking_confidence", default=0.5),
                safe_region_y_max=self._config.get("detection", "hand", "safe_region_y_max", default=0.7),
                down_duration_frames=self._config.get("detection", "hand", "down_duration_frames", default=15),
            )
        except (ImportError, FileNotFoundError):
            logger.warning("Hand detection unavailable.")
            self._hand_detector = None

        phone_enabled = self._config.get("detection", "phone", "enabled", default=True)
        if phone_enabled:
            try:
                self._phone_detector = PhoneDetector(
                    model_path=self._config.get("detection", "phone", "model_path"),
                    confidence_threshold=self._config.get("detection", "phone", "confidence_threshold", default=0.4),
                    check_interval=self._config.get("detection", "phone", "check_interval", default=3),
                    face_region_y_min=self._config.get("detection", "phone", "face_region_y_min", default=0.0),
                    face_region_y_max=self._config.get("detection", "phone", "face_region_y_max", default=0.5),
                )
            except (ImportError, FileNotFoundError):
                logger.warning("Phone detection unavailable.")
                self._phone_detector = None
        else:
            self._phone_detector = None

        self._eye_analyzer = EyeAnalyzer(
            ear_threshold=self._config.get("detection", "eyes", "ear_threshold", default=0.21),
            consecutive_frames=self._config.get("detection", "eyes", "ear_consecutive_frames", default=45),
            blink_min=self._config.get("detection", "eyes", "blink_min_frames", default=2),
            blink_max=self._config.get("detection", "eyes", "blink_max_frames", default=6),
        )
        self._mouth_analyzer = MouthAnalyzer(
            mar_threshold=self._config.get("detection", "mouth", "mar_threshold", default=0.50),
            yawn_consecutive_frames=self._config.get("detection", "mouth", "yawn_consecutive_frames", default=30),
        )
        self._head_pose = HeadPoseDetector(
            pitch_threshold=self._config.get("detection", "head_pose", "pitch_threshold", default=15),
            yaw_threshold=self._config.get("detection", "head_pose", "yaw_threshold", default=20),
            away_consecutive_frames=self._config.get("detection", "head_pose", "away_consecutive_frames", default=20),
        )

    def toggle_monitoring(self):
        if self._monitoring:
            self.stop()
            return False
        self.start()
        return True

    def start(self):
        if self._monitoring:
            return
        if self._face_detector is None:
            self._init_detectors()

        self._camera = WebcamCapture(
            camera_index=self._config.get("camera", "index", default=0),
            width=self._config.get("camera", "width", default=640),
            height=self._config.get("camera", "height", default=480),
            target_fps=self._config.get("camera", "target_fps", default=30),
        )
        if not self._camera.open():
            logger.error("Failed to open webcam.")
            if self._window:
                self._alert_manager.check_and_alert("DANGER", phone_detected=False)
            return

        self._session_start = time.time()
        self._session_elapsed = 0.0
        self._score_history = []
        self._risk_engine.reset()
        self._alert_manager.reset()
        self._alarm_manager.reset()

        self._stop_event.clear()
        self._monitoring = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        if self._phone_detector is not None:
            self._phone_detector.start()
        logger.info("Monitoring started")

    def stop(self):
        if not self._monitoring:
            return
        self._monitoring = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._phone_detector is not None:
            self._phone_detector.stop()
        if self._camera:
            self._camera.close()
            self._camera = None
        self._save_session()
        logger.info("Monitoring stopped")

    def _process_loop(self):
        while not self._stop_event.is_set():
            ret, frame = self._camera.read()
            if not ret:
                time.sleep(0.02)
                continue

            h, w = frame.shape[:2]
            self._annotated_frame = frame.copy()

            timestamp_ms = int(time.time() * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_landmarks = None
            if self._face_detector is not None:
                try:
                    has_face = self._face_detector.detect(rgb, timestamp_ms)
                    face_landmarks = self._face_detector.landmarks
                except Exception as e:
                    logger.error("Face detection error: %s", e)
                    has_face = False
            else:
                has_face = False

            if has_face and face_landmarks is not None:
                self._eye_analyzer.analyze(face_landmarks, w, h)
                self._mouth_analyzer.analyze(face_landmarks, w, h)
                self._head_pose.analyze(face_landmarks, w, h)
                if self._show_head_arrow:
                    self._head_pose.draw_axis(self._annotated_frame, w, h, face_landmarks)
                self._draw_face_overlay(self._annotated_frame, face_landmarks, w, h)
            else:
                self._eye_analyzer.analyze(None, w, h)
                self._mouth_analyzer.analyze(None, w, h)
                self._head_pose.analyze(None, w, h)
                cv2.putText(self._annotated_frame, "NO FACE DETECTED",
                            (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 165, 255), 2)

            if self._hand_detector is not None:
                self._hand_detector.detect(rgb, timestamp_ms)
                if self._show_landmarks:
                    self._hand_detector.draw_landmarks(self._annotated_frame, w, h)

            if self._phone_detector is not None:
                self._phone_detector.submit_frame(frame.copy())
                self._phone_detector.draw(self._annotated_frame)

            self._update_risk()
            self._update_metrics()

    def _draw_face_overlay(self, frame, landmarks, w, h):
        if self._show_landmarks:
            # Eyes
            for eye_name, indices in EYE_LANDMARK_GROUPS.items():
                color = (0, 255, 0) if self._eye_analyzer.eye_status == "OPEN" else (0, 0, 255)
                for idx in indices:
                    lm = landmarks[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, color, -1)
            # Mouth
            for idx in MOUTH_LANDMARKS:
                lm = landmarks[idx]
                mx, my = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (mx, my), 2, (255, 255, 0), -1)
            # Face boundary points
            for idx in [1, 152, 10]:
                lm = landmarks[idx]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 0, 255), -1)

    def _update_risk(self):
        eye_closed = self._eye_analyzer.closure_detected
        eye_frames = self._eye_analyzer.closure_duration_frames
        yawning = self._mouth_analyzer.yawn_detected
        yawn_frames = self._mouth_analyzer.yawn_duration_frames
        looking_away = self._head_pose.looking_away
        away_frames = self._head_pose.away_frames
        hand_down_left = self._hand_detector.left_hand_down if self._hand_detector else False
        hand_down_right = self._hand_detector.right_hand_down if self._hand_detector else False
        phone = self._phone_detector.detected if self._phone_detector else False

        self._risk_engine.update(
            eye_closure=eye_closed, eye_closure_frames=eye_frames,
            yawn=yawning, yawn_frames=yawn_frames,
            looking_away=looking_away, away_frames=away_frames,
            hand_down_left=hand_down_left, hand_down_right=hand_down_right,
            phone_detected=phone,
        )

    def _update_metrics(self):
        fps = self._camera.fps if self._camera else 0
        elapsed = time.time() - self._session_start if self._monitoring else 0
        self._session_elapsed = elapsed
        self._score_history.append(self._risk_engine.safety_score)

        alert = self._alert_manager.check_and_alert(
            self._risk_engine.state,
            phone_detected=self._phone_detector.detected if self._phone_detector else False,
        )

        alarm_state = self._alarm_manager.update(
            eye_closed=bool(self._eye_analyzer.closure_detected),
            eyes_off=not (self._face_detector.detected if self._face_detector else False),
            yawn=bool(self._mouth_analyzer.yawn_detected),
            phone=bool(self._phone_detector.detected if self._phone_detector else False),
        )
        if alarm_state["active"]:
            logger.warning("ALARM: %s", ", ".join(alarm_state["reasons"]))

        hand_down_events = 0
        if self._hand_detector:
            hand_down_events = self._hand_detector.total_hand_down_events

        self._latest_metrics = {
            "state": self._risk_engine.state,
            "safety_score": self._risk_engine.safety_score,
            "ear": self._eye_analyzer.ear,
            "mar": self._mouth_analyzer.mar,
            "eye_status": self._eye_analyzer.eye_status,
            "mouth_status": self._mouth_analyzer.mouth_status,
            "head_direction": self._head_pose.direction,
            "pitch": self._head_pose.pitch,
            "yaw": self._head_pose.yaw,
            "left_hand": self._hand_detector.left_hand_status if self._hand_detector else "NOT DETECTED",
            "right_hand": self._hand_detector.right_hand_status if self._hand_detector else "NOT DETECTED",
            "phone_detected": self._phone_detector.detected if self._phone_detector else False,
            "yawns": self._mouth_analyzer.yawn_count,
            "closures": self._eye_analyzer.total_closures,
            "looking_away": self._head_pose.total_away_events,
            "hand_down": hand_down_events,
            "phone_events": self._phone_detector.total_phone_events if self._phone_detector else 0,
            "blinks": self._eye_analyzer.blink_count,
            "duration_seconds": elapsed,
            "avg_score": sum(self._score_history) / len(self._score_history) if self._score_history else 100,
            "max_risk": self._risk_engine.max_risk_level,
            "fps": fps,
            "face_detected": self._face_detector.detected if self._face_detector else False,
            "alert": alert,
            "alarm": alarm_state,
        }

        if self._window:
            self._window.after(0, self._window.update_dashboard, self._latest_metrics.copy())
            self._window.after(0, self._window._camera_panel.update_frame, self._annotated_frame.copy())

    def get_latest_metrics(self):
        return self._latest_metrics

    def _save_session(self):
        if self._db is None:
            return
        duration = self._session_elapsed
        if duration < 3:
            return
        summary = SessionSummary(
            start_time=self._session_start,
            end_time=time.time(),
            duration_seconds=duration,
            total_yawns=self._mouth_analyzer.yawn_count,
            total_eye_closures=self._eye_analyzer.total_closures,
            total_distraction_events=self._risk_engine.total_risk_events,
            total_phone_events=self._phone_detector.total_phone_events if self._phone_detector else 0,
            total_hand_down_events=self._hand_detector.total_hand_down_events if self._hand_detector else 0,
            total_looking_away_events=self._head_pose.total_away_events,
            avg_safety_score=sum(self._score_history) / len(self._score_history) if self._score_history else 100,
            max_risk_level=self._risk_engine.max_risk_level,
        )
        try:
            session_id = self._db.save_session(summary)
            logger.info("Saved session id=%d", session_id)
            if self._window:
                self._window.show_session_summary(self._format_summary(summary))
        except Exception as e:
            logger.error("Failed to save session: %s", e)

    def _format_summary(self, summary):
        mins, secs = divmod(int(summary.duration_seconds), 60)
        lines = [
            "",
            "=" * 40,
            "      SESSION SUMMARY",
            "=" * 40,
            "",
            f"Duration:                    {mins:02d}:{secs:02d}",
            f"Yawns:                       {summary.total_yawns}",
            f"Eye Closure Events:          {summary.total_eye_closures}",
            f"Looking Away:                {summary.total_looking_away_events}",
            f"Hand Down:                   {summary.total_hand_down_events}",
            f"Phone Events:                {summary.total_phone_events}",
            "",
            f"Average Safety Score:        {summary.avg_safety_score:.0f}/100",
            f"Maximum Risk:                {summary.max_risk_level}",
            "",
            f"Overall Status:              {summary.max_risk_level}",
            "=" * 40,
            "",
        ]
        return "\n".join(lines)

    def run_calibration(self):
        if not self._face_detector and not self._monitoring:
            self._init_detectors()
        try:
            ear_values = []
            mar_values = []
            cam = WebcamCapture(
                camera_index=self._config.get("camera", "index", default=0),
                width=self._config.get("camera", "width", default=640),
                height=self._config.get("camera", "height", default=480),
                target_fps=15,
            )
            if not cam.open():
                logger.error("Webcam unavailable for calibration")
                return
            start = time.time()
            while time.time() - start < 6.0:
                ret, frame = cam.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ts = int(time.time() * 1000)
                    if self._face_detector.detect(rgb, ts):
                        lm = self._face_detector.landmarks
                        h, w = frame.shape[:2]
                        self._eye_analyzer.analyze(lm, w, h)
                        self._mouth_analyzer.analyze(lm, w, h)
                        if self._eye_analyzer.eye_status != "CLOSED":
                            ear_values.append(self._eye_analyzer.ear)
                        mar_values.append(self._mouth_analyzer.mar)
                time.sleep(0.1)
            cam.close()
            if ear_values:
                base_ear = sum(ear_values) / len(ear_values)
                new_threshold = max(0.15, base_ear * 0.7)
                self._eye_analyzer.update_thresholds(ear_threshold=new_threshold)
                logger.info("Calibrated EAR threshold: %.3f (base %.3f)", new_threshold, base_ear)
            if mar_values:
                base_mar = sum(mar_values) / len(mar_values)
                new_mar = max(0.40, base_mar + 0.30)
                self._mouth_analyzer.update_thresholds(mar_threshold=new_mar)
                logger.info("Calibrated MAR threshold: %.3f", new_mar)
        except Exception as e:
            logger.error("Calibration failed: %s", e)

    def open_config(self):
        try:
            os.startfile(self._config._path)
        except Exception:
            logger.info("Config path: %s", self._config._path)

    def shutdown(self):
        self.stop()
        self._alarm_manager.close()
        if self._db:
            self._db.close()
        logger.info("Application shutdown complete")