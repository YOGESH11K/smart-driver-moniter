import os
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    logger.warning("mediapipe not installed. Hand detection disabled.")


class HandDetector:
    def __init__(self, model_path, max_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 safe_region_y_max=0.7, down_duration_frames=15):
        if not HAS_MEDIAPIPE:
            raise ImportError("mediapipe is required")
        resolved = self._resolve_model(model_path)
        base_options = mp_python.BaseOptions(model_asset_path=resolved)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._safe_y_max = safe_region_y_max
        self._down_frames = down_duration_frames
        self._results = None

        self._hands_data = {
            "left": {"status": "NOT DETECTED", "down_counter": 0, "is_down": False},
            "right": {"status": "NOT DETECTED", "down_counter": 0, "is_down": False},
        }
        self._total_hand_down_events = 0

        logger.info("HandLandmarker loaded: %s", resolved)

    def _resolve_model(self, path):
        if os.path.exists(path):
            return path
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full = os.path.join(base, path)
        if os.path.exists(full):
            return full
        raise FileNotFoundError(f"Hand landmarker model not found: {path}")

    def detect(self, rgb_frame, timestamp_ms):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._results = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        for key in self._hands_data:
            self._hands_data[key]["status"] = "NOT DETECTED"
            self._hands_data[key]["is_down"] = False
            self._hands_data[key]["down_counter"] = 0

        if self._results and self._results.hand_landmarks:
            for idx, hand_lm in enumerate(self._results.hand_landmarks):
                handedness_label = (
                    getattr(self._results.handedness[idx][0], "category_name", None)
                    or getattr(self._results.handedness[idx][0], "label", "Right")
                )
                key = "left" if handedness_label == "Left" else "right"
                wrist_y = hand_lm[0].y
                is_down = wrist_y > self._safe_y_max

                hand = self._hands_data[key]
                hand["status"] = "SAFE"
                if is_down:
                    hand["down_counter"] += 1
                    if hand["down_counter"] >= self._down_frames:
                        hand["status"] = "DOWN"
                        hand["is_down"] = True
                        if not hand.get("_event_fired", False):
                            self._total_hand_down_events += 1
                            hand["_event_fired"] = True
                    else:
                        hand["status"] = "SAFE"
                else:
                    hand["down_counter"] = 0
                    hand["_event_fired"] = False
                    hand["status"] = "SAFE"

    def draw_landmarks(self, frame, w, h):
        if self._results and self._results.hand_landmarks:
            for hand_lm in self._results.hand_landmarks:
                for lm in hand_lm:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

    @property
    def left_hand_status(self):
        return self._hands_data["left"]["status"]

    @property
    def right_hand_status(self):
        return self._hands_data["right"]["status"]

    @property
    def left_hand_down(self):
        return self._hands_data["left"]["is_down"]

    @property
    def right_hand_down(self):
        return self._hands_data["right"]["is_down"]

    @property
    def total_hand_down_events(self):
        return self._total_hand_down_events

    def reset(self):
        for key in self._hands_data:
            self._hands_data[key] = {"status": "NOT DETECTED", "down_counter": 0, "is_down": False}
        self._total_hand_down_events = 0
