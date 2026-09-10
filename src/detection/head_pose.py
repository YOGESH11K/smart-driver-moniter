import math
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

PNP_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0],
], dtype=np.float64)

NOSE_TIP_IDX = 1


class HeadPoseDetector:
    def __init__(self, pitch_threshold=15, yaw_threshold=20,
                 away_consecutive_frames=20):
        self._pitch_threshold = pitch_threshold
        self._yaw_threshold = yaw_threshold
        self._away_frames = away_consecutive_frames

        self._pitch = 0.0
        self._yaw = 0.0
        self._roll = 0.0
        self._direction = "FORWARD"
        self._looking_away = False
        self._away_counter = 0
        self._total_away_events = 0
        self._away_event_active = False
        self._rvec = None
        self._tvec = None

    def analyze(self, landmarks, w, h):
        if landmarks is None:
            self._direction = "NO FACE"
            self._looking_away = False
            return

        image_points = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in PNP_LANDMARK_INDICES
        ], dtype=np.float64)

        focal_length = w
        cam_matrix = np.array([
            [focal_length, 0, w / 2.0],
            [0, focal_length, h / 2.0],
            [0, 0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, image_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            self._direction = "UNKNOWN"
            return

        self._rvec = rvec
        self._tvec = tvec

        rot_matrix, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_matrix)
        self._pitch = float(angles[0])
        self._yaw = float(angles[1])
        self._roll = float(angles[2])

        self._direction = self._classify_direction()

        is_away = (abs(self._pitch) > self._pitch_threshold or
                   abs(self._yaw) > self._yaw_threshold)

        if is_away:
            self._away_counter += 1
            if self._away_counter >= self._away_frames:
                self._looking_away = True
                if not self._away_event_active:
                    self._away_event_active = True
                    self._total_away_events += 1
                    logger.info("Looking away detected (#%d)", self._total_away_events)
        else:
            self._away_counter = 0
            self._looking_away = False
            self._away_event_active = False

    def _classify_direction(self):
        if abs(self._pitch) <= self._pitch_threshold and abs(self._yaw) <= self._yaw_threshold:
            return "FORWARD"
        if self._yaw > self._yaw_threshold:
            return "RIGHT"
        if self._yaw < -self._yaw_threshold:
            return "LEFT"
        if self._pitch > self._pitch_threshold:
            return "DOWN"
        if self._pitch < -self._pitch_threshold:
            return "UP"
        return "FORWARD"

    def draw_axis(self, frame, w, h, landmarks=None):
        if self._rvec is None or self._tvec is None:
            return
        nose = None
        if landmarks is not None:
            nose = (int(landmarks[NOSE_TIP_IDX].x * w), int(landmarks[NOSE_TIP_IDX].y * h))
        elif hasattr(self, "_last_landmarks") and self._last_landmarks is not None:
            nose = (int(self._last_landmarks[NOSE_TIP_IDX].x * w),
                    int(self._last_landmarks[NOSE_TIP_IDX].y * h))
        try:
            axis_points = np.float32([
                [0, 0, 0], [80, 0, 0], [0, -80, 0], [0, 0, 80]
            ]).reshape(-1, 1, 3)
            focal_length = w
            cam_matrix = np.array([
                [focal_length, 0, w / 2.0],
                [0, focal_length, h / 2.0],
                [0, 0, 1.0],
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            imgpts, _ = cv2.projectPoints(axis_points, self._rvec, self._tvec, cam_matrix, dist_coeffs)
            origin = tuple(imgpts[0].ravel().astype(int))
            cv2.line(frame, origin, tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 3)
            cv2.line(frame, origin, tuple(imgpts[2].ravel().astype(int)), (0, 255, 0), 3)
            cv2.line(frame, origin, tuple(imgpts[3].ravel().astype(int)), (255, 0, 0), 3)
        except Exception:
            pass

    def update_thresholds(self, pitch=None, yaw=None, away_frames=None):
        if pitch is not None:
            self._pitch_threshold = pitch
        if yaw is not None:
            self._yaw_threshold = yaw
        if away_frames is not None:
            self._away_frames = away_frames

    @property
    def pitch(self):
        return self._pitch

    @property
    def yaw(self):
        return self._yaw

    @property
    def roll(self):
        return self._roll

    @property
    def direction(self):
        return self._direction

    @property
    def looking_away(self):
        return self._looking_away

    @property
    def away_frames(self):
        return self._away_counter

    @property
    def total_away_events(self):
        return self._total_away_events

    def reset(self):
        self._pitch = 0.0
        self._yaw = 0.0
        self._roll = 0.0
        self._direction = "FORWARD"
        self._looking_away = False
        self._away_counter = 0
        self._total_away_events = 0
        self._away_event_active = False
        self._rvec = None
        self._tvec = None
