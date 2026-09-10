import logging

from src.utils.math_helpers import compute_ear, ConsecutiveCounter

logger = logging.getLogger(__name__)

RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]


class EyeAnalyzer:
    def __init__(self, ear_threshold=0.21, consecutive_frames=45,
                 blink_min=2, blink_max=6):
        self._ear_threshold = ear_threshold
        self._consecutive_frames = consecutive_frames
        self._blink_min = blink_min
        self._blink_max = blink_max

        self._ear = 0.0
        self._right_ear = 0.0
        self._left_ear = 0.0
        self._eye_status = "OPEN"
        self._blink_count = 0
        self._closure_counter = ConsecutiveCounter()
        self._closure_detected = False
        self._total_closures = 0
        self._closure_duration_frames = 0

    def analyze(self, landmarks, w, h):
        if landmarks is None:
            self._eye_status = "NO FACE"
            return

        self._right_ear = compute_ear(landmarks, RIGHT_EYE_INDICES, w, h)
        self._left_ear = compute_ear(landmarks, LEFT_EYE_INDICES, w, h)
        self._ear = (self._right_ear + self._left_ear) / 2.0

        is_closed = self._ear < self._ear_threshold
        self._closure_counter.update(is_closed)

        if is_closed:
            self._eye_status = "CLOSED"
            self._closure_duration_frames = self._closure_counter.count

            if self._closure_counter.count == self._blink_min:
                pass
            elif self._closure_counter.count >= self._consecutive_frames:
                if not self._closure_detected:
                    self._closure_detected = True
                    self._total_closures += 1
                    logger.info("Prolonged eye closure detected (#%d)", self._total_closures)
        else:
            self._eye_status = "OPEN"
            count = self._closure_counter.count
            if self._blink_min <= count <= self._blink_max:
                self._blink_count += 1
                logger.debug("Blink detected (#%d)", self._blink_count)
            self._closure_counter.reset()
            self._closure_detected = False
            self._closure_duration_frames = 0

    def update_thresholds(self, ear_threshold=None, consecutive_frames=None):
        if ear_threshold is not None:
            self._ear_threshold = ear_threshold
        if consecutive_frames is not None:
            self._consecutive_frames = consecutive_frames

    @property
    def ear(self):
        return self._ear

    @property
    def right_ear(self):
        return self._right_ear

    @property
    def left_ear(self):
        return self._left_ear

    @property
    def eye_status(self):
        return self._eye_status

    @property
    def blink_count(self):
        return self._blink_count

    @property
    def closure_detected(self):
        return self._closure_detected

    @property
    def total_closures(self):
        return self._total_closures

    @property
    def closure_duration_frames(self):
        return self._closure_duration_frames

    def reset(self):
        self._ear = 0.0
        self._right_ear = 0.0
        self._left_ear = 0.0
        self._eye_status = "OPEN"
        self._blink_count = 0
        self._closure_counter.reset()
        self._closure_detected = False
        self._total_closures = 0
        self._closure_duration_frames = 0
