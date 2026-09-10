import logging

from src.utils.math_helpers import compute_mar, ConsecutiveCounter

logger = logging.getLogger(__name__)


class MouthAnalyzer:
    def __init__(self, mar_threshold=0.50, yawn_consecutive_frames=30):
        self._mar_threshold = mar_threshold
        self._yawn_frames = yawn_consecutive_frames

        self._mar = 0.0
        self._mouth_status = "NORMAL"
        self._yawn_count = 0
        self._yawn_counter = ConsecutiveCounter()
        self._yawn_detected = False
        self._yawn_duration_frames = 0

    def analyze(self, landmarks, w, h):
        if landmarks is None:
            self._mouth_status = "NO FACE"
            return

        self._mar = compute_mar(landmarks, w, h)

        is_yawning = self._mar > self._mar_threshold
        self._yawn_counter.update(is_yawning)

        if is_yawning:
            self._mouth_status = "YAWNING"
            self._yawn_duration_frames = self._yawn_counter.count
            if self._yawn_counter.count >= self._yawn_frames:
                if not self._yawn_detected:
                    self._yawn_detected = True
                    self._yawn_count += 1
                    logger.info("Yawn detected (#%d)", self._yawn_count)
        else:
            self._mouth_status = "NORMAL"
            self._yawn_counter.reset()
            self._yawn_detected = False
            self._yawn_duration_frames = 0

    def update_thresholds(self, mar_threshold=None, yawn_frames=None):
        if mar_threshold is not None:
            self._mar_threshold = mar_threshold
        if yawn_frames is not None:
            self._yawn_frames = yawn_frames

    @property
    def mar(self):
        return self._mar

    @property
    def mouth_status(self):
        return self._mouth_status

    @property
    def yawn_count(self):
        return self._yawn_count

    @property
    def yawn_detected(self):
        return self._yawn_detected

    @property
    def yawn_duration_frames(self):
        return self._yawn_duration_frames

    def reset(self):
        self._mar = 0.0
        self._mouth_status = "NORMAL"
        self._yawn_count = 0
        self._yawn_counter.reset()
        self._yawn_detected = False
        self._yawn_duration_frames = 0
