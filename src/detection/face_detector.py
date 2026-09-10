import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    logger.warning("mediapipe not installed. Face detection disabled.")


class FaceDetector:
    def __init__(self, model_path, num_faces=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        if not HAS_MEDIAPIPE:
            raise ImportError("mediapipe is required")
        resolved = self._resolve_model(model_path)
        base_options = mp_python.BaseOptions(model_asset_path=resolved)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._results = None
        self._face_detected = False
        self._landmarks = None
        self._transform_matrix = None
        logger.info("FaceLandmarker loaded: %s", resolved)

    def _resolve_model(self, path):
        if os.path.exists(path):
            return path
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full = os.path.join(base, path)
        if os.path.exists(full):
            return full
        raise FileNotFoundError(f"Face landmarker model not found: {path}")

    def detect(self, rgb_frame, timestamp_ms):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._results = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        if self._results and self._results.face_landmarks:
            self._face_detected = True
            self._landmarks = self._results.face_landmarks[0]
            if self._results.facial_transformation_matrixes:
                self._transform_matrix = np.array(self._results.facial_transformation_matrixes[0])
            else:
                self._transform_matrix = None
        else:
            self._face_detected = False
            self._landmarks = None
            self._transform_matrix = None

        return self._face_detected

    @property
    def detected(self):
        return self._face_detected

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def transform_matrix(self):
        return self._transform_matrix

    @property
    def raw_results(self):
        return self._results
