import os
import threading
import time
import logging
import cv2

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    logger.warning("ultralytics not installed. Phone detection disabled.")

CELL_PHONE_CLASS = 67


class PhoneDetector:
    def __init__(self, model_path="models/yolov8n.pt",
                 confidence_threshold=0.4, check_interval=3,
                 face_region_y_min=0.0, face_region_y_max=0.5):
        if not HAS_YOLO:
            raise ImportError("ultralytics is required for phone detection")
        resolved = self._resolve_model(model_path)
        self._model = YOLO(resolved)
        self._conf_threshold = confidence_threshold
        self._check_interval = check_interval
        self._face_y_min = face_region_y_min
        self._face_y_max = face_region_y_max

        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_id = 0
        self._last_processed_id = -1
        self._phone_detected = False
        self._phone_confidence = 0.0
        self._phone_box = None
        self._phone_in_face_region = False
        self._total_phone_events = 0
        self._event_active = False
        self._running = False
        self._worker = None

    def _resolve_model(self, path):
        if os.path.exists(path):
            return path
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full = os.path.join(base, path)
        if os.path.exists(full):
            return full
        try:
            from ultralytics import YOLO
            YOLO(os.path.basename(path))
            return path
        except Exception:
            raise FileNotFoundError(f"YOLO model not found: {path}")

    def start(self):
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self):
        processed_since_check = 0
        while self._running:
            with self._lock:
                frame = self._latest_frame
                frame_id = self._frame_id
            if frame is None or frame_id <= self._last_processed_id:
                time.sleep(0.05)
                continue

            processed_since_check += 1
            if processed_since_check < self._check_interval:
                time.sleep(0.02)
                continue

            processed_since_check = 0
            self._last_processed_id = frame_id
            self._detect_now(frame)

    def submit_frame(self, frame):
        self._frame_id += 1
        with self._lock:
            self._latest_frame = frame

    def _detect_now(self, frame):
        detected = False
        confidence = 0.0
        box = None
        in_region = False

        try:
            results = self._model(frame, verbose=False)
            for result in results:
                for bbox in result.boxes:
                    cls = int(bbox.cls[0])
                    conf = float(bbox.conf[0])
                    if cls == CELL_PHONE_CLASS and conf >= self._conf_threshold:
                        detected = True
                        confidence = conf
                        x1, y1, x2, y2 = bbox.xyxy[0].tolist()
                        box = (int(x1), int(y1), int(x2), int(y2))
                        h = frame.shape[0]
                        phone_center_y = (y1 + y2) / 2 / h if h > 0 else 0
                        in_region = self._face_y_min <= phone_center_y <= self._face_y_max
                        break
                if detected:
                    break
        except Exception as e:
            logger.error("YOLO inference error: %s", e)
            return

        with self._lock:
            self._phone_detected = detected
            self._phone_confidence = confidence
            self._phone_box = box
            self._phone_in_face_region = in_region
            if detected and not self._event_active:
                self._event_active = True
                self._total_phone_events += 1
                logger.info("Phone detected (#%d, conf=%.2f)", self._total_phone_events, confidence)
            elif not detected:
                self._event_active = False

    def draw(self, frame):
        with self._lock:
            detected = self._phone_detected
            box = self._phone_box
            conf = self._phone_confidence
            in_region = self._phone_in_face_region
        if detected and box:
            x1, y1, x2, y2 = box
            color = (0, 0, 255) if in_region else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"PHONE {conf:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    @property
    def detected(self):
        with self._lock:
            return self._phone_detected

    @property
    def confidence(self):
        with self._lock:
            return self._phone_confidence

    @property
    def in_face_region(self):
        with self._lock:
            return self._phone_in_face_region

    @property
    def total_phone_events(self):
        with self._lock:
            return self._total_phone_events

    def stop(self):
        self._running = False
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def reset(self):
        with self._lock:
            self._phone_detected = False
            self._phone_confidence = 0.0
            self._phone_box = None
            self._total_phone_events = 0
            self._event_active = False