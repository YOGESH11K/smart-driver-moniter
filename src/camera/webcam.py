import time
import logging
import threading
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WebcamCapture:
    def __init__(self, camera_index=0, width=640, height=480, target_fps=30):
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._target_fps = target_fps
        self._cap = None
        self._running = False
        self._thread = None
        self._frame_queue = deque(maxlen=2)
        self._lock = threading.Lock()
        self._fps_counter = deque(maxlen=30)
        self._last_frame_time = 0
        self._current_fps = 0.0

    def open(self):
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            logger.error("Cannot open camera %d", self._camera_index)
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Webcam opened: camera %d (%dx%d @ %d fps)",
                     self._camera_index, self._width, self._height, self._target_fps)
        return True

    def _capture_loop(self):
        frame_interval = 1.0 / self._target_fps
        while self._running and self._cap and self._cap.isOpened():
            t0 = time.time()
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            now = time.time()
            if self._last_frame_time > 0:
                dt = now - self._last_frame_time
                self._fps_counter.append(dt)
                if self._fps_counter:
                    self._current_fps = 1.0 / (sum(self._fps_counter) / len(self._fps_counter))
            self._last_frame_time = now
            with self._lock:
                self._frame_queue.append((now, frame))
            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self):
        with self._lock:
            if self._frame_queue:
                return True, self._frame_queue[-1][1].copy()
        return False, None

    @property
    def fps(self):
        return self._current_fps

    @property
    def is_opened(self):
        return self._running and self._cap is not None and self._cap.isOpened()

    def close(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Webcam closed")
