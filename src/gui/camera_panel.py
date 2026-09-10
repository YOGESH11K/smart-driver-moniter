import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

from src.detection.face_detector import FaceDetector
from src.detection.eye_analyzer import RIGHT_EYE_INDICES, LEFT_EYE_INDICES
from src.detection.head_pose import NOSE_TIP_IDX


class CameraPanel(ctk.CTkFrame):
    def __init__(self, master, gui_config, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self._config = gui_config
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._display_w = 640
        self._display_h = 480

        self._canvas = ctk.CTkCanvas(
            self, width=self._display_w, height=self._display_h,
            bg="#1a1a2e", highlightthickness=1, highlightbackground="#2a2a4a",
        )
        self._canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self._image_ref = None
        self._frame_info = {}

    def _fit_display_size(self, frame_w, frame_h):
        avail_w = self._canvas.winfo_width() or 640
        avail_h = self._canvas.winfo_height() or 480
        scale = min(avail_w / frame_w, avail_h / frame_h, 1.0)
        return int(frame_w * scale), int(frame_h * scale)

    def update_frame(self, annotated_frame):
        if annotated_frame is None:
            self._show_message("Waiting for camera...")
            return

        h, w = annotated_frame.shape[:2]
        disp_w, disp_h = self._fit_display_size(w, h)

        if (disp_w, disp_h) != (w, h):
            resized = cv2.resize(annotated_frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            resized = annotated_frame

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)
        self._image_ref = photo
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=photo)

    def _show_message(self, msg):
        self._canvas.delete("all")
        self._canvas.create_text(
            self._canvas.winfo_width() // 2 or 320,
            self._canvas.winfo_height() // 2 or 240,
            text=msg, fill="#95a5a6", font=("Segoe UI", 14),
        )

    def draw_status_border(self, color, thickness=6):
        self._canvas.configure(highlightbackground=color, highlightthickness=thickness)