import cv2
import numpy as np
import customtkinter as ctk

from src.gui.camera_panel import CameraPanel
from src.gui.metrics_panel import MetricsPanel
from src.gui.status_bar import StatusBar
from src.gui.theme import set_appearance, STATE_COLORS, color_for
from src.gui.calibration_dialog import CalibrationDialog


class MainWindow(ctk.CTk):
    def __init__(self, app, gui_config):
        set_appearance(gui_config.get("theme", "dark"))
        super().__init__()
        self._app = app
        self._config = gui_config

        width = gui_config.get("width", 1200)
        height = gui_config.get("height", 700)
        self.title("Smart Driver Monitor")
        self.geometry(f"{width}x{height}")
        self.minsize(900, 600)

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._camera_panel = CameraPanel(self, gui_config)
        self._camera_panel.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=5)
        self._metrics_panel = MetricsPanel(self)
        self._metrics_panel.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=5)
        self._status_bar = StatusBar(self)
        self._status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

        self._flash_job = None
        self._alarm_flashing = False

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_header(self):
        header = ctk.CTkFrame(self, height=52, corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(header, text="  Smart Driver Monitor System  ",
                             font=("Segoe UI", 20, "bold"), text_color="#3498db")
        title.grid(row=0, column=0, sticky="w", padx=10, pady=8)

        self._calibrate_btn = ctk.CTkButton(
            header, text="Calibrate", width=110,
            fg_color="#8e44ad", hover_color="#732d91",
            command=self._open_calibration)
        self._calibrate_btn.grid(row=0, column=1, padx=5, pady=8)

        self._start_btn = ctk.CTkButton(
            header, text="Start Monitoring", width=160,
            fg_color="#27ae60", hover_color="#229954",
            command=self._toggle_monitoring)
        self._start_btn.grid(row=0, column=2, padx=5, pady=8)

        self._config_btn = ctk.CTkButton(
            header, text="Config", width=90,
            fg_color="#34495e", hover_color="#2c3e50",
            command=self._app.open_config)
        self._config_btn.grid(row=0, column=3, padx=(5, 15), pady=8)

    def _toggle_monitoring(self):
        running = self._app.toggle_monitoring()
        if running:
            self._start_btn.configure(text="Stop Monitoring", fg_color="#e74c3c", hover_color="#c0392b")
        else:
            self._start_btn.configure(text="Start Monitoring", fg_color="#27ae60", hover_color="#229954")

    def _open_calibration(self):
        CalibrationDialog(self, self._app.run_calibration)

    def _on_close(self):
        self._app.shutdown()
        self.destroy()

    def update_dashboard(self, metrics):
        state = metrics.get("state", "SAFE")
        self._metrics_panel.set_status(state)
        self._metrics_panel.set_score(metrics.get("safety_score", 100))

        self._metrics_panel.set_metric("ear", f"{metrics.get('ear', 0):.2f}")
        self._metrics_panel.set_metric("mar", f"{metrics.get('mar', 0):.2f}")

        eye = metrics.get("eye_status", "OPEN")
        eye_color = color_for(eye)
        self._metrics_panel.set_metric("eye_status", eye, eye_color)

        mouth = metrics.get("mouth_status", "NORMAL")
        mouth_color = color_for(mouth)
        self._metrics_panel.set_metric("mouth_status", mouth, mouth_color)

        head = metrics.get("head_direction", "FORWARD")
        head_color = color_for(head)
        self._metrics_panel.set_metric("head", head, head_color)

        pitch = metrics.get("pitch", 0)
        yaw = metrics.get("yaw", 0)
        self._metrics_panel.set_metric("pitch_yaw", f"{pitch:+.0f}/{yaw:+.0f}")

        lh = metrics.get("left_hand", "NOT DETECTED")
        rh = metrics.get("right_hand", "NOT DETECTED")
        self._metrics_panel.set_metric("left_hand", lh, color_for(lh))
        self._metrics_panel.set_metric("right_hand", rh, color_for(rh))

        phone = metrics.get("phone_detected", False)
        if phone:
            self._metrics_panel.set_metric("phone", "DETECTED", "#e74c3c")
        else:
            self._metrics_panel.set_metric("phone", "NOT DETECTED", "#95a5a6")

        self._metrics_panel.set_metric("yawns", str(metrics.get("yawns", 0)))
        self._metrics_panel.set_metric("closures", str(metrics.get("closures", 0)))
        self._metrics_panel.set_metric("looking_away", str(metrics.get("looking_away", 0)))
        self._metrics_panel.set_metric("hand_down", str(metrics.get("hand_down", 0)))
        self._metrics_panel.set_metric("phone_events", str(metrics.get("phone_events", 0)))
        self._metrics_panel.set_metric("blinks", str(metrics.get("blinks", 0)))

        duration = metrics.get("duration_seconds", 0)
        mins, secs = divmod(int(duration), 60)
        self._metrics_panel.set_metric("duration", f"{mins:02d}:{secs:02d}")
        self._metrics_panel.set_metric("avg_score", f"{metrics.get('avg_score', 100):.0f}/100")
        self._metrics_panel.set_metric("max_risk", metrics.get("max_risk", "SAFE"),
                                       color_for(metrics.get("max_risk", "SAFE")))

        state_color = STATE_COLORS.get(state, "#f1c40f")
        self._camera_panel.draw_status_border(state_color)

        alert = metrics.get("alert")
        if alert:
            self._status_bar.set_alert(alert[1], state_color)
        else:
            self._status_bar.set_alert("")

        self._update_alarm_ui(metrics.get("alarm") or {})

        self._status_bar.set_fps(metrics.get("fps", 0))
        self._status_bar.set_time(metrics.get("duration_seconds", 0))
        self._status_bar.set_face_status(metrics.get("face_detected", False))

    def _update_alarm_ui(self, alarm):
        active = alarm.get("active", False)
        sounding = alarm.get("sounding", False)
        reasons = alarm.get("reasons", [])

        if sounding:
            text, color = "ALARM ACTIVE", "#ff3333"
        elif any(r.startswith("EYES") for r in reasons):
            text, color = "DROWSINESS DETECTED", "#e74c3c"
        elif "YAWNING DETECTED" in reasons:
            text, color = "YAWNING DETECTED", "#e67e22"
        elif "PHONE DETECTED" in reasons:
            text, color = "PHONE DETECTED", "#e74c3c"
        else:
            text, color = "NORMAL", "#2ecc71"

        self._metrics_panel.set_alarm_status(text, color)
        reason_text = ("⚠ ALERT: " + ", ".join(reasons)) if reasons else ""
        self._metrics_panel.set_alarm_reason(reason_text)
        if reason_text:
            self._status_bar.set_alert(reason_text, "#e74c3c")

        self._alarm_flashing = bool(sounding and active)
        if self._alarm_flashing:
            self._start_alarm_flash()
        else:
            self._stop_alarm_flash()
            if reasons:
                self._camera_panel.draw_status_border(color)

    def _start_alarm_flash(self):
        if self._flash_job is None:
            self._flash_on = False
            self._flash_job = self.after(400, self._flash_tick)

    def _flash_tick(self):
        self._flash_job = None
        self._flash_on = not self._flash_on
        color = "#ff3333" if self._flash_on else "#3d0a0a"
        self._camera_panel.draw_status_border(color)
        self._metrics_panel.set_alarm_status("ALARM ACTIVE", color)
        if self._alarm_flashing:
            self._flash_job = self.after(400, self._flash_tick)

    def _stop_alarm_flash(self):
        if self._flash_job is not None:
            self.after_cancel(self._flash_job)
            self._flash_job = None

    def show_session_summary(self, summary_text):
        from src.gui.session_dialog import SessionDialog
        SessionDialog(self, summary_text)