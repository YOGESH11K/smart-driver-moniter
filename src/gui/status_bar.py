import customtkinter as ctk


class StatusBar(ctk.CTkFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, height=34, *args, **kwargs)
        self.grid_columnconfigure(1, weight=1)

        self._alert_label = ctk.CTkLabel(
            self, text="", font=("Segoe UI", 12, "bold"), anchor="w",
            text_color="#ecf0f1",
        )
        self._alert_label.grid(row=0, column=0, sticky="w", padx=12, pady=4)

        self._fps_label = ctk.CTkLabel(
            self, text="FPS: --", font=("Segoe UI", 11), text_color="#95a5a6",
        )
        self._fps_label.grid(row=0, column=2, sticky="e", padx=8, pady=4)

        self._time_label = ctk.CTkLabel(
            self, text="00:00", font=("Segoe UI", 11), text_color="#95a5a6",
        )
        self._time_label.grid(row=0, column=3, sticky="e", padx=12, pady=4)

        self._frame_ctr = ctk.CTkLabel(
            self, text="", font=("Segoe UI", 11), text_color="#95a5a6",
        )
        self._frame_ctr.grid(row=0, column=1, sticky="w", padx=8, pady=4)

    def set_alert(self, message, color=None):
        if message:
            self._alert_label.configure(text=f"{message}", text_color=color or "#e74c3c")
        else:
            self._alert_label.configure(text="")

    def set_fps(self, fps):
        self._fps_label.configure(text=f"FPS: {fps:.0f}")

    def set_time(self, elapsed_seconds):
        mins, secs = divmod(int(elapsed_seconds), 60)
        self._time_label.configure(text=f"{mins:02d}:{secs:02d}")

    def set_face_status(self, has_face):
        if has_face:
            self._frame_ctr.configure(text="Face: Detect", text_color="#2ecc71")
        else:
            self._frame_ctr.configure(text="Face: Not Detected", text_color="#e74c3c")