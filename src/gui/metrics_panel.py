import customtkinter as ctk

from src.gui.theme import color_for, STATE_COLORS


class MetricRow(ctk.CTkFrame):
    def __init__(self, master, label, *args, **kwargs):
        super().__init__(master, fg_color="transparent", *args, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self._label = ctk.CTkLabel(self, text=label, font=("Segoe UI", 13), anchor="w")
        self._label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self._value = ctk.CTkLabel(self, text="-", font=("Segoe UI", 13, "bold"), anchor="e",
                                   text_color="#ecf0f1")
        self._value.grid(row=0, column=1, sticky="e", padx=5, pady=2)

    def set(self, text, color=None):
        self._value.configure(text=text)
        if color:
            self._value.configure(text_color=color)

    def set_color(self, color):
        self._value.configure(text_color=color)


class SectionHeader(ctk.CTkLabel):
    def __init__(self, master, text):
        super().__init__(master, text=f"  {text}  ", font=("Segoe UI", 13, "bold"),
                         text_color="#3498db", anchor="w")


class MetricsPanel(ctk.CTkScrollableFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, width=380, *args, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self._rows = {}

        self._status_label = ctk.CTkLabel(
            self, text="SAFE", font=("Segoe UI", 26, "bold"),
            text_color=STATE_COLORS["SAFE"],
        )
        self._status_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))

        self._score_label = ctk.CTkLabel(
            self, text="Safety Score: --/100", font=("Segoe UI", 15),
            text_color="#ecf0f1",
        )
        self._score_label.grid(row=1, column=0, sticky="ew", padx=10)

        self._alarm_status_label = ctk.CTkLabel(
            self, text="NORMAL", font=("Segoe UI", 19, "bold"),
            text_color="#2ecc71",
        )
        self._alarm_status_label.grid(row=2, column=0, sticky="ew", padx=10, pady=(12, 0))

        self._alarm_reason_label = ctk.CTkLabel(
            self, text="", font=("Segoe UI", 13, "bold"),
            text_color="#e74c3c", anchor="w", wraplength=350,
        )
        self._alarm_reason_label.grid(row=3, column=0, sticky="ew", padx=10)

        row = 4

        def add_section(text):
            nonlocal row
            sec = SectionHeader(self, text)
            sec.grid(row=row, column=0, sticky="ew", padx=5, pady=(10, 2))
            self._rows[f"__section_{text}"] = sec
            row += 1

        def add_row(key, label):
            nonlocal row
            r = MetricRow(self, label)
            r.grid(row=row, column=0, sticky="ew", padx=5, pady=1)
            self._rows[key] = r
            row += 1

        add_section("VITALS")
        add_row("ear", "EAR")
        add_row("mar", "MAR")
        add_row("eye_status", "Eye Status")
        add_row("mouth_status", "Mouth Status")
        add_row("head", "Head")
        add_row("pitch_yaw", "Pitch/Yaw")

        add_section("HANDS")
        add_row("left_hand", "Left Hand")
        add_row("right_hand", "Right Hand")

        add_section("PHONE")
        add_row("phone", "Phone")

        add_section("EVENTS")
        add_row("yawns", "Yawns")
        add_row("closures", "Eye Closures")
        add_row("looking_away", "Looking Away")
        add_row("hand_down", "Hand Down")
        add_row("phone_events", "Phone Events")
        add_row("blinks", "Blinks")

        add_section("SESSION")
        add_row("duration", "Duration")
        add_row("avg_score", "Avg Score")
        add_row("max_risk", "Max Risk")

    def set_status(self, state):
        color = STATE_COLORS.get(state, "#f1c40f")
        self._status_label.configure(text=state, text_color=color)

    def set_score(self, score):
        self._score_label.configure(text=f"Safety Score: {score:.0f}/100")

    def set_alarm_status(self, text, color=None):
        self._alarm_status_label.configure(text=text, text_color=color or "#2ecc71")

    def set_alarm_reason(self, text):
        self._alarm_reason_label.configure(text=text)

    def set_metric(self, key, text, color=None):
        row = self._rows.get(key)
        if row:
            row.set(text, color)

    def set_metric_color(self, key, color):
        row = self._rows.get(key)
        if row:
            row.set_color(color)