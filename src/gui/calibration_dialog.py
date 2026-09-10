import customtkinter as ctk

STEPS = [
    "Sit in a normal driving posture.\nLook straight ahead at the camera.",
    "Keep your eyes fully open and still.",
    "Blink your eyes normally 3 times.",
    "Open your mouth naturally and hold it.",
]


class CalibrationDialog(ctk.CTkToplevel):
    def __init__(self, master, on_complete):
        super().__init__(master)
        self.title("Calibration")
        self.geometry("480x360")
        self.resizable(False, False)
        self.transient(master)

        self._on_complete = on_complete
        self._step = 0

        self.grid_columnconfigure(0, weight=1)

        self._header = ctk.CTkLabel(self, text="CALIBRATION", font=("Segoe UI", 22, "bold"))
        self._header.grid(row=0, column=0, pady=(20, 10))

        self._step_label = ctk.CTkLabel(self, text="Step 1/4", font=("Segoe UI", 14),
                                        text_color="#3498db")
        self._step_label.grid(row=1, column=0)

        self._instruction = ctk.CTkLabel(self, text=STEPS[0], font=("Segoe UI", 15),
                                         justify="center", wraplength=400)
        self._instruction.grid(row=2, column=0, padx=20, pady=20)

        self._next_btn = ctk.CTkButton(self, text="Next", command=self._next, width=140,
                                       fg_color="#27ae60", hover_color="#229954")
        self._next_btn.grid(row=3, column=0, pady=10)

        self._cancel_btn = ctk.CTkButton(self, text="Cancel", command=self.destroy, width=140,
                                         fg_color="#e74c3c", hover_color="#c0392b")
        self._cancel_btn.grid(row=4, column=0, pady=(0, 20))

    def _next(self):
        self._step += 1
        if self._step >= len(STEPS):
            self._on_complete()
            self.destroy()
            return
        self._step_label.configure(text=f"Step {self._step + 1}/{len(STEPS)}")
        self._instruction.configure(text=STEPS[self._step])
        if self._step == len(STEPS) - 1:
            self._next_btn.configure(text="Start")