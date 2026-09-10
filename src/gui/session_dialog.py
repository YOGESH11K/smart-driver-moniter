import customtkinter as ctk


class SessionDialog(ctk.CTkToplevel):
    def __init__(self, master, summary_text):
        super().__init__(master)
        self.title("Session Summary")
        self.geometry("420x520")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        self.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(self, text="SESSION SUMMARY", font=("Segoe UI", 20, "bold"))
        title.grid(row=0, column=0, pady=(20, 10))

        text = ctk.CTkTextbox(self, width=360, height=340, font=("Consolas", 12))
        text.grid(row=1, column=0, padx=20, pady=10)
        text.insert("1.0", summary_text)
        text.configure(state="disabled")

        close_btn = ctk.CTkButton(self, text="Close", command=self.destroy, width=120,
                                  fg_color="#3498db", hover_color="#2980b9")
        close_btn.grid(row=2, column=0, pady=(0, 20))