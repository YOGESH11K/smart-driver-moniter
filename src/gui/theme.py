import customtkinter as ctk
import tkinter as tk

STATE_COLORS = {
    "SAFE": "#2ecc71",
    "CAUTION": "#f1c40f",
    "ATTENTION REQUIRED": "#e67e22",
    "DROWSY / DISTRACTED": "#e74c3c",
    "DANGER": "#8b0000",
}

STATUS_COLORS = {
    "OPEN": "#2ecc71",
    "CLOSED": "#e74c3c",
    "NORMAL": "#2ecc71",
    "YAWNING": "#e74c3c",
    "FORWARD": "#2ecc71",
    "LEFT": "#e67e22",
    "RIGHT": "#e67e22",
    "UP": "#e67e22",
    "DOWN": "#e67e22",
    "SAFE": "#2ecc71",
    "DOWN": "#e74c3c",
    "NOT DETECTED": "#95a5a6",
    "DETECTED": "#e74c3c",
    "NO FACE": "#95a5a6",
    "UNKNOWN": "#95a5a6",
}


def color_for(key, fallback="#f1c40f"):
    return STATUS_COLORS.get(key, fallback)


def set_appearance(theme="dark"):
    ctk.set_appearance_mode(theme)
    ctk.set_default_color_theme("blue")