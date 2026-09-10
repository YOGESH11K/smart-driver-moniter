import os
import time
import logging

logger = logging.getLogger(__name__)

try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False
    logger.warning("pygame not available. Sound alerts disabled.")


ALERT_MESSAGES = {
    "SAFE": "Driver appears alert.",
    "CAUTION": "Please stay attentive.",
    "ATTENTION REQUIRED": "Keep your eyes on the road.",
    "DROWSY / DISTRACTED": "Drowsiness detected. Please take a break.",
    "DANGER": "DANGER! Critical drowsiness level!",
    "PHONE": "Avoid using your phone while driving!",
}


class AlertManager:
    def __init__(self, cooldown_seconds=5, sound_enabled=True,
                 sound_path="assets/sounds/", visual_duration=3):
        self._cooldown = cooldown_seconds
        self._sound_enabled = sound_enabled and HAS_PYGAME
        self._sound_path = sound_path
        self._visual_duration = visual_duration

        self._last_alert_time = 0
        self._current_alert = None
        self._alert_expires = 0
        self._sounds = {}
        self._load_sounds()

    def _load_sounds(self):
        if not self._sound_enabled:
            return
        for level in ["low", "medium", "high"]:
            path = os.path.join(self._sound_path, f"alert_{level}.wav")
            if os.path.exists(path):
                try:
                    self._sounds[level] = pygame.mixer.Sound(path)
                except Exception as e:
                    logger.warning("Failed to load sound %s: %s", path, e)
            else:
                self._sounds[level] = None
        if not any(self._sounds.values()):
            self._sound_enabled = False

    def check_and_alert(self, state, phone_detected=False):
        now = time.time()
        alert_to_show = None

        if phone_detected:
            alert_to_show = ("PHONE", ALERT_MESSAGES["PHONE"])
        elif state != "SAFE":
            alert_to_show = (state, ALERT_MESSAGES.get(state, "Unknown alert"))

        if alert_to_show is None:
            if now > self._alert_expires:
                self._current_alert = None
            return self._current_alert

        level, message = alert_to_show
        if now - self._last_alert_time < self._cooldown:
            return self._current_alert

        self._current_alert = (level, message)
        self._alert_expires = now + self._visual_duration
        self._last_alert_time = now

        self._play_sound(level)

        return self._current_alert

    def _play_sound(self, level):
        if not self._sound_enabled:
            return
        sound_level = {
            "SAFE": "low",
            "CAUTION": "low",
            "ATTENTION REQUIRED": "medium",
            "DROWSY / DISTRACTED": "medium",
            "DANGER": "high",
            "PHONE": "medium",
        }.get(level, "low")

        sound = self._sounds.get(sound_level)
        if sound:
            try:
                sound.play()
            except Exception as e:
                logger.warning("Failed to play sound: %s", e)

    @property
    def current_alert(self):
        return self._current_alert

    @property
    def sound_enabled(self):
        return self._sound_enabled

    @sound_enabled.setter
    def sound_enabled(self, value):
        self._sound_enabled = value and HAS_PYGAME

    def reset(self):
        self._last_alert_time = 0
        self._current_alert = None
        self._alert_expires = 0
