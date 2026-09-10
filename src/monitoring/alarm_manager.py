"""Non-blocking audible alarm for the driver monitoring system.

Separate from the detection logic. Receives high-level danger booleans
(eye closure, eyes off the road, yawning, phone use) every frame, keeps
the set of active reasons, and plays a single alarm sound on a background
thread so video processing never blocks.

Behaviour:
  * A danger signal must persist for 2 consecutive update calls before it can
    sound (the frame-level consecutive-frame filters in the detectors are the
    primary anti-false-trigger layer; this streak is a second confirmation).
  * The sound triggers on the *first confirmed* dangerous event only; it never
    continuously re-triggers on a persistent condition.
  * A global cooldown prevents rapid re-triggering (also covers the same yawn).
  * A separate yawn cooldown debounces the same yawn.
  * Multiple simultaneous conditions merge into one sound; every reason is
    still reported to the UI.
  * The alarm stops immediately when the dangerous condition clears.
"""

import os
import time
import threading
import logging

logger = logging.getLogger(__name__)

try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    HAS_PYGAME = True
    HAS_WINSOUND = False
except Exception:
    HAS_PYGAME = False
    try:
        import winsound
        HAS_WINSOUND = True
    except Exception:
        HAS_WINSOUND = False

REASON_EYE_CLOSED = "EYES CLOSED"
REASON_EYES_OFF = "EYES OFF THE ROAD"
REASON_YAWN = "YAWNING DETECTED"
REASON_PHONE = "PHONE DETECTED"

_MIN_CONFIRM_UPDATES = 2


class _AlarmPlayer(threading.Thread):
    """Background thread that plays at most one alarm sound at a time."""

    def __init__(self, sound_path, sound_file, sound_enabled):
        super().__init__(daemon=True)
        self._file = os.path.join(sound_path, sound_file) if sound_path else sound_file
        self._enabled = sound_enabled and (HAS_PYGAME or HAS_WINSOUND)
        self._sound = None
        self._cv = threading.Condition()
        self._active = False
        self._duration = 0.0
        self._stopped = False
        self._load()
        self.start()

    def _load(self):
        if not self._enabled:
            return
        if not self._file or not os.path.exists(self._file):
            logger.warning("Alarm sound file not found, alarm will be silent: %s", self._file)
            self._sound = None
            return
        try:
            if HAS_PYGAME:
                self._sound = pygame.mixer.Sound(self._file)
            else:
                self._sound = self._file
        except Exception as e:
            logger.warning("Failed to load alarm sound %s: %s", self._file, e)
            self._sound = None

    def play(self, duration):
        if not self._enabled or self._sound is None:
            return
        with self._cv:
            self._duration = max(0.1, float(duration))
            self._active = True
            self._cv.notify_all()

    def stop(self):
        with self._cv:
            self._active = False
            self._cv.notify_all()

    def is_playing(self):
        with self._cv:
            return self._active

    def close(self):
        with self._cv:
            self._stopped = True
            self._active = False
            self._cv.notify_all()

    def _play_sound(self):
        try:
            if HAS_PYGAME:
                self._sound.play(loops=-1)
            elif HAS_WINSOUND:
                winsound.PlaySound(self._file, winsound.SND_ASYNC | winsound.SND_LOOP)
        except Exception as e:
            logger.warning("Failed to start alarm sound: %s", e)

    def _stop_sound(self):
        if self._sound is None:
            return
        try:
            if HAS_PYGAME:
                self._sound.stop()
            elif HAS_WINSOUND:
                winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception as e:
            logger.warning("Failed to stop alarm sound: %s", e)

    def run(self):
        while True:
            with self._cv:
                while not self._active and not self._stopped:
                    self._cv.wait()
                if self._stopped:
                    self._stop_sound()
                    return
                duration = self._duration
                self._active = False

            self._play_sound()

            with self._cv:
                self._cv.wait(timeout=duration)
                self._stop_sound()


class AlarmManager:
    """Tracks danger conditions and triggers/merges a single audible alarm."""

    def __init__(self, *, sound_enabled=True, sound_path="assets/sounds/",
                 sound_file="alarm.wav", cooldown_seconds=10.0,
                 alarm_duration_seconds=8.0, yawn_cooldown_seconds=8.0,
                 eyes_off_threshold_seconds=1.5, player=None):
        self._cooldown = float(cooldown_seconds)
        self._alarm_duration = float(alarm_duration_seconds)
        self._yawn_cooldown = float(yawn_cooldown_seconds)
        self._eyes_off_threshold = float(eyes_off_threshold_seconds)

        self._active_reasons = []
        self._active_streak = 0
        self._yawn_was_active = False
        self._eyes_off_since = None
        self._last_trigger_time = 0.0
        self._last_yawn_trigger_time = 0.0
        self._pending_trigger = False

        self._player = player if player is not None else _AlarmPlayer(
            sound_path, sound_file, sound_enabled
        )

    def update(self, *, eye_closed=False, eyes_off=False, yawn=False, phone=False):
        """Feed current frame signals. Returns the current alarm state dict."""
        now = time.time()
        reasons = []

        if eye_closed:
            reasons.append(REASON_EYE_CLOSED)

        if eyes_off:
            if self._eyes_off_since is None:
                self._eyes_off_since = now
            elif now - self._eyes_off_since >= self._eyes_off_threshold:
                reasons.append(REASON_EYES_OFF)
        else:
            self._eyes_off_since = None

        if yawn:
            reasons.append(REASON_YAWN)

        if phone:
            reasons.append(REASON_PHONE)

        active = bool(reasons)
        self._active_reasons = reasons

        if active:
            self._active_streak += 1
        else:
            self._active_streak = 0
            self._pending_trigger = False
            self._player.stop()

        yawn_new = (REASON_YAWN in reasons) and not self._yawn_was_active
        self._yawn_was_active = REASON_YAWN in reasons

        cooldown_ok = (now - self._last_trigger_time) >= self._cooldown
        yawn_blocked = yawn_new and (now - self._last_yawn_trigger_time) < self._yawn_cooldown

        trigger_allowed = cooldown_ok and not yawn_blocked

        if active and self._active_streak == _MIN_CONFIRM_UPDATES and trigger_allowed:
            self._pending_trigger = False
            self._trigger(now, REASON_YAWN in reasons)
        elif active and self._pending_trigger and trigger_allowed:
            self._pending_trigger = False
            self._trigger(now, REASON_YAWN in reasons)
        elif active and self._active_streak == _MIN_CONFIRM_UPDATES and not trigger_allowed:
            self._pending_trigger = True

        return self.state

    def _trigger(self, now, yawn_event):
        self._last_trigger_time = now
        if yawn_event:
            self._last_yawn_trigger_time = now
        self._player.play(self._alarm_duration)
        logger.info("ALARM triggered: %s", ", ".join(self._active_reasons))

    @property
    def state(self):
        return {
            "active": bool(self._active_reasons),
            "sounding": self._player.is_playing(),
            "reasons": list(self._active_reasons),
            "cooldown_remaining": max(0.0, self._cooldown - (time.time() - self._last_trigger_time)),
        }

    @property
    def alarm_message(self):
        if not self._active_reasons:
            return ""
        return "⚠ ALERT: " + ", ".join(self._active_reasons)

    def reset(self):
        self._active_reasons = []
        self._active_streak = 0
        self._yawn_was_active = False
        self._eyes_off_since = None
        self._last_trigger_time = 0.0
        self._last_yawn_trigger_time = 0.0
        self._pending_trigger = False
        self._player.stop()

    def close(self):
        if hasattr(self._player, "close"):
            self._player.close()
        elif hasattr(self._player, "stop"):
            self._player.stop()