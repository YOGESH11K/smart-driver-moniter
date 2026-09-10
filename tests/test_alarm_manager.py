"""Tests for the real-time alarm manager.

Uses a fake always-on player so audio is never actually played here,
and a fake clock so no wall time is needed.
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.alarm_manager import AlarmManager
from src.monitoring.alarm_manager import (
    REASON_EYE_CLOSED,
    REASON_EYES_OFF,
    REASON_YAWN,
    REASON_PHONE,
)


class FakePlayer:
    def __init__(self):
        self.play_calls = []
        self._playing = False

    def play(self, duration):
        self.play_calls.append(round(duration, 3))
        self._playing = True

    def stop(self):
        self._playing = False

    def is_playing(self):
        return self._playing

    def close(self):
        self.stop()


class Clock:
    def __init__(self, start=1_000_000.0):
        self.t = start

    def now(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make_manager(**kwargs):
    kwargs.setdefault("cooldown_seconds", 10)
    kwargs.setdefault("alarm_duration_seconds", 8)
    kwargs.setdefault("yawn_cooldown_seconds", 8)
    kwargs.setdefault("eyes_off_threshold_seconds", 1.5)
    player = FakePlayer()
    manager = AlarmManager(sound_enabled=True, player=player, **kwargs)
    return manager, player


def feed(manager, clock, *, eye_closed=False, eyes_off=False, yawn=False,
         phone=False, frames=1, step=0.1):
    state = None
    for _ in range(frames):
        state = manager.update(eye_closed=eye_closed, eyes_off=eyes_off,
                               yawn=yawn, phone=phone)
        clock.advance(step)
    return state


@patch("src.monitoring.alarm_manager.time.time")
def test_eye_closure_triggers_once_and_stops_on_clear(mock_time):
    clock = Clock()
    mock_time.side_effect = clock.now
    manager, player = make_manager()

    state1 = feed(manager, clock, eye_closed=True)
    state2 = feed(manager, clock, eye_closed=True)
    assert state1["active"] and REASON_EYE_CLOSED in state1["reasons"]
    assert state2["sounding"], "sound should start once the event is confirmed"

    assert len(player.play_calls) == 1

    state3 = feed(manager, clock, eye_closed=True, frames=10)
    assert state3["sounding"]
    assert len(player.play_calls) == 1, "a persistent condition must not re-trigger"

    state4 = feed(manager, clock, eye_closed=False)
    assert state4["active"] is False
    assert state4["sounding"] is False
    assert player.is_playing() is False, "alarm must stop when the condition clears"


@patch("src.monitoring.alarm_manager.time.time")
def test_single_frame_blip_does_not_trigger(mock_time):
    clock = Clock()
    mock_time.side_effect = clock.now
    manager, player = make_manager()

    feed(manager, clock, eye_closed=True, frames=1)
    feed(manager, clock, eye_closed=False, frames=3)

    assert len(player.play_calls) == 0, "a single noisy frame must not start the alarm"


@patch("src.monitoring.alarm_manager.time.time")
def test_yawn_triggers_and_does_not_repeat_for_same_yawn(mock_time):
    clock = Clock()
    mock_time.side_effect = clock.now
    manager, player = make_manager()

    feed(manager, clock, yawn=True, frames=3, step=0.1)
    assert len(player.play_calls) == 1

    feed(manager, clock, yawn=True, frames=30, step=0.1)
    assert len(player.play_calls) == 1, "same persistent yawn must not re-trigger"

    feed(manager, clock, yawn=False, frames=3, step=0.1)
    assert player.is_playing() is False


@patch("src.monitoring.alarm_manager.time.time")
def test_phone_triggers(mock_time):
    clock = Clock()
    mock_time.side_effect = clock.now
    manager, player = make_manager()

    feed(manager, clock, yawn=False, phone=True, frames=3, step=0.1)
    assert REASON_PHONE in manager.state["reasons"]
    assert len(player.play_calls) == 1


@patch("src.monitoring.alarm_manager.time.time")
def test_multiple_simultaneous_conditions_play_one_sound_with_all_reasons(mock_time):
    clock = Clock()
    mock_time.side_effect = clock.now
    manager, player = make_manager()

    feed(manager, clock, eye_closed=True, phone=True, frames=3, step=0.1)

    state = manager.state
    assert state["active"]
    assert set(state["reasons"]) == {REASON_EYE_CLOSED, REASON_PHONE}
    assert len(player.play_calls) == 1, "only one alarm sound for simultaneous conditions"
    assert "EYES CLOSED" in manager.alarm_message
    assert "PHONE DETECTED" in manager.alarm_message


@patch("src.monitoring.alarm_manager.time.time")
def test_cooldown_blocks_rapid_re_trigger(mock_time):
    clock = Clock()
    mock_time.side_effect = clock.now
    manager, player = make_manager(cooldown_seconds=10)

    feed(manager, clock, eye_closed=True, frames=3, step=0.1)
    assert len(player.play_calls) == 1
    feed(manager, clock, eye_closed=False, frames=3, step=0.1)

    # New event too soon after the previous one: suppressed by cooldown.
    feed(manager, clock, yawn=True, frames=3, step=0.1)
    assert len(player.play_calls) == 1, "cooldown must suppress a rapid re-trigger"
    assert manager.state["cooldown_remaining"] > 0

    # After the cooldown expires the same still-active condition re-arms.
    clock.advance(12)
    feed(manager, clock, yawn=True, frames=3, step=0.1)
    assert len(player.play_calls) == 2, "alarm re-arms once the cooldown has elapsed"


@patch("src.monitoring.alarm_manager.time.time")
def test_eyes_off_requires_time_threshold(mock_time):
    clock = Clock()
    mock_time.side_effect = clock.now
    manager, player = make_manager(eyes_off_threshold_seconds=1.5)

    # Short face drop-out: under threshold, no alarm.
    feed(manager, clock, eyes_off=True, frames=5, step=0.2)
    assert len(player.play_calls) == 0

    # Long face drop-out: threshold reached, alarm sounds.
    feed(manager, clock, eyes_off=True, frames=10, step=0.2)
    assert REASON_EYES_OFF in manager.state["reasons"]
    assert len(player.play_calls) == 1

    feed(manager, clock, eyes_off=False, frames=3, step=0.2)
    assert player.is_playing() is False


@patch("src.monitoring.alarm_manager.time.time")
def test_reset_clears_everything(mock_time):
    clock = Clock()
    mock_time.side_effect = clock.now
    manager, player = make_manager()

    feed(manager, clock, phone=True, frames=3, step=0.1)
    assert len(player.play_calls) == 1
    manager.reset()
    assert manager.state["active"] is False
    assert manager.state["reasons"] == []
    assert player.is_playing() is False


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception:
                failures += 1
                print(f"  FAIL  {name}")
                traceback.print_exc()
    print(f"\n{len([n for n in globals() if n.startswith('test_')]) - failures} passed, {failures} failed.")
    sys.exit(1 if failures else 0)