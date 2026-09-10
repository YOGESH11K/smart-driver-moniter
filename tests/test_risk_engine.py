import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.risk_engine import RiskEngine


def test_safe_initial():
    engine = RiskEngine()
    assert engine.state == "SAFE"
    assert engine.safety_score >= 80


def test_eye_closure_raises_risk():
    engine = RiskEngine()
    for i in range(15):
        engine.update(eye_closure=True, eye_closure_frames=60)
    assert engine.safety_score < 80
    assert engine.state != "SAFE"


def test_phone_detection_raises_risk():
    engine = RiskEngine()
    for i in range(15):
        engine.update(phone_detected=True)
    assert engine.safety_score < 80


def test_danger_state():
    engine = RiskEngine()
    for i in range(15):
        engine.update(eye_closure=True, eye_closure_frames=120,
                      yawn=True, yawn_frames=60,
                      looking_away=True, away_frames=80,
                      phone_detected=True,
                      hand_down_left=True)
    assert engine.state == "DANGER"
    assert engine.safety_score < 20


def test_recovery():
    engine = RiskEngine()
    for i in range(15):
        engine.update(eye_closure=True, eye_closure_frames=120)
    assert engine.state != "SAFE"
    for i in range(60):
        engine.update()
    assert engine.state == "SAFE"


def test_state_hysteresis():
    engine = RiskEngine()
    for i in range(5):
        engine.update(eye_closure=True, eye_closure_frames=60)
    assert engine.state == "SAFE"


def test_max_risk_tracking():
    engine = RiskEngine()
    for i in range(15):
        engine.update(eye_closure=True, eye_closure_frames=120,
                      looking_away=True, away_frames=80)
    assert engine.max_risk_level != "SAFE"


if __name__ == "__main__":
    test_safe_initial()
    test_eye_closure_raises_risk()
    test_phone_detection_raises_risk()
    test_danger_state()
    test_recovery()
    test_state_hysteresis()
    test_max_risk_tracking()
    print("All risk engine tests passed.")