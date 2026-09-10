"""Smart Driver Monitor - Web API (backend for Render).

Reuses the same weighted risk scoring used by the desktop app.
The browser frontend computes EAR / MAR / head pose with MediaPipe.js
and sends the metrics here; this service classifies the risk state.
"""

import time
import uuid
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from risk_engine import RiskEngine, STATES

app = FastAPI(
    title="Smart Driver Monitor API",
    description="Risk classification API for the Smart Driver Monitoring system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

started_at = time.time()

DEFAULT_CONFIG = {
    "ear_threshold": 0.21,
    "ear_consecutive_frames": 45,
    "mar_threshold": 0.50,
    "yawn_consecutive_frames": 30,
    "pitch_threshold": 15.0,
    "yaw_threshold": 20.0,
    "away_consecutive_frames": 20,
    "weights": {
        "eye_closure": 35,
        "yawn": 20,
        "head_away": 25,
        "hand_down": 10,
        "phone": 30,
    },
    "state_thresholds": {
        "safe_min": 80,
        "caution_min": 60,
        "attention_min": 40,
        "drowsy_min": 20,
    },
    "states": STATES,
}


class Metrics(BaseModel):
    session_id: str = Field(default_factory=lambda: "-")
    ear: float = Field(0.0, ge=0, le=1)
    mar: float = Field(0.0, ge=0, le=1)
    pitch: float = Field(0.0, ge=-90, le=90)
    yaw: float = Field(0.0, ge=-90, le=90)
    roll: float = Field(0.0, ge=-90, le=90)
    eye_closed: bool = False
    eye_closure_frames: int = Field(0, ge=0)
    yawn: bool = False
    yawn_frames: int = Field(0, ge=0)
    looking_away: bool = False
    away_frames: int = Field(0, ge=0)
    hand_down: bool = False
    phone: bool = False
    face_present: bool = True


class SessionResult(BaseModel):
    session_id: str
    state: str
    safety_score: float
    risk_score: float
    reasons: list
    server_ts: float


_sessions: Dict[str, RiskEngine] = {}
_LAST_GLOBAL_RESULT: dict = {}


def _cleanup_sessions():
    if len(_sessions) < 100:
        return
    # drop the oldest session, keep a simple LRU-ish guard
    oldest = None
    for sid in list(_sessions.keys()):
        oldest = sid
        break
    if oldest:
        _sessions.pop(oldest, None)


def _get_engine(session_id: str) -> RiskEngine:
    if session_id not in _sessions:
        _sessions[session_id] = RiskEngine()
        _cleanup_sessions()
    return _sessions[session_id]


@app.get("/")
def root():
    return {
        "name": "Smart Driver Monitor API",
        "health": "ok",
        "uptime_seconds": round(time.time() - started_at, 1),
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "active_sessions": len(_sessions), "uptime_seconds": round(time.time() - started_at, 1)}


@app.get("/api/config")
def config():
    return DEFAULT_CONFIG


@app.get("/api/last")
def last():
    return _LAST_GLOBAL_RESULT


@app.get("/api/states")
def states():
    return {"states": STATES}


@app.post("/api/analyze", response_model=SessionResult)
def analyze(m: Metrics):
    session_id = m.session_id if m.session_id and m.session_id != "-" else "default"
    engine = _get_engine(session_id)

    eye_closure = m.eye_closed
    yawn = m.yawn
    looking_away = m.looking_away

    engine.update(
        eye_closure=eye_closure,
        eye_closure_frames=m.eye_closure_frames,
        yawn=yawn,
        yawn_frames=m.yawn_frames,
        looking_away=looking_away,
        away_frames=m.away_frames,
        hand_down_left=m.hand_down,
        hand_down_right=False,
        phone_detected=m.phone,
    )

    reasons = []
    if eye_closure:
        reasons.append("EYES CLOSED")
    if yawn:
        reasons.append("YAWNING")
    if looking_away:
        reasons.append("LOOKING AWAY")
    if m.hand_down and not m.face_present:
        reasons.append("HANDS OFF WHEEL")
    if m.phone:
        reasons.append("PHONE DETECTED")

    result = {
        "session_id": session_id,
        "state": engine.state,
        "safety_score": round(engine.safety_score, 2),
        "risk_score": round(engine.risk_score, 2),
        "reasons": reasons,
        "server_ts": time.time(),
    }
    _LAST_GLOBAL_RESULT.update(result)
    return result


@app.post("/api/reset")
def reset(session_id: str = "default"):
    sid = session_id if session_id else "default"
    if sid in _sessions:
        _sessions[sid].reset()
    return {"ok": True, "session_id": sid}