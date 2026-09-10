import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionEvent:
    timestamp: float
    event_type: str
    severity: str
    metric_value: float = 0.0
    duration_seconds: float = 0.0


@dataclass
class SessionSummary:
    session_id: Optional[int] = None
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    total_yawns: int = 0
    total_eye_closures: int = 0
    total_distraction_events: int = 0
    total_phone_events: int = 0
    total_hand_down_events: int = 0
    total_looking_away_events: int = 0
    avg_safety_score: float = 0.0
    max_risk_level: str = "SAFE"
    events: list = field(default_factory=list)
