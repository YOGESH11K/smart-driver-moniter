import time

STATES = ["SAFE", "CAUTION", "ATTENTION REQUIRED", "DROWSY / DISTRACTED", "DANGER"]


class RiskEngine:
    def __init__(self, weights=None, state_thresholds=None, transition_frames=None):
        self._weights = weights or {
            "eye_closure": 35,
            "yawn": 20,
            "head_away": 25,
            "hand_down": 10,
            "phone": 30,
        }
        self._thresholds = state_thresholds or {
            "safe_min": 80,
            "caution_min": 60,
            "attention_min": 40,
            "drowsy_min": 20,
        }
        self._transitions = transition_frames or {
            "to_caution": 10,
            "to_attention": 15,
            "to_danger": 10,
            "recovery_to_caution": 40,
            "recovery_to_safe": 50,
        }

        self._risk_score = 0
        self._safety_score = 100
        self._state = "SAFE"
        self._state_counter = 0
        self._max_risk_level = "SAFE"
        self._total_risk_events = 0
        self._event_active = False

    def update(self, *, eye_closure=False, eye_closure_frames=0,
               yawn=False, yawn_frames=0,
               looking_away=False, away_frames=0,
               hand_down_left=False, hand_down_right=False,
               phone_detected=False):

        risk_points = 0.0

        if eye_closure:
            duration_factor = min(eye_closure_frames / 60.0, 1.0)
            risk_points += self._weights["eye_closure"] * duration_factor

        if yawn:
            duration_factor = min(yawn_frames / 45.0, 1.0)
            risk_points += self._weights["yawn"] * duration_factor

        if looking_away:
            duration_factor = min(away_frames / 40.0, 1.0)
            risk_points += self._weights["head_away"] * duration_factor

        if hand_down_left or hand_down_right:
            risk_points += self._weights["hand_down"] * 0.8

        if phone_detected:
            risk_points += self._weights["phone"] * 0.9

        max_possible = sum(self._weights.values())
        normalized_risk = min((risk_points / max_possible) * 100, 100) if max_possible > 0 else 0
        self._risk_score = normalized_risk
        self._safety_score = max(0, 100 - normalized_risk)

        new_state = self._compute_state(self._safety_score)

        if new_state != self._state:
            self._state_counter += 1
            needed = self._get_transition_count(self._state, new_state)
            if self._state_counter >= needed:
                old_state = self._state
                self._state = new_state
                self._state_counter = 0
                if self._state_index(new_state) > self._state_index(old_state):
                    self._total_risk_events += 1
        else:
            self._state_counter = 0

        risk_idx = self._state_index(self._state)
        max_risk_idx = self._state_index(self._max_risk_level)
        if risk_idx > max_risk_idx:
            self._max_risk_level = self._state

        if self._state != "SAFE" and not self._event_active:
            self._event_active = True
        elif self._state == "SAFE":
            self._event_active = False

    def _compute_state(self, safety_score):
        if safety_score >= self._thresholds["safe_min"]:
            return "SAFE"
        if safety_score >= self._thresholds["caution_min"]:
            return "CAUTION"
        if safety_score >= self._thresholds["attention_min"]:
            return "ATTENTION REQUIRED"
        if safety_score >= self._thresholds["drowsy_min"]:
            return "DROWSY / DISTRACTED"
        return "DANGER"

    def _get_transition_count(self, from_state, to_state):
        from_idx = self._state_index(from_state)
        to_idx = self._state_index(to_state)
        if to_idx > from_idx:
            if to_idx == 1:
                return self._transitions["to_caution"]
            if to_idx == 2:
                return self._transitions["to_attention"]
            if to_idx >= 4:
                return self._transitions["to_danger"]
        else:
            if to_idx <= 1:
                return self._transitions["recovery_to_safe"]
            if to_idx <= 2:
                return self._transitions["recovery_to_caution"]
        return 5

    @staticmethod
    def _state_index(state):
        for i, s in enumerate(STATES):
            if s == state:
                return i
        return 0

    @property
    def safety_score(self):
        return self._safety_score

    @property
    def risk_score(self):
        return self._risk_score

    @property
    def state(self):
        return self._state

    @property
    def max_risk_level(self):
        return self._max_risk_level

    @property
    def total_risk_events(self):
        return self._total_risk_events

    def reset(self):
        self._risk_score = 0
        self._safety_score = 100
        self._state = "SAFE"
        self._state_counter = 0
        self._max_risk_level = "SAFE"
        self._total_risk_events = 0
        self._event_active = False