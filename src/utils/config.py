import os
import yaml
import copy


_DEFAULTS = {
    "camera": {"index": 0, "width": 640, "height": 480, "target_fps": 30},
    "detection": {
        "face": {
            "model_path": "models/face_landmarker.task",
            "num_faces": 1,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        },
        "hand": {
            "model_path": "models/hand_landmarker.task",
            "max_hands": 2,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "safe_region_y_max": 0.7,
            "down_duration_frames": 15,
        },
        "phone": {
            "enabled": True,
            "model_path": "models/yolov8n.pt",
            "confidence_threshold": 0.4,
            "check_interval": 3,
            "face_region_y_min": 0.0,
            "face_region_y_max": 0.5,
        },
        "eyes": {
            "ear_threshold": 0.21,
            "ear_consecutive_frames": 45,
            "blink_min_frames": 2,
            "blink_max_frames": 6,
        },
        "mouth": {
            "mar_threshold": 0.50,
            "yawn_consecutive_frames": 30,
        },
        "head_pose": {
            "pitch_threshold": 15,
            "yaw_threshold": 20,
            "away_consecutive_frames": 20,
        },
    },
    "alarm": {
        "enabled": True,
        "sound_enabled": True,
        "sound_path": "assets/sounds/",
        "sound_file": "alarm.wav",
        "cooldown_seconds": 10,
        "alarm_duration_seconds": 8,
        "yawn_cooldown_seconds": 8,
        "eyes_off_threshold_seconds": 1.5,
    },
    "risk_engine": {
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
        "transition_frames": {
            "to_caution": 10,
            "to_attention": 15,
            "to_danger": 10,
            "recovery_to_caution": 40,
            "recovery_to_safe": 50,
        },
    },
    "alerts": {
        "cooldown_seconds": 5,
        "sound_enabled": True,
        "sound_path": "assets/sounds/",
        "visual_duration_seconds": 3,
    },
    "gui": {
        "theme": "dark",
        "width": 1200,
        "height": 700,
        "show_landmarks": True,
        "show_head_arrow": True,
    },
    "storage": {
        "enabled": True,
        "db_path": "~/.smart_driver_monitor/sessions.db",
    },
    "logging": {
        "level": "INFO",
        "file": "driver_monitor.log",
    },
}


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(base_dir, "config", "config.yaml")
        self._path = config_path
        self._data = copy.deepcopy(_DEFAULTS)
        self.load()

    def load(self):
        if os.path.exists(self._path):
            with open(self._path, "r") as f:
                user_config = yaml.safe_load(f) or {}
            self._data = _deep_merge(_DEFAULTS, user_config)
        else:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w") as f:
                yaml.dump(_DEFAULTS, f, default_flow_style=False)

    def get(self, *keys, default=None):
        val = self._data
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

    def __getitem__(self, keys):
        if isinstance(keys, tuple):
            return self.get(*keys)
        return self.get(keys)

    @property
    def data(self):
        return self._data
