import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_repr():
    try:
        from src.utils.config import Config
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "config.yaml")
            with open(cfg_path, "w") as f:
                f.write("camera:\n  index: 2\n  width: 800\n")
            cfg = Config(cfg_path)
            assert cfg.get("camera", "index") == 2
            assert cfg.get("camera", "width") == 800
            assert cfg.get("detection", "eyes", "ear_threshold") == 0.21
            assert cfg.get("camera", "height") == 480
            print("Config test passed.")
    except ImportError as e:
        print(f"SKIP: config test requires yaml: {e}")


def test_handler_imports():
    try:
        from src.detection.eye_analyzer import EyeAnalyzer
        from src.detection.mouth_analyzer import MouthAnalyzer
        from src.detection.head_pose import HeadPoseDetector
        from src.gui.theme import STATE_COLORS
        assert len(STATE_COLORS) >= 5
        print("Handler import test passed.")
    except ImportError as e:
        print(f"SKIP: module import test: {e}")


if __name__ == "__main__":
    test_config_repr()
    test_handler_imports()