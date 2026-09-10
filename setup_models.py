import os
import sys
import urllib.request
import urllib.error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_URLS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
}


def download_file(url, dest):
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  [OK] Already exists ({size_mb:.1f} MB): {os.path.basename(dest)}")
        return True
    print(f"  Downloading {os.path.basename(dest)}...")
    try:
        urllib.request.urlretrieve(url, dest, _progress_hook)
        print()
        return True
    except urllib.error.URLError as e:
        print(f"\n  [ERROR] Failed to download: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def _progress_hook(count, block_size, total_size):
    if total_size > 0:
        pct = min(count * block_size * 100 // total_size, 100)
        bar_len = 30
        filled = bar_len * pct // 100
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {pct}%", end="", flush=True)
    else:
        downloaded_mb = count * block_size / (1024 * 1024)
        print(f"\r  Downloaded {downloaded_mb:.1f} MB", end="", flush=True)


def setup_yolo():
    try:
        from ultralytics import YOLO
        model_path = os.path.join(MODELS_DIR, "yolov8n.pt")
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  [OK] Already exists ({size_mb:.1f} MB): yolov8n.pt")
            return True
        print("  Downloading yolov8n.pt via ultralytics...")
        model = YOLO("yolov8n.pt")
        import shutil
        cache_path = os.path.expanduser("~/.cache/ultralytics/ultralytics/cfg/models/v8/yolov8n.pt")
        if not os.path.exists(model_path) and os.path.exists(cache_path):
            shutil.copy2(cache_path, model_path)
        elif not os.path.exists(model_path):
            print("  [INFO] YOLO model downloaded to ultralytics cache.")
        print("  [OK] yolov8n.pt ready.")
        return True
    except ImportError:
        print("  [SKIP] ultralytics not installed. Phone detection will be disabled.")
        return False
    except Exception as e:
        print(f"  [WARN] YOLO download failed: {e}")
        print("  [INFO] You can install ultralytics and download manually: pip install ultralytics")
        return False


def main():
    print("=" * 50)
    print("  Smart Driver Monitor - Model Setup")
    print("=" * 50)
    print()

    os.makedirs(MODELS_DIR, exist_ok=True)

    all_ok = True
    for filename, url in MODEL_URLS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if not download_file(url, dest):
            all_ok = False

    print()
    setup_yolo()

    print()
    print("=" * 50)
    if all_ok:
        print("  All models ready!")
    else:
        print("  Some models failed. Check errors above.")
    print("=" * 50)


if __name__ == "__main__":
    main()
