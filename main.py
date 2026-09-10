"""Smart Driver Monitoring & Drowsiness Detection System

Real-time computer-vision driver monitoring using facial landmarks,
eye aspect ratio (EAR), mouth aspect ratio (MAR), head pose,
hand position, and optional phone detection.

Student/research prototype. Not a certified automotive safety system.
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Smart Driver Monitoring & Drowsiness Detection System"
    )
    parser.add_argument("--config", default=None, help="Path to config YAML file")
    parser.add_argument("--no-phone", action="store_true",
                        help="Disable phone (YOLO) detection for lower CPU usage")
    args = parser.parse_args()

    from src.utils.logging_setup import setup_logging
    setup_logging()

    from src.app import DriverMonitorApp
    app = DriverMonitorApp(config_path=args.config)
    if args.no_phone:
        app._config._data["detection"]["phone"]["enabled"] = False

    from src.gui.main_window import MainWindow
    window = MainWindow(app, app._config.get("gui"))
    app.set_window(window)

    try:
        window.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()