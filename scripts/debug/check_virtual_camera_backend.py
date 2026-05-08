from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    try:
        import pyvirtualcam
    except ImportError as exc:
        print("pyvirtualcam is not installed. Run install_runtime.bat first.")
        print(f"Original error: {exc}")
        return 1

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        with pyvirtualcam.Camera(
            width=640,
            height=480,
            fps=30,
            fmt=pyvirtualcam.PixelFormat.BGR,
            print_fps=False,
        ) as cam:
            cam.send(frame)
            print(f"Virtual camera backend is ready: {cam.device}")
            return 0
    except Exception as exc:
        print("No usable system virtual camera backend was found.")
        print("Windows: install OBS Studio with OBS Virtual Camera, then restart this app.")
        print("Alternative: install Unity Capture or another pyvirtualcam-compatible backend.")
        print("Original backend error:")
        print(exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
