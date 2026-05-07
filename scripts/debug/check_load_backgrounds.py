# scripts/debug/check_load_backgrounds.py
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

for lib in ['torch', 'mediapipe', 'onnxruntime']:
    try: __import__(lib)
    except ImportError: pass

from algorithms.factory import BackgroundChangerFactory

factory = BackgroundChangerFactory(algorithm_id=1)

# 查看load_backgrounds的文档和参数
import inspect
print(inspect.getsource(factory.load_backgrounds))
