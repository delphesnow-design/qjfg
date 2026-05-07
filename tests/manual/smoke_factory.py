# tests/manual/smoke_factory.py
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

# 1. 创建factory
factory = BackgroundChangerFactory(algorithm_id=1)   # MediaPipe
print("factory创建成功:", type(factory))
print("factory属性:", [x for x in dir(factory) if not x.startswith('_')])

# 2. 用纯色假背景测试process_frame
fake_bg = np.zeros((480, 640, 3), dtype=np.uint8)
fake_bg[:] = (0, 255, 0)   # 绿色

# 看factory需要如何设置背景
print("\nfactory.backgrounds类型:", type(factory.backgrounds) if hasattr(factory, 'backgrounds') else "无此属性")
print("factory.changer属性:", [x for x in dir(factory.changer) if not x.startswith('_')] if hasattr(factory, 'changer') else "无changer属性")

# 3. 测试一帧
test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
try:
    result = factory.process_frame(test_frame)
    print("\nprocess_frame成功, result shape:", result.shape)
except Exception as e:
    print("\nprocess_frame报错:", e)
