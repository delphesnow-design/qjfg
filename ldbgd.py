# check_load_bg.py
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for lib in ['torch', 'mediapipe', 'onnxruntime']:
    try: __import__(lib)
    except ImportError: pass

from algorithms.factory import BackgroundChangerFactory

factory = BackgroundChangerFactory(algorithm_id=1)

# 查看load_backgrounds的文档和参数
import inspect
print(inspect.getsource(factory.load_backgrounds))