# tests/manual/smoke_single_ppm100.py
import os
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
from config.constants import PPM100_IMAGE_DIR, PPM100_MATTE_DIR

PPM_IMAGE_DIR = PPM100_IMAGE_DIR
PPM_MATTE_DIR = PPM100_MATTE_DIR

# 取第一张图
fname = sorted(os.listdir(PPM_IMAGE_DIR))[0]
image = cv2.imread(os.path.join(PPM_IMAGE_DIR, fname))
gt    = cv2.imread(os.path.join(PPM_MATTE_DIR, fname), cv2.IMREAD_GRAYSCALE)

print(f"图像: {fname}")
print(f"image shape: {image.shape}")
print(f"gt shape: {gt.shape}")

# 缩放
h, w = image.shape[:2]
if max(h, w) > 720:
    scale = 720 / max(h, w)
    image = cv2.resize(image, (int(w*scale), int(h*scale)), cv2.INTER_AREA)
    gt    = cv2.resize(gt,    (int(w*scale), int(h*scale)), cv2.INTER_AREA)
print(f"缩放后 image: {image.shape}, gt: {gt.shape}")

# 创建factory
print("\n创建factory...")
factory = BackgroundChangerFactory(algorithm_id=1)
print("factory创建成功")

# 设置绿色背景
h, w = image.shape[:2]
green = np.zeros((h, w, 3), dtype=np.uint8)
green[:] = (0, 255, 0)
factory.backgrounds              = [green]
factory.current_background_index = 0
print(f"背景设置完成, backgrounds长度: {len(factory.backgrounds)}")
print(f"factory.changer.backgrounds长度: {len(factory.changer.backgrounds)}")

# 调用process_frame
print("\n调用process_frame...")
result = factory.process_frame(image)
print(f"process_frame成功, result shape: {result.shape}")

# 显示结果
cv2.imshow("original", image)
cv2.imshow("result", result)
cv2.waitKey(3000)
cv2.destroyAllWindows()
