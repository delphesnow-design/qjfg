import os
import cv2
import numpy as np

# ── 改成你的实际路径 ──
PPM_IMAGE_DIR = r"C:\Users\19800\Desktop\MY\cs\qianjingfengge\PPM-100\image"
PPM_MATTE_DIR = r"C:\Users\19800\Desktop\MY\cs\qianjingfengge\PPM-100\matte"

# 第一步：列出文件
image_files = sorted([
    f for f in os.listdir(PPM_IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
matte_files = sorted([
    f for f in os.listdir(PPM_MATTE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

print(f"image目录文件数: {len(image_files)}")
print(f"matte目录文件数: {len(matte_files)}")
print(f"image前3个: {image_files[:3]}")
print(f"matte前3个: {matte_files[:3]}")

# 第二步：检查文件名是否对应
fname = image_files[0]
matte_name = fname   # 同名JPG
matte_path = os.path.join(PPM_MATTE_DIR, matte_name)
print(f"\n第一张图: {fname}")
print(f"对应matte路径: {matte_path}")
print(f"matte文件存在: {os.path.exists(matte_path)}")

# 第三步：读取并检查内容
img = cv2.imread(os.path.join(PPM_IMAGE_DIR, fname))
gt  = cv2.imread(matte_path, cv2.IMREAD_GRAYSCALE)

print(f"\nimage shape: {img.shape if img is not None else 'None'}")
print(f"matte shape: {gt.shape  if gt  is not None else 'None'}")

if gt is not None:
    print(f"matte max: {gt.max()}, min: {gt.min()}")
    print(f"matte unique前10个值: {np.unique(gt)[:10]}")