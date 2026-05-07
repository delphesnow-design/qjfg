import cv2
import numpy as np
from PIL import Image


def load_image_safely(file_path):
    """安全加载图片，支持OpenCV和PIL双加载策略，解决中文路径问题"""
    # 尝试使用OpenCV加载图片
    bg = cv2.imread(file_path)

    # 如果OpenCV加载失败，尝试使用PIL
    if bg is None:
        try:
            pil_image = Image.open(file_path)
            # 转换为RGB模式（如果需要）
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            # 转换为numpy数组
            bg = np.array(pil_image)
            # 转换为BGR格式（OpenCV格式）
            bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"PIL加载图片失败: {e}")
            return None

    return bg


def resize_with_aspect_ratio(image, target_width, target_height):
    """保持宽高比调整图像尺寸"""
    h, w = image.shape[:2]
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def create_default_background(width=640, height=480):
    """创建默认蓝色背景"""
    default_bg = np.zeros((height, width, 3), dtype=np.uint8)
    default_bg[:] = (100, 100, 200)  # 蓝色背景
    return default_bg


def create_gradient_background(width=640, height=480):
    """创建渐变背景"""
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        color = int(255 * (i / height))
        gradient[i, :] = [color, color, 255 - color]  # 紫色渐变
    return gradient
