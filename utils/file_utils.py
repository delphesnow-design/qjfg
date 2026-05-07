import os
import shutil
from datetime import datetime
import cv2


def ensure_directory(directory_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def get_unique_filename(directory, filename):
    """生成唯一的文件名，避免覆盖现有文件"""
    name, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename

    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{name}_{counter}{ext}"
        counter += 1

    return unique_filename


def save_image_safely(image, file_path):
    """安全保存图像，支持OpenCV和PIL双保存策略"""
    try:
        # 确保目录存在
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory(directory)

        # 尝试使用OpenCV保存
        success = cv2.imwrite(file_path, image)

        if not success:
            # 如果OpenCV保存失败，尝试使用PIL
            from PIL import Image

            # 转换BGR到RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(file_path)
            return True
        return True

    except Exception as e:
        print(f"保存图像失败: {e}")
        return False


def copy_file_safely(source_path, dest_path):
    """安全复制文件"""
    try:
        # 确保目标目录存在
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            ensure_directory(dest_dir)

        shutil.copy2(source_path, dest_path)
        return True
    except Exception as e:
        print(f"复制文件失败: {e}")
        return False


def get_supported_image_formats():
    """获取支持的图像格式"""
    return [".jpg", ".jpeg", ".png", ".bmp"]


def get_recording_filename():
    """获取录屏文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"recording_{timestamp}.avi"


def get_screenshot_filename():
    """获取截图文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"screenshot_{timestamp}.jpg"
