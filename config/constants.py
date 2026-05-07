import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 资源目录
ASSET_DIR = os.path.join(PROJECT_ROOT, "assets")
BACKGROUND_DIR = os.path.join(ASSET_DIR, "backgrounds")
SCREENSHOT_DIR = os.path.join(ASSET_DIR, "screenshots")
RECORDING_DIR = os.path.join(ASSET_DIR, "recordings")

# 本地数据与实验输出目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "experiments")
EXPERIMENT_RESULT_DIR = os.path.join(EXPERIMENT_DIR, "results")
TEST_FIXTURE_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")
GREEN_BACKGROUND_DIR = os.path.join(TEST_FIXTURE_DIR, "backgrounds")
GREEN_BACKGROUND_PATH = os.path.join(GREEN_BACKGROUND_DIR, "green.jpg")

# PPM-100 数据集路径，可通过环境变量覆盖
PPM100_ROOT = os.environ.get("PPM100_DIR", os.path.join(DATA_DIR, "PPM-100"))
PPM100_IMAGE_DIR = os.environ.get(
    "PPM100_IMAGE_DIR", os.path.join(PPM100_ROOT, "image")
)
PPM100_MATTE_DIR = os.environ.get(
    "PPM100_MATTE_DIR", os.path.join(PPM100_ROOT, "matte")
)

# 模型路径
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MEDIAPIPE_MODEL_PATH = os.path.join(MODEL_DIR, "selfie_multiclass_256x256.tflite")
MODNET_MODEL_PATH = os.path.join(MODEL_DIR, "modnet_photographic_portrait_matting.ckpt")
RVM_MODEL_PATH = os.path.join(MODEL_DIR, "rvm_mobilenetv3_fp32.onnx")

# 向后兼容旧名称
MODEL_PATH = MEDIAPIPE_MODEL_PATH

# =============== 算法选择参数 ===============
# 修改这里的值来选择不同的算法：
# ALGORITHM_ID = 0  # MODNet
# ALGORITHM_ID = 1  # MediaPipe
# ALGORITHM_ID = 2  # RVM
# =========================================
ALGORITHM_ID = 0


# 支持的图像格式
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]

# 默认窗口尺寸
DEFAULT_WINDOW_WIDTH = 900
DEFAULT_WINDOW_HEIGHT = 600

# 录制参数
RECORDING_FPS = 30
VIDEO_CODEC = "XVID"

# 状态消息持续时间（毫秒）
STATUS_DURATION = 3000

# 颜色常量
COLOR_SUCCESS = "#27ae60"  # 绿色
COLOR_WARNING = "#f39c12"  # 橙色
COLOR_ERROR = "#e74c3c"  # 红色
COLOR_INFO = "#666666"  # 灰色

# 按钮样式
BUTTON_STYLES = {
    "screenshot": """
        QPushButton {
            padding: 6px 10px;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            color: white;
            min-height: 32px;
            min-width: 80px;
            background-color: #27ae60;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #27ae60, stop:1 #219a52);
        }
        QPushButton:pressed {
            padding: 6px 10px;
            min-height: 32px;
            min-width: 80px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #219a52, stop:1 #27ae60);
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2ecc71, stop:1 #27ae60);
        }
    """,
    "recording_start": """
        QPushButton {
            padding: 6px 10px;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            color: white;
            min-height: 32px;
            min-width: 80px;
            background-color: #e74c3c;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b);
        }
        QPushButton:pressed {
            padding: 6px 10px;
            min-height: 32px;
            min-width: 80px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #c0392b, stop:1 #e74c3c);
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #e74c3c);
        }
    """,
    "recording_stop": """
        QPushButton {
            padding: 6px 10px;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            color: white;
            min-height: 32px;
            min-width: 80px;
            background-color: #e74c3c;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #c0392b, stop:1 #e74c3c);
        }
        QPushButton:pressed {
            padding: 6px 10px;
            min-height: 32px;
            min-width: 80px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b);
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #e74c3c);
        }
    """,
    "upload": """
        QPushButton {
            padding: 6px 12px;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            color: white;
            min-height: 32px;
            background-color: #3498db;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
        }
        QPushButton:pressed {
            padding: 6px 12px;
            min-height: 32px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #3498db);
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #3498db);
        }
    """,
    "exit": """
        QPushButton {
            padding: 6px 10px;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            color: white;
            min-height: 32px;
            min-width: 80px;
            background-color: #95a5a6;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #95a5a6, stop:1 #7f8c8d);
        }
        QPushButton:pressed {
            padding: 6px 10px;
            min-height: 32px;
            min-width: 80px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7f8c8d, stop:1 #95a5a6);
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #bdc3c7, stop:1 #95a5a6);
        }
    """,
}
