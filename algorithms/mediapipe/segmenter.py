import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
from config.constants import BACKGROUND_DIR


class BackgroundChanger:
    """背景切换器 - 处理人像分割和背景替换的核心逻辑"""

    def __init__(self, model_path="models/selfie_multiclass_256x256.tflite"):
        self.backgrounds = []
        self.current_background_index = 0
        self.segmenter = None
        self.model_path = model_path
        self._initialize_segmenter()

    def _initialize_segmenter(self):
        """初始化MediaPipe分割器"""
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            # ─── 【修复点1】改用 output_confidence_masks=True ───────────────────────
            # output_category_mask=True  ← 原来的写法（有 Bug）
            #   原理：Tasks API 对分割结果取 argmax 得到类别索引整数掩码
            #   Bug：selfie_segmenter.tflite 不同版本将"人像"编码为 1 或 255，
            #        下游用 mask==1 判断时若实际值为 255 则全部判断为 False，
            #        foreground_mask 全零 → 混合公式退化为"只显示背景"
            #
            # output_confidence_masks=True  ← 修复后的写法
            #   原理：返回每个类别的置信度浮点图，值域 [0.0, 1.0]
            #   对 selfie_segmenter.tflite：
            #     confidence_masks[0] → 背景置信度（0=确定是人，1=确定是背景）
            #     confidence_masks[1] → 人像置信度（0=确定是背景，1=确定是人）
            #   直接用 confidence_masks[1] 作为前景 alpha，语义清晰、版本鲁棒
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                output_category_mask=False,       # 关闭类别整数掩码
                output_confidence_masks=True,     # 开启置信度浮点掩码（修复关键）
            )
            # vision.ImageSegmenter.create_from_options(options)
            #   工厂方法（Factory Method）：根据 options 对象构建分割器实例
            #   返回 ImageSegmenter 对象，封装了 TFLite 模型加载与推理逻辑
            self.segmenter = vision.ImageSegmenter.create_from_options(options)
            print("成功初始化MediaPipe Image Segmenter (Tasks API)")
        except Exception as e:
            # except Exception as e：捕获所有异常类及其子类
            # as e：将异常对象绑定到变量 e，可用 str(e) / e.args 获取错误信息
            print(f"初始化MediaPipe失败: {e}")
            self.segmenter = None

    def load_backgrounds(self, folder_path=BACKGROUND_DIR):
        """加载背景图片"""
        self.backgrounds = []

        # 创建背景文件夹（如果不存在）
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        # 如果文件夹为空，创建示例背景
        if not os.listdir(folder_path):
            print("背景文件夹为空，创建示例背景...")
            default_bg = self._create_default_background()
            gradient_bg = self._create_gradient_background()
            cv2.imwrite(os.path.join(folder_path, "sample_default.jpg"), default_bg)
            cv2.imwrite(os.path.join(folder_path, "sample_gradient.jpg"), gradient_bg)

        # 加载所有支持格式的图片
        supported_formats = [".jpg", ".jpeg", ".png", ".bmp"]
        for filename in os.listdir(folder_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_formats:
                file_path = os.path.join(folder_path, filename)
                bg = self._load_image_safely(file_path)
                if bg is not None and bg.size > 0:
                    self.backgrounds.append(bg)
                    print(f"已加载背景: {filename}")
                else:
                    print(f"无法加载背景: {filename}")

        print(f"成功加载 {len(self.backgrounds)} 个背景图片")
        return len(self.backgrounds)

    def process_frame(self, frame):
        """处理单帧图像，应用背景替换"""
        if self.segmenter is None or len(self.backgrounds) == 0:
            return frame

        try:
            # 转换为RGB格式（MediaPipe需要）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 创建MediaPipe图像
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 执行分割
            segmentation_result = self.segmenter.segment(mp_image)

            # ─── 【修复点2】从置信度掩码中直接提取人像前景 alpha ──────────────────
            # segmentation_result.confidence_masks
            #   属性访问：返回 List[mp.Image]，每个元素对应一个类别的置信度图
            #   对 selfie_segmenter.tflite：
            #     索引 0 → 背景置信度图
            #     索引 1 → 人像前景置信度图  ← 我们需要这个
            confidence_masks = segmentation_result.confidence_masks
            # confidence_masks[1]：下标运算符 []，访问列表第2个元素（索引从0开始）
            #   返回 mp.Image 对象，image_format 为 GRAY_FLOAT32
            # .numpy_view()：零拷贝视图方法，返回底层内存的 numpy ndarray
            #   dtype=float32，shape=(H, W)，值域 [0.0, 1.0]
            #   1.0 表示"确定是人像"，0.0 表示"确定是背景"
            # .astype(np.float32)：显式类型转换，确保数据类型为 float32
            #   即使 numpy_view() 已经是 float32，显式转换也能保证后续运算正确
            foreground_mask = confidence_masks[1].numpy_view().astype(np.float32)

            # 【对比原来的做法及其 Bug】
            # ✗ 原代码（有 Bug）：
            #   mask = category_mask.numpy_view()           # 获取整数类别掩码
            #   foreground_mask = np.zeros_like(mask, ...)
            #   for i in range(1, 6):                       # 期望人像=类别1
            #       foreground_mask = np.maximum(           # 但实际可能是255
            #           foreground_mask, (mask == i).astype(np.uint8)
            #       )                                       # 若人像=255，mask==1全False
            #   → foreground_mask 全零 → 画面只剩背景
            #
            # ✓ 修复后：直接用置信度图作为 alpha，规避类别整数编码的版本差异

            # 应用高斯模糊使人像边缘过渡更自然（柔化硬边）
            # cv2.GaussianBlur(src, ksize, sigmaX)
            #   src    : 输入图像，float32 单通道，值域 [0,1]
            #   ksize  : 卷积核大小 (width, height)，必须为正奇数，(15,15) 覆盖约 ±7 像素边缘
            #   sigmaX : X 方向标准差；传 0 时 OpenCV 自动由 ksize 推导
            #            公式：sigma = 0.3 * ((ksize-1)*0.5 - 1) + 0.8
            #   返回：与 src 同 shape/dtype 的卷积结果，人像边缘由阶跃变成平滑渐变
            foreground_mask = cv2.GaussianBlur(foreground_mask, (15, 15), 0)

            # 获取当前背景
            current_bg = self.backgrounds[self.current_background_index]

            # 调整背景尺寸以匹配帧尺寸
            h, w = frame.shape[:2]
            if current_bg.shape[0] != h or current_bg.shape[1] != w:
                current_bg = cv2.resize(
                    current_bg, (w, h), interpolation=cv2.INTER_AREA
                )

            # ─── 将单通道 alpha 扩展为 3 通道，用于 BGR 图像的逐像素混合 ───────────
            # np.stack(arrays, axis)：沿新轴拼接数组序列
            #   [foreground_mask] * 3 → Python 列表复制：[arr, arr, arr]（同一对象的3个引用）
            #   axis=-1              → 在最后一个轴（即 channel 轴）插入新维度
            #   输入：3× (H, W) float32
            #   输出：(H, W, 3) float32，三个通道值完全相同（灰度 alpha → 彩色 alpha）
            mask_3ch = np.stack([foreground_mask] * 3, axis=-1)

            # ─── Alpha 合成公式（Porter-Duff "Over" 操作）────────────────────────────
            # result = 前景 × alpha + 背景 × (1 - alpha)
            #
            #   frame        : uint8 (H,W,3)，BGR 原始帧，人像在此
            #   mask_3ch     : float32 (H,W,3)，值 1.0=人像中心，0.0=背景，边缘柔和过渡
            #   current_bg   : uint8 (H,W,3)，替换用的背景图
            #   (1-mask_3ch) : float32，alpha 的补数（背景权重）
            #
            #   NumPy 广播规则（Broadcasting）：
            #     uint8 × float32 → float64（自动提升精度）
            #     两项相加后调用 .astype(np.uint8) 截断回 [0,255]
            #
            #   直觉理解：
            #     人像中心（alpha≈1）：result ≈ frame × 1 + bg × 0  = 原始人像  ✓
            #     背景区域（alpha≈0）：result ≈ frame × 0 + bg × 1  = 替换背景  ✓
            #     边缘（alpha≈0.5）  ：result ≈ 两者各半混合          = 自然过渡  ✓
            result = (frame * mask_3ch + current_bg * (1 - mask_3ch)).astype(np.uint8)

            return result

        except Exception as e:
            print(f"处理帧时出错: {e}")
            return frame

    def next_background(self):
        """切换到下一个背景"""
        if len(self.backgrounds) > 0:
            self.current_background_index = (self.current_background_index + 1) % len(
                self.backgrounds
            )
            return True
        return False

    def get_current_background_name(self, backgrounds_folder="背景"):
        """获取当前背景的文件名"""
        if len(self.backgrounds) == 0:
            return "无背景"

        # 获取背景文件列表
        supported_formats = [".jpg", ".jpeg", ".png", ".bmp"]
        background_files = []
        if os.path.exists(backgrounds_folder):
            for filename in os.listdir(backgrounds_folder):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in supported_formats:
                    background_files.append(filename)

        if self.current_background_index < len(background_files):
            return background_files[self.current_background_index]
        return f"背景 {self.current_background_index + 1}"

    def _create_default_background(self):
        """创建默认蓝色背景"""
        default_bg = np.zeros((480, 640, 3), dtype=np.uint8)
        default_bg[:] = (100, 100, 200)  # 蓝色背景
        return default_bg

    def _create_gradient_background(self):
        """创建渐变背景"""
        height, width = 480, 640
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            color = int(255 * (i / height))
            gradient[i, :] = [color, color, 255 - color]  # 紫色渐变
        return gradient

    def _load_image_safely(self, file_path):
        """安全加载图片，支持OpenCV和PIL双加载策略，解决中文路径问题"""
        # 尝试使用OpenCV加载图片
        bg = cv2.imread(file_path)

        # 如果OpenCV加载失败，尝试使用PIL
        if bg is None:
            try:
                from PIL import Image

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
