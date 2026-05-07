import os
import cv2
import numpy as np
from PIL import Image
from config.constants import BACKGROUND_DIR


class RVMBackgroundChanger:
    """RobustVideoMatting 背景替换器 - 专为实时视频设计，内置时序一致性"""

    def __init__(self, model_path="models/rvm_mobilenetv3_fp32.onnx"):
        self.backgrounds = []
        self.current_background_index = 0
        self.model_path = model_path
        self.session = None

        # RVM 循环状态（用于时序一致性）
        self.rec = [np.zeros([1, 1, 1, 1], dtype=np.float32)] * 4
        self.downsample_ratio = 0.25  # 下采样比率（越小越快，0.25适合CPU）

        # 性能优化参数
        self.performance_mode = False
        self.frame_skip_counter = 0
        self.frame_skip_interval = 2
        self.prev_frame = None

        self._initialize_rvm()

    def _initialize_rvm(self):
        """初始化 RVM ONNX 模型"""
        try:
            import onnxruntime as ort

            if not os.path.exists(self.model_path):
                print(f"警告: RVM模型文件未找到: {self.model_path}")
                print(
                    "请下载模型: https://github.com/PeterL1n/RobustVideoMatting/releases"
                )
                return

            # 创建 ONNX 推理会话
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            print("成功加载RVM模型")

        except ImportError:
            print("onnxruntime 未安装，请执行: pip install onnxruntime")
            self.session = None
        except Exception as e:
            print(f"初始化RVM失败: {e}")
            self.session = None

    def load_backgrounds(self, folder_path=BACKGROUND_DIR):
        """加载背景图片"""
        self.backgrounds = []

        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        if not os.listdir(folder_path):
            print("背景文件夹为空，创建示例背景...")
            default_bg = self._create_default_background()
            gradient_bg = self._create_gradient_background()
            cv2.imwrite(os.path.join(folder_path, "sample_default.jpg"), default_bg)
            cv2.imwrite(os.path.join(folder_path, "sample_gradient.jpg"), gradient_bg)

        supported_formats = [".jpg", ".jpeg", ".png", ".bmp"]
        for filename in os.listdir(folder_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in supported_formats:
                file_path = os.path.join(folder_path, filename)
                bg = self._load_image_safely(file_path)
                if bg is not None and bg.size > 0:
                    self.backgrounds.append(bg)
                    print(f"已加载背景: {filename}")

        print(f"成功加载 {len(self.backgrounds)} 个背景图片")
        return len(self.backgrounds)

    def process_frame(self, frame):
        """处理单帧图像，应用RVM背景替换"""
        if self.session is None or len(self.backgrounds) == 0:
            return frame

        try:
            # 性能模式跳帧
            if self.performance_mode:
                self.frame_skip_counter += 1
                if self.frame_skip_counter % self.frame_skip_interval != 0:
                    if self.prev_frame is not None:
                        return self.prev_frame
                    return frame
                self.frame_skip_counter = 0

            original_h, original_w = frame.shape[:2]

            # 执行 RVM 推理
            alpha_mask = self._run_rvm_inference(frame)

            if alpha_mask is None:
                return frame

            # 获取当前背景并调整到原始帧尺寸
            current_bg = self.backgrounds[self.current_background_index]
            if current_bg.shape[0] != original_h or current_bg.shape[1] != original_w:
                current_bg = cv2.resize(
                    current_bg, (original_w, original_h), interpolation=cv2.INTER_AREA
                )

            # ─── 在原始分辨率上执行 Alpha 合成 ───────────────────────────────────────
            # np.stack([alpha_mask]*3, axis=-1)
            #   alpha_mask : (H, W) float32，单通道 alpha
            #   [alpha_mask]*3 : Python 列表乘法，产生含3个相同数组引用的列表
            #   axis=-1 : 在末尾插入新轴（等价于 axis=2），沿通道维度堆叠
            #   结果 mask_3ch : (H, W, 3) float32，RGB 三通道权重相同
            mask_3ch = np.stack([alpha_mask] * 3, axis=-1)

            # Porter-Duff Alpha 合成（Over 操作）：
            #   result[y,x] = frame[y,x] × alpha[y,x] + bg[y,x] × (1 - alpha[y,x])
            #
            #   RVM 的 alpha 含义（有别于普通二值 Mask）：
            #     RVM 输出的是"软 alpha（Soft Alpha Matte）"：
            #       发丝、半透明区域的 alpha 为 0.3~0.7，实现自然头发边缘融合
            #       这是 RVM 相较于普通分割算法的核心优势
            #
            #   数据类型提升链：
            #     frame(uint8) × mask_3ch(float32) → float64（NumPy 自动上转型）
            #     float64 + float64 → float64
            #     .astype(np.uint8) → 截断到 [0,255]（小于0截为0，大于255截为255）
            result = (frame * mask_3ch + current_bg * (1 - mask_3ch)).astype(np.uint8)

            self.prev_frame = result.copy()
            return result

        except Exception as e:
            print(f"RVM处理帧时出错: {e}")
            return frame

    def _run_rvm_inference(self, frame):
        """执行 RVM ONNX 推理"""
        try:
            original_h, original_w = frame.shape[:2]

            # BGR -> RGB, HWC -> CHW, 归一化到 [0, 1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            src = frame_rgb.astype(np.float32) / 255.0
            src = np.transpose(src, (2, 0, 1))  # (3, H, W)
            src = np.expand_dims(src, axis=0)  # (1, 3, H, W)

            # 下采样比率
            downsample_ratio = np.array([self.downsample_ratio], dtype=np.float32)

            # 执行推理（带循环状态，实现时序一致性）
            inputs = {
                "src": src,
                "r1i": self.rec[0],
                "r2i": self.rec[1],
                "r3i": self.rec[2],
                "r4i": self.rec[3],
                "downsample_ratio": downsample_ratio,
            }
            fgr, pha, *rec = self.session.run(None, inputs)

            # 更新循环状态，供下一帧推理使用（RVM 时序一致性的核心机制）
            # list(rec)：将 *rec 解包得到的列表（或 tuple）显式转为 list
            #   确保 self.rec 是可变列表，后续可通过索引赋值
            self.rec = list(rec)

            # ─── 提取 alpha 遮罩 ────────────────────────────────────────────────────
            # pha 的原始形状：(1, 1, H, W)
            #   维度含义：(batch=1, channel=1, height, width)
            #   这是 NCHW 格式（batch-channel-height-width），PyTorch/ONNX 标准布局
            #
            # pha[0, 0]：NumPy 高级索引（Advanced Indexing）
            #   第一个 0 → batch 维度取第0个（唯一的那个 batch）
            #   第二个 0 → channel 维度取第0个（唯一的 alpha 通道）
            #   结果：shape (H, W)，dtype float32，值域理论上 [0.0, 1.0]
            #         1.0 = 人像前景（完全不透明）
            #         0.0 = 背景（完全透明）
            alpha = pha[0, 0]

            # np.clip(a, a_min, a_max)：将数组元素截断到 [a_min, a_max] 范围
            #   模型输出理论上已在 [0,1]，但浮点运算误差可能产生如 -0.001 或 1.0002
            #   clip 保证后续混合公式不会出现负数像素值或超出 255 的溢出
            alpha = np.clip(alpha, 0, 1)

            return alpha

        except Exception as e:
            print(f"RVM推理失败: {e}")
            return None

    def next_background(self):
        """切换到下一个背景"""
        if len(self.backgrounds) > 0:
            self.current_background_index = (self.current_background_index + 1) % len(
                self.backgrounds
            )
            return True
        return False

    def toggle_performance_mode(self):
        """切换性能模式"""
        self.performance_mode = not self.performance_mode
        if self.performance_mode:
            self.downsample_ratio = 0.15
        else:
            self.downsample_ratio = 0.25
        mode_name = "性能模式" if self.performance_mode else "高质量模式"
        print(f"RVM已切换到 {mode_name}")
        return self.performance_mode

    def get_current_background_name(self, backgrounds_folder="背景"):
        """获取当前背景的文件名"""
        if len(self.backgrounds) == 0:
            return "无背景"

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
        default_bg = np.zeros((480, 640, 3), dtype=np.uint8)
        default_bg[:] = (100, 100, 200)
        return default_bg

    def _create_gradient_background(self):
        height, width = 480, 640
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            color = int(255 * (i / height))
            gradient[i, :] = [color, color, 255 - color]
        return gradient

    def _load_image_safely(self, file_path):
        bg = cv2.imread(file_path)
        if bg is None:
            try:
                pil_image = Image.open(file_path)
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                bg = np.array(pil_image)
                bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"PIL加载图片失败: {e}")
                return None
        return bg
