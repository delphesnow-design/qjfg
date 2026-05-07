import os
import cv2
import numpy as np
from PIL import Image
from config.constants import BACKGROUND_DIR, MODNET_MODEL_PATH


class MODNetBackgroundChanger:
    """MODNet背景切换器 - 使用MODNet进行高质量人像抠图和背景替换"""

    def __init__(self, model_path=MODNET_MODEL_PATH):
        self.backgrounds = []
        self.current_background_index = 0
        self.model = None
        self.model_path = model_path
        self.device = None

        # 性能优化参数
        self.performance_mode = False
        self.frame_skip_counter = 0
        self.frame_skip_interval = 2
        self.ref_size = 384  # 推理分辨率（256最快/384折中/512最精细）

        # 时序一致性参数（关闭可消除残影）
        self.prev_alpha = None
        self.prev_frame = None
        self.use_temporal_consistency = False

        # Matte 后处理参数
        self.matte_sharpening = True
        self.sharpening_strength = 2.0

        self._initialize_modnet()

    def _initialize_modnet(self):
        """初始化MODNet模型"""
        try:
            # 检查PyTorch是否可用
            import torch
            import torch.nn as nn

            # 检查CUDA可用性
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"使用设备: {self.device}")

            # 导入MODNet模型定义
            from .model import MODNet

            # 创建模型实例
            self.model = MODNet(backbone_pretrained=False)
            self.model = (
                nn.DataParallel(self.model) if torch.cuda.is_available() else self.model
            )
            self.model.to(self.device)

            # 加载预训练权重
            if os.path.exists(self.model_path):
                weights = torch.load(
                    self.model_path, map_location=self.device, weights_only=True
                )
                # 处理DataParallel保存的权重（key带module.前缀）
                if not isinstance(self.model, nn.DataParallel):
                    weights = {k.replace("module.", ""): v for k, v in weights.items()}
                self.model.load_state_dict(weights)
                self.model.eval()
                print("成功加载MODNet模型")
            else:
                print(f"警告: MODNet模型文件未找到: {self.model_path}")
                print("请下载预训练模型并放置在models目录下")
                self.model = None

        except ImportError as e:
            print(f"PyTorch未安装: {e}")
            print("请安装PyTorch: pip install torch torchvision")
            self.model = None
        except Exception as e:
            print(f"初始化MODNet失败: {e}")
            self.model = None

    def load_backgrounds(self, folder_path=BACKGROUND_DIR):
        """加载背景图片（与MediaPipe版本保持一致）"""
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
                else:
                    print(f"无法加载背景: {filename}")

        print(f"成功加载 {len(self.backgrounds)} 个背景图片")
        return len(self.backgrounds)

    def process_frame(self, frame):
        """处理单帧图像，应用MODNet背景替换"""
        if self.model is None or len(self.backgrounds) == 0:
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

            # 执行MODNet分割，直接在原图上推理，内部处理缩放
            alpha_mask = self._run_modnet_inference(frame)

            if alpha_mask is None:
                return frame

            # Matte 后处理：锐化 + 边缘平滑（解决灯光/高光导致的半透明问题）
            if self.matte_sharpening:
                alpha_mask = self._sharpen_matte(alpha_mask)

            # 时序一致性优化
            if self.use_temporal_consistency and self.prev_alpha is not None:
                if self.prev_alpha.shape == alpha_mask.shape:
                    alpha_mask = self._apply_temporal_smoothing(alpha_mask)

            self.prev_alpha = alpha_mask.copy()

            # 获取当前背景并调整到原始帧尺寸
            current_bg = self.backgrounds[self.current_background_index]
            if current_bg.shape[0] != original_h or current_bg.shape[1] != original_w:
                current_bg = cv2.resize(
                    current_bg, (original_w, original_h), interpolation=cv2.INTER_AREA
                )

            # 在原始分辨率上混合前景和背景
            mask_3ch = np.stack([alpha_mask] * 3, axis=-1)
            result = (frame * mask_3ch + current_bg * (1 - mask_3ch)).astype(np.uint8)

            self.prev_frame = result.copy()
            return result

        except Exception as e:
            print(f"MODNet处理帧时出错: {e}")
            return frame

    def _run_modnet_inference(self, frame):
        """执行MODNet推理（按官方预处理流程）"""
        try:
            import torch
            import torch.nn.functional as F
            import torchvision.transforms as transforms

            original_h, original_w = frame.shape[:2]

            # 转换为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(frame_rgb)

            # 标准化变换
            im_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            im = im_transform(im_pil)  # (3, H, W)

            # 保持宽高比缩放到 ref_size，并对齐到 32 的倍数
            ref_size = self.ref_size

            # 按短边缩放到 ref_size，保持宽高比
            im_h, im_w = im.shape[1], im.shape[2]
            if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
                if im_w >= im_h:
                    new_h = ref_size
                    new_w = int(im_w / im_h * ref_size)
                else:
                    new_w = ref_size
                    new_h = int(im_h / im_w * ref_size)
            else:
                new_h, new_w = im_h, im_w

            # 对齐到 32 的倍数（MODNet encoder 下采样要求）
            new_h = ((new_h - 1) // 32 + 1) * 32
            new_w = ((new_w - 1) // 32 + 1) * 32

            # resize 输入张量
            im = im.unsqueeze(0)  # (1, 3, H, W)
            im_resized = F.interpolate(im, size=(new_h, new_w), mode="area")
            im_resized = im_resized.to(self.device)

            # 推理
            with torch.no_grad():
                _, _, matte = self.model(im_resized, True)

            # 将 matte resize 回原始尺寸
            matte = F.interpolate(matte, size=(original_h, original_w), mode="area")
            matte = matte[0, 0].cpu().numpy()
            matte = np.clip(matte, 0, 1)

            return matte

        except Exception as e:
            print(f"MODNet推理失败: {e}")
            return None

    def _sharpen_matte(self, matte):
        """锐化 matte，减少半透明区域"""
        # 对比度增强 + 边缘平滑
        midpoint = 0.5
        matte = 1.0 / (1.0 + np.exp(-self.sharpening_strength * 6 * (matte - midpoint)))
        matte = cv2.GaussianBlur(matte.astype(np.float32), (5, 5), 0)
        return np.clip(matte, 0.0, 1.0)

    def _apply_temporal_smoothing(self, current_alpha):
        """时序平滑处理"""
        alpha = 0.3
        smoothed_alpha = alpha * current_alpha + (1 - alpha) * self.prev_alpha
        return np.clip(smoothed_alpha, 0.0, 1.0)

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
        mode_name = "性能模式" if self.performance_mode else "高质量模式"
        print(f"MODNet已切换到 {mode_name}")
        return self.performance_mode

    def toggle_temporal_consistency(self):
        """切换时序一致性"""
        self.use_temporal_consistency = not self.use_temporal_consistency
        status = "启用" if self.use_temporal_consistency else "禁用"
        print(f"MODNet时序一致性已{status}")
        return self.use_temporal_consistency

    def get_current_background_name(self, backgrounds_folder=BACKGROUND_DIR):
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
        """创建默认蓝色背景"""
        default_bg = np.zeros((480, 640, 3), dtype=np.uint8)
        default_bg[:] = (100, 100, 200)
        return default_bg

    def _create_gradient_background(self):
        """创建渐变背景"""
        height, width = 480, 640
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            color = int(255 * (i / height))
            gradient[i, :] = [color, color, 255 - color]
        return gradient

    def _load_image_safely(self, file_path):
        """安全加载图片"""
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
