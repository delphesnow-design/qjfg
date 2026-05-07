import os


class BackgroundChangerFactory:
    """背景替换算法工厂 - 统一接口管理不同算法"""

    ALGORITHMS = {
        0: "modnet", 1: "mediapipe", 2: "rvm",
        3: "mog2", 4: "knn", 5: "grabcut", 6: "lobster", 7: "subsense",
    }

    def __init__(self, algorithm_id=1, **kwargs):
        """
        初始化背景替换器

        Args:
            algorithm_id (int): 算法ID (0=MODNet, 1=MediaPipe, 2=RVM)
            **kwargs: 算法特定的参数
        """
        self.algorithm_id = algorithm_id
        self.algorithm_name = self.ALGORITHMS.get(algorithm_id, "mediapipe")
        self.changer = None
        self._initialize_changer(**kwargs)

    def _initialize_changer(self, **kwargs):
        """根据算法ID初始化对应的背景替换器（延迟导入避免DLL冲突）"""
        if self.algorithm_id == 0:
            # MODNet - 延迟导入，避免与MediaPipe的DLL冲突
            from algorithms.modnet.segmenter import MODNetBackgroundChanger

            model_path = kwargs.get(
                "model_path", "models/modnet_photographic_portrait_matting.ckpt"
            )
            self.changer = MODNetBackgroundChanger(model_path=model_path)

            # 检查MODNet是否可用
            if self.changer.model is None:
                print("警告: MODNet初始化失败，回退到MediaPipe")
                self.algorithm_id = 1
                self.algorithm_name = "mediapipe"
                from algorithms.mediapipe.segmenter import BackgroundChanger

                model_path = kwargs.get(
                    "mediapipe_model_path", "models/selfie_multiclass_256x256.tflite"
                )
                self.changer = BackgroundChanger(model_path=model_path)

        elif self.algorithm_id == 2:
            # RVM - 延迟导入
            from algorithms.rvm.segmenter import RVMBackgroundChanger

            model_path = kwargs.get("model_path", "models/rvm_mobilenetv3_fp32.onnx")
            self.changer = RVMBackgroundChanger(model_path=model_path)

            # 检查RVM是否可用
            if self.changer.session is None:
                print("警告: RVM初始化失败，回退到MediaPipe")
                self.algorithm_id = 1
                self.algorithm_name = "mediapipe"
                from algorithms.mediapipe.segmenter import BackgroundChanger

                model_path = kwargs.get(
                    "mediapipe_model_path", "models/selfie_multiclass_256x256.tflite"
                )
                self.changer = BackgroundChanger(model_path=model_path)

        elif self.algorithm_id == 1:
            # MediaPipe - 延迟导入
            from algorithms.mediapipe.segmenter import BackgroundChanger

            model_path = kwargs.get("model_path", "models/selfie_multiclass_256x256.tflite")
            self.changer = BackgroundChanger(model_path=model_path)

        elif self.algorithm_id in (3, 4, 5, 6, 7):
            from algorithms.cv_classic.segmenter import CVClassicBackgroundChanger
            method_map = {3: "MOG2", 4: "KNN", 5: "GrabCut", 6: "LOBSTER", 7: "SuBSENSE"}
            self.changer = CVClassicBackgroundChanger(method=method_map[self.algorithm_id])

        else:
            # 默认使用MediaPipe
            from algorithms.mediapipe.segmenter import BackgroundChanger

            print(f"未知算法ID {self.algorithm_id}，使用默认MediaPipe")
            model_path = kwargs.get("model_path", "models/selfie_multiclass_256x256.tflite")
            self.changer = BackgroundChanger(model_path=model_path)
            self.algorithm_id = 1
            self.algorithm_name = "mediapipe"

    def load_backgrounds(self, folder_path=None):
        """加载背景图片"""
        from config.constants import BACKGROUND_DIR

        if folder_path is None:
            folder_path = BACKGROUND_DIR
        return self.changer.load_backgrounds(folder_path)

    def process_frame(self, frame):
        """处理单帧图像"""
        return self.changer.process_frame(frame)

    def next_background(self):
        """切换到下一个背景"""
        return self.changer.next_background()

    def get_current_background_name(self, backgrounds_folder="背景"):
        """获取当前背景名称"""
        return self.changer.get_current_background_name(backgrounds_folder)

    def toggle_performance_mode(self):
        """切换性能模式（如果支持）"""
        if hasattr(self.changer, "toggle_performance_mode"):
            return self.changer.toggle_performance_mode()
        return False

    def toggle_temporal_consistency(self):
        """切换时序一致性（如果支持）"""
        if hasattr(self.changer, "toggle_temporal_consistency"):
            return self.changer.toggle_temporal_consistency()
        return False

    def toggle_mog2_preprocessing(self):
        """切换MOG2预处理（如果支持）"""
        if hasattr(self.changer, "toggle_mog2_preprocessing"):
            return self.changer.toggle_mog2_preprocessing()
        return False

    def set_segmentation_mode(self, mode):
        """设置分割模式（如果支持）"""
        if hasattr(self.changer, "set_segmentation_mode"):
            return self.changer.set_segmentation_mode(mode)
        return False

    def set_quality_level(self, level):
        """设置质量级别（如果支持）"""
        if hasattr(self.changer, "set_quality_level"):
            return self.changer.set_quality_level(level)
        return False

    @property
    def current_background_index(self):
        """获取当前背景索引"""
        return self.changer.current_background_index

    @current_background_index.setter
    def current_background_index(self, value):
        """设置当前背景索引"""
        self.changer.current_background_index = value

    @property
    def backgrounds(self):
        """获取背景列表"""
        return self.changer.backgrounds


def create_background_changer(algorithm_id=1, **kwargs):
    """
    创建背景替换器的便捷函数

    Args:
        algorithm_id (int): 0=MODNet, 1=MediaPipe, 2=RVM
        **kwargs: 算法特定参数

    Returns:
        BackgroundChangerFactory: 背景替换器实例
    """
    return BackgroundChangerFactory(algorithm_id, **kwargs)


# 使用示例
if __name__ == "__main__":
    # 创建MODNet背景替换器
    modnet_changer = create_background_changer(algorithm_id=0)

    # 创建MediaPipe背景替换器
    mediapipe_changer = create_background_changer(algorithm_id=1)

    print(f"MODNet changer: {modnet_changer.algorithm_name}")
    print(f"MediaPipe changer: {mediapipe_changer.algorithm_name}")
