from config.constants import (
    BACKGROUND_DIR,
    MEDIAPIPE_MODEL_PATH,
    MODNET_MODEL_PATH,
    OPTIMAL_ALGORITHM_ID,
    OPTIMAL_ALGORITHM_NAME,
    RVM_MODEL_PATH,
)


class BackgroundChangerFactory:
    """背景替换算法工厂。

    当前上位机运行链路默认使用直播压力测试综合最优的 RVM 算法。显式传入
    旧算法 ID 时仍可复现实验脚本，避免历史对比报告失真。
    """

    ALGORITHMS = {
        0: "MODNet", 1: "MediaPipe", 2: "RVM",
        3: "MOG2", 4: "KNN", 5: "GrabCut", 6: "LOBSTER", 7: "SuBSENSE",
    }

    def __init__(self, algorithm_id=OPTIMAL_ALGORITHM_ID, **kwargs):
        """
        初始化背景替换器

        Args:
            algorithm_id (int): 默认 RVM；旧实验脚本可显式传入其它算法 ID。
            **kwargs: 算法特定参数
        """
        self.algorithm_id = algorithm_id
        self.algorithm_name = self.ALGORITHMS.get(algorithm_id, OPTIMAL_ALGORITHM_NAME)
        self.changer = None
        self._initialize_changer(**kwargs)

    def _initialize_changer(self, **kwargs):
        """根据算法ID初始化对应的背景替换器（延迟导入避免DLL冲突）"""
        if self.algorithm_id == 0:
            from algorithms.modnet.segmenter import MODNetBackgroundChanger

            model_path = kwargs.get("model_path", MODNET_MODEL_PATH)
            self.changer = MODNetBackgroundChanger(model_path=model_path)
            if self.changer.model is None:
                print(f"警告: MODNet初始化失败，回退到{OPTIMAL_ALGORITHM_NAME}")
                self._init_optimal_changer()

        elif self.algorithm_id == 2:
            from algorithms.rvm.segmenter import RVMBackgroundChanger

            model_path = kwargs.get("model_path", RVM_MODEL_PATH)
            self.changer = RVMBackgroundChanger(model_path=model_path)
            if self.changer.session is None:
                print("警告: RVM初始化失败，回退到MOG2")
                self._init_mog2_fallback()

        elif self.algorithm_id == 1:
            from algorithms.mediapipe.segmenter import BackgroundChanger

            model_path = kwargs.get("model_path", MEDIAPIPE_MODEL_PATH)
            self.changer = BackgroundChanger(model_path=model_path)

        elif self.algorithm_id in (3, 4, 5, 6, 7):
            from algorithms.cv_classic.segmenter import CVClassicBackgroundChanger

            method_map = {
                3: "MOG2", 4: "KNN", 5: "GrabCut", 6: "LOBSTER", 7: "SuBSENSE"
            }
            self.changer = CVClassicBackgroundChanger(method=method_map[self.algorithm_id])

        else:
            print(f"未知算法ID {self.algorithm_id}，回退到{OPTIMAL_ALGORITHM_NAME}")
            self._init_optimal_changer()

    def _init_optimal_changer(self):
        self.algorithm_id = OPTIMAL_ALGORITHM_ID
        self.algorithm_name = OPTIMAL_ALGORITHM_NAME
        self._initialize_changer()

    def _init_mog2_fallback(self):
        from algorithms.cv_classic.segmenter import CVClassicBackgroundChanger

        self.algorithm_id = 3
        self.algorithm_name = "MOG2"
        self.changer = CVClassicBackgroundChanger(method="MOG2")

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

    def get_current_background_name(self, backgrounds_folder=BACKGROUND_DIR):
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

    @backgrounds.setter
    def backgrounds(self, value):
        """设置背景列表"""
        self.changer.backgrounds = value


def create_background_changer(algorithm_id=OPTIMAL_ALGORITHM_ID, **kwargs):
    """
    创建背景替换器的便捷函数

    Args:
        algorithm_id (int): 默认 RVM；旧实验脚本可显式传入其它算法 ID。
        **kwargs: 算法特定参数

    Returns:
        BackgroundChangerFactory: 背景替换器实例
    """
    return BackgroundChangerFactory(algorithm_id, **kwargs)


if __name__ == "__main__":
    changer = create_background_changer()
    print(f"Active changer: {changer.algorithm_name}")
