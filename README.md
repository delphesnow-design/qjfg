# 实时背景切换工具

基于人像分割的实时视频背景替换桌面应用，支持多种算法切换，使用 Python + PyQt5 构建。

## 功能特点

- **三种分割算法可选**：MODNet / MediaPipe / RVM，在 `config/constants.py` 中一键切换
- **实时背景替换**：无需绿幕，摄像头画面实时替换背景
- **多背景管理**：支持多张背景图片动态切换和上传
- **截图 & 录屏**：一键保存当前画面或录制视频

## 算法对比

| 算法 | 速度 | 质量 | 特点 |
| ------ | ------ | ------ | ------ |
| **MediaPipe** (ID=1) | ⚡⚡⚡ 极快 | ★★★ | Google 出品，CPU 友好 |
| **RVM** (ID=2) | ⚡⚡ 快 (40FPS) | ★★★★★ | 视频专用，内置时序一致性 |
| **MODNet** (ID=0) | ⚡ 中 | ★★★★ | Alpha Matte 抠图 |

## 安装

```bash
pip install -r requirements.txt
```

> **注意**：MODNet 需要 PyTorch，RVM 需要 onnxruntime。如果只用 MediaPipe 可以不安装这两个。

## 项目结构

```结构
qianjingfengge/
├── algorithms/                 # 分割算法模块
│   ├── factory.py              # 算法工厂（统一接口）
│   ├── video_thread.py         # 视频处理线程
│   ├── mediapipe/              # MediaPipe 算法
│   │   └── segmenter.py
│   ├── modnet/                 # MODNet 算法
│   │   ├── model.py            # 模型定义
│   │   └── segmenter.py
│   └── rvm/                    # RVM 算法
│       └── segmenter.py
├── assets/                     # 资源文件
│   ├── backgrounds/            # 背景图片
│   ├── screenshots/            # 截图
│   └── recordings/             # 录屏
├── config/
│   └── constants.py            # 配置（算法选择在这里）
├── gui/
│   └── main_window.py          # GUI 主窗口
├── models/                     # 预训练模型
│   ├── selfie_segmenter.tflite       # MediaPipe
│   ├── modnet_photographic_portrait_matting.ckpt  # MODNet
│   └── rvm_mobilenetv3_fp32.onnx     # RVM
├── utils/                      # 工具函数
│   ├── file_utils.py
│   └── image_utils.py
├── main.py                     # 应用入口
├── requirements.txt
└── README.md
```

## 使用方法

### 1. 选择算法

编辑 `config/constants.py`：

```python
ALGORITHM_ID = 0  # MODNet
ALGORITHM_ID = 1  # MediaPipe
ALGORITHM_ID = 2  # RVM（推荐）
```

### 2. 运行程序

```bash
python main.py
```

### 3. GUI 操作

- **背景选择**：下拉菜单选择已加载的背景图片
- **上传背景**：点击"📤 上传背景"添加本地图片
- **截图**：点击"📸 截图"保存当前画面到 `assets/screenshots/`
- **录屏**：点击录屏按钮开始/停止录制，保存到 `assets/recordings/`

## 系统要求

- Python 3.8+
- 摄像头设备
- Windows / Linux / macOS

## 技术栈

- **人像分割**：MediaPipe / MODNet (PyTorch) / RVM (ONNX Runtime)
- **GUI 框架**：PyQt5
- **视频处理**：OpenCV
- **图像处理**：NumPy / Pillow

### 4. 故障排除

- **WinError 1114 DLL错误**：确保 `main.py` 中 torch 在 PyQt5 之前导入
- **MODNet 模型加载失败**：检查模型文件路径，确认 PyTorch 已安装
- **RVM 推理失败**：确认 `onnxruntime` 已安装，模型文件存在
- **截图/上传失败**：检查目标路径权限，确认磁盘空间和文件格式
- **人物分割效果差**：改善光照条件，减少背光/顶光，尝试切换算法

## 5. 未来改进方向

1. **视频背景支持**：扩展背景类型支持视频文件
2. **网络摄像头支持**：支持多个摄像头设备选择
3. **GPU加速**：支持 CUDA 提升 MODNet/RVM 推理速度
4. **更多算法**：集成 RMBG-2.0、BiRefNet 等新一代模型
5. **配置文件支持**：支持JSON/YAML配置文件
