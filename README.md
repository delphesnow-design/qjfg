# 实时背景切换工具

基于 RVM（Robust Video Matting）的实时视频背景替换桌面上位机，使用 Python + tkinter 构建。

## 功能特点

- 默认使用直播压力测试综合最优算法：RVM
- 支持界面切换 8 种算法：MODNet / MediaPipe / RVM / MOG2 / KNN / GrabCut / LOBSTER / SuBSENSE
- 实时背景替换：无需绿幕，摄像头画面实时替换背景
- 背景资源管理：内置背景、上传背景、动态切换
- 截图与录屏：保存到 `assets/screenshots/` 和 `assets/recordings/`
- 虚拟摄像头输出：将处理后画面输出到系统虚拟摄像头，供 Zoom、Teams、腾讯会议等软件选择
- 实时视频优先：RVM 在真实人像直播模拟中综合分最高，速度接近传统算法

## 快速开始

新机器第一次拉取项目后，先运行独立安装脚本：

```bat
install_runtime.bat
```

安装脚本会自动创建 `.venv`，并安装 `requirements.txt` 中的依赖。

默认 RVM 模型需要提前放到：

```text
models/rvm_mobilenetv3_fp32.onnx
```

安装完成后启动上位机：

```bat
start_upper_computer.bat
```

首次安装需要联网下载依赖；后续启动不需要重复安装。

## 手动安装

```bash
pip install -r requirements.txt
```

当前上位机运行链路默认使用 RVM，需要 `onnxruntime` 和模型文件：

```text
models/rvm_mobilenetv3_fp32.onnx
```

MODNet、MediaPipe 和传统 CV 算法仍保留在 `algorithms/` 中，可在上位机右侧「分割算法」下拉框中切换，便于现场对比、复查旧实验报告和重新评估。

虚拟摄像头输出依赖 `pyvirtualcam`，并要求系统里已有虚拟摄像头后端。Windows 上可安装 OBS Studio 的 OBS Virtual Camera，或安装 Unity Capture 等兼容后端。若未安装后端，程序会提示未检测到系统虚拟摄像头设备。

## 运行

```bash
python main.py
```

Windows 下也可以直接双击一键启动脚本：

```bat
start_upper_computer.bat
```

默认算法在 `config/constants.py` 中配置，启动后仍可在界面切换：

```python
OPTIMAL_ALGORITHM_ID = 2
OPTIMAL_ALGORITHM_NAME = "RVM"
```

启动上位机后，点击右侧「开启虚拟摄像头」，再到第三方会议软件的摄像头列表中选择对应虚拟摄像头设备。

可以先单独检查虚拟摄像头后端是否可用：

```bash
python scripts/debug/check_virtual_camera_backend.py
```

## 项目结构

```text
qjfg/
├── main.py                     # 应用入口
├── requirements.txt            # 上位机运行依赖
├── install_runtime.bat         # Windows 首次安装脚本，创建 .venv 并安装依赖
├── start_upper_computer.bat    # Windows 一键启动脚本
├── README.md                   # 项目说明
├── algorithms/                 # RVM 默认运行链路与历史算法实现
│   ├── factory.py
│   ├── video_thread.py
│   ├── cv_classic/
│   ├── mediapipe/
│   ├── modnet/
│   └── rvm/
├── config/                     # 项目路径、默认算法、界面参数
│   └── constants.py
├── gui/                        # tkinter 桌面界面
├── utils/                      # 文件与图像工具
├── assets/                     # 应用资源
│   ├── backgrounds/            # 背景图片
│   ├── screenshots/            # 运行时截图，已忽略
│   └── recordings/             # 运行时录屏，已忽略
├── models/                     # 本地模型文件，权重文件已忽略
├── scripts/                    # 辅助脚本
│   ├── debug/                  # 调试、排查、数据检查
│   └── evaluation/             # 实验评估与结果汇总
├── tests/                      # 手动验证脚本与测试夹具
│   ├── fixtures/
│   └── manual/
├── data/                       # 本地数据集目录，数据文件已忽略
├── experiments/                # 实验输出目录，results 已忽略
└── docs/                       # 开发记录、错误总结、结构说明
```

更详细的分类说明见 `docs/项目结构.md`。

## 常用脚本

```bash
install_runtime.bat
start_upper_computer.bat
python scripts/evaluation/webcam_experiment.py
python scripts/evaluation/live_stream_simulation_report.py
python scripts/evaluation/ppm100_eval.py
python scripts/evaluation/merge_ppm100_summary.py
python scripts/debug/check_virtual_camera_backend.py
python tests/manual/smoke_factory.py
```

PPM-100 默认放在 `data/PPM-100/`：

```text
data/PPM-100/
├── image/
└── matte/
```

也可以通过环境变量覆盖：

```powershell
$env:PPM100_DIR="D:\datasets\PPM-100"
```

## 故障排除

- 背景为空：确认 `assets/backgrounds/` 中存在支持格式图片
- RVM 初始化失败：确认已安装 `onnxruntime`，且 `models/rvm_mobilenetv3_fp32.onnx` 存在
- 虚拟摄像头启动失败：确认已安装 `pyvirtualcam`，并且系统已安装 OBS Virtual Camera 或其他兼容虚拟摄像头设备；安装后重启上位机，再运行 `python scripts/debug/check_virtual_camera_backend.py` 检查
- PPM-100 评估找不到数据：确认数据集目录或 `PPM100_DIR` 环境变量
- 摄像头无法打开：确认摄像头未被其他程序占用，并检查系统摄像头权限
