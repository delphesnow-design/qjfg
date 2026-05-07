# 实时背景切换工具

基于人像分割的实时视频背景替换桌面应用，支持多种算法切换，使用 Python + tkinter 构建。

## 功能特点

- 多种分割算法：MODNet / MediaPipe / RVM / 传统 CV 算法
- 实时背景替换：无需绿幕，摄像头画面实时替换背景
- 背景资源管理：内置背景、上传背景、动态切换
- 截图与录屏：保存到 `assets/screenshots/` 和 `assets/recordings/`
- 实验脚本：支持摄像头性能实验与 PPM-100 客观评估

## 安装

```bash
pip install -r requirements.txt
```

MODNet 需要 PyTorch，RVM 需要 onnxruntime。如果只使用 MediaPipe，可按需减少依赖。

## 运行

```bash
python main.py
```

算法选择在 `config/constants.py` 中配置：

```python
ALGORITHM_ID = 0  # MODNet
ALGORITHM_ID = 1  # MediaPipe
ALGORITHM_ID = 2  # RVM
```

## 项目结构

```text
qjfg/
├── main.py                     # 应用入口
├── requirements.txt            # Python 依赖
├── README.md                   # 项目说明
├── algorithms/                 # 分割算法与统一工厂
│   ├── factory.py
│   ├── video_thread.py
│   ├── cv_classic/
│   ├── mediapipe/
│   ├── modnet/
│   └── rvm/
├── config/                     # 项目路径、算法选择、界面参数
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
python scripts/evaluation/webcam_experiment.py
python scripts/evaluation/ppm100_eval.py
python scripts/evaluation/merge_ppm100_summary.py
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

- 模型加载失败：确认对应权重文件已放入 `models/`
- 背景为空：确认 `assets/backgrounds/` 中存在支持格式图片
- PPM-100 评估找不到数据：确认数据集目录或 `PPM100_DIR` 环境变量
- DLL 初始化失败：保持 `main.py` 中深度学习库在 GUI 导入前预加载
