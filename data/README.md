# 本地数据目录

PPM-100 数据集默认放在这里：

```text
data/PPM-100/
├── image/
└── matte/
```

数据文件通常较大，`data/PPM-100/` 已加入 `.gitignore`。如果数据集放在其他位置，可设置 `PPM100_DIR`、`PPM100_IMAGE_DIR` 或 `PPM100_MATTE_DIR` 环境变量。
