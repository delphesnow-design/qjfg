# 模型目录

将本地模型权重放在这里。常用文件名：

```text
models/selfie_multiclass_256x256.tflite
models/modnet_photographic_portrait_matting.ckpt
models/rvm_mobilenetv3_fp32.onnx
```

当前上位机默认使用 RVM，因此 `models/rvm_mobilenetv3_fp32.onnx` 是启动必需文件。MODNet 和 MediaPipe 权重用于历史对比、PPM-100 评估和直播压力测试。

权重文件通常较大，已通过 `.gitignore` 排除。
