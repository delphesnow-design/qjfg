#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时背景切换工具（tkinter 版）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 必须在导入 GUI 之前预加载深度学习库，避免 DLL 冲突
try:
    import torch
except ImportError:
    pass

try:
    import mediapipe
except ImportError:
    pass

try:
    import onnxruntime
except ImportError:
    pass

import tkinter as tk
from config.constants import ALGORITHM_ID
from gui.main_window import BackgroundChangerGUI


def main():
    root = tk.Tk()
    app = BackgroundChangerGUI(root, algorithm_id=ALGORITHM_ID)
    root.mainloop()


if __name__ == "__main__":
    main()
