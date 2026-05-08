#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时背景切换工具（tkinter 版）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk

from config.constants import ALGORITHM_ID, OPTIMAL_ALGORITHM_NAME
from gui.main_window import BackgroundChangerGUI


def main():
    root = tk.Tk()
    BackgroundChangerGUI(root, algorithm_id=ALGORITHM_ID)
    root.title(f"背景切换上位机 - {OPTIMAL_ALGORITHM_NAME}")
    root.mainloop()


if __name__ == "__main__":
    main()
