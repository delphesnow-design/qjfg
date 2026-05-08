import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from algorithms.factory import create_background_changer
from algorithms.video_thread import VideoThread
from utils.file_utils import (
    ensure_directory,
    get_unique_filename,
    save_image_safely,
    copy_file_safely,
    get_supported_image_formats,
    get_recording_filename,
    get_screenshot_filename,
)
from utils.image_utils import load_image_safely
from config.constants import (
    BACKGROUND_DIR,
    SCREENSHOT_DIR,
    RECORDING_DIR,
    DEFAULT_WINDOW_WIDTH,
    DEFAULT_WINDOW_HEIGHT,
    OPTIMAL_ALGORITHM_ID,
    OPTIMAL_ALGORITHM_NAME,
    COLOR_SUCCESS,
    COLOR_WARNING,
    COLOR_ERROR,
    COLOR_INFO,
)

# ── 颜色 / 字体常量 ──────────────────────────────────────
BG_MAIN    = "#f8f9fa"
BG_VIDEO   = "#2c3e50"
BG_PANEL   = "#ffffff"
FG_TITLE   = "#2c3e50"
FG_LABEL   = "#495057"
FONT_TITLE = ("Microsoft YaHei", 16, "bold")
FONT_LABEL = ("Microsoft YaHei", 10, "bold")
FONT_BTN   = ("Microsoft YaHei", 9, "bold")
FONT_COMBO = ("Microsoft YaHei", 9)
FONT_STATUS= ("Microsoft YaHei", 9)

# 按钮颜色表
BTN_COLORS = {
    "cam_open":   ("#27ae60", "#2ecc71"),   # (normal, hover)
    "cam_close":  ("#e74c3c", "#c0392b"),
    "screenshot": ("#27ae60", "#2ecc71"),
    "rec_start":  ("#e74c3c", "#c0392b"),
    "rec_stop":   ("#c0392b", "#e74c3c"),
    "virtual_start": ("#8e44ad", "#9b59b6"),
    "virtual_stop":  ("#6c3483", "#8e44ad"),
    "upload":     ("#3498db", "#2980b9"),
    "exit":       ("#95a5a6", "#7f8c8d"),
}

ALGORITHM_OPTIONS = [
    (0, "MODNet"),
    (1, "MediaPipe"),
    (2, "RVM"),
    (3, "MOG2"),
    (4, "KNN"),
    (5, "GrabCut"),
    (6, "LOBSTER"),
    (7, "SuBSENSE"),
]


def _make_btn(parent, text, command, color_key, **kw):
    """创建统一风格按钮"""
    bg, hover = BTN_COLORS[color_key]
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg="white", activebackground=hover, activeforeground="white",
        font=FONT_BTN, relief="flat", cursor="hand2",
        padx=8, pady=5, **kw
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=hover))
    btn.bind("<Leave>", lambda e: btn.config(bg=btn._normal_bg if hasattr(btn, "_normal_bg") else bg))
    btn._normal_bg = bg
    return btn


class BackgroundChangerGUI:
    def __init__(self, root: tk.Tk, algorithm_id: int = OPTIMAL_ALGORITHM_ID):
        self.root = root
        self.root.title(f"背景切换上位机 - {OPTIMAL_ALGORITHM_NAME}")
        self.root.geometry(f"{DEFAULT_WINDOW_WIDTH}x{DEFAULT_WINDOW_HEIGHT}")
        self.root.configure(bg=BG_MAIN)
        self.root.minsize(700, 480)

        self.background_changer = create_background_changer(algorithm_id=algorithm_id)
        self.current_algorithm_id = self.background_changer.algorithm_id
        self.root.title(f"背景切换上位机 - {self.background_changer.algorithm_name}")
        self.is_recording = False
        self.virtual_camera_requested = False
        self._virtual_camera_announced_device = ""
        self.current_frame = None
        self.original_frame = None
        self._photo = None          # 保持 PhotoImage 引用，防止 GC 回收
        self._poll_job = None
        self._switching_algorithm = False
        self._algorithm_switch_token = 0

        self._build_ui()
        self._start_thread()
        self.load_backgrounds()

        # 窗口关闭时安全退出
        self.root.protocol("WM_DELETE_WINDOW", self.close_application)

    # ── UI 构建 ──────────────────────────────────────────────

    def _build_ui(self):
        # ── 标题 ──
        tk.Label(
            self.root, text="🎨 背景切换工具",
            font=("Microsoft YaHei", 18, "bold"),
            bg=BG_MAIN, fg=FG_TITLE
        ).pack(pady=(10, 4))

        # ── 主内容区（视频 + 控制面板）──
        content = tk.Frame(self.root, bg=BG_MAIN)
        content.pack(fill="both", expand=True, padx=12, pady=4)

        # 视频区
        video_frame = tk.Frame(content, bg=BG_VIDEO, bd=0)
        video_frame.pack(side="left", fill="both", expand=True)

        self.video_label = tk.Label(video_frame, bg="#34495e")
        self.video_label.pack(fill="both", expand=True, padx=3, pady=3)

        # 控制面板
        panel = tk.Frame(content, bg=BG_PANEL, width=210, relief="flat", bd=1)
        panel.pack(side="right", fill="y", padx=(8, 0))
        panel.pack_propagate(False)

        self._build_panel(panel)

        # ── 状态栏 ──
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var,
            font=FONT_STATUS, bg="#ffffff", fg=COLOR_INFO,
            anchor="center", relief="flat", bd=1,
            padx=8, pady=5
        )
        self.status_label.pack(fill="x", padx=12, pady=(4, 8))

    def _build_panel(self, panel: tk.Frame):
        pad = dict(padx=10, pady=3, fill="x")

        # ── 算法选择 ──
        tk.Label(panel, text="🤖 分割算法", font=FONT_LABEL,
                 bg=BG_PANEL, fg=FG_TITLE).pack(anchor="w", padx=10, pady=(12, 2))

        self.algo_var = tk.StringVar()
        self.algo_combo = ttk.Combobox(
            panel,
            textvariable=self.algo_var,
            values=[name for _, name in ALGORITHM_OPTIONS],
            state="readonly",
            font=FONT_COMBO,
            width=22,
        )
        self.algo_combo.pack(**pad)
        self.algo_combo.bind("<<ComboboxSelected>>", self._on_algo_changed)
        self._set_algorithm_combo(self.current_algorithm_id)

        tk.Label(
            panel, text=f"默认推荐：{OPTIMAL_ALGORITHM_NAME}",
            font=("Microsoft YaHei", 8), bg=BG_PANEL, fg=COLOR_INFO
        ).pack(anchor="w", padx=10, pady=(0, 3))

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=8, pady=6)

        # ── 背景管理 ──
        tk.Label(panel, text="🖼️ 背景管理", font=FONT_LABEL,
                 bg=BG_PANEL, fg=FG_TITLE).pack(anchor="w", padx=10, pady=(2, 2))

        self.bg_var = tk.StringVar()
        self.bg_combo = ttk.Combobox(
            panel, textvariable=self.bg_var,
            state="readonly", font=FONT_COMBO, width=22
        )
        self.bg_combo.pack(**pad)
        self.bg_combo.bind("<<ComboboxSelected>>", self._on_bg_changed)

        _make_btn(panel, "📤 上传背景", self.upload_background, "upload").pack(**pad)

        ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=8, pady=6)

        # ── 功能操作（摄像头按钮置顶）──
        tk.Label(panel, text="⚙️ 功能操作", font=FONT_LABEL,
                 bg=BG_PANEL, fg=FG_TITLE).pack(anchor="w", padx=10, pady=(2, 2))

        self.btn_open_cam = _make_btn(
            panel, "📷 开启摄像头", self.open_camera, "cam_open")
        self.btn_open_cam.pack(**pad)
        self.btn_open_cam.config(state="disabled")   # 初始已开启

        self.btn_close_cam = _make_btn(
            panel, "🚫 关闭摄像头", self.close_camera, "cam_close")
        self.btn_close_cam.pack(**pad)

        self.btn_screenshot = _make_btn(
            panel, "📸 截图", self.take_screenshot, "screenshot")
        self.btn_screenshot.pack(**pad)

        self.btn_record = _make_btn(
            panel, "⏺️ 开始录屏", self.toggle_recording, "rec_start")
        self.btn_record.pack(**pad)

        self.btn_virtual_camera = _make_btn(
            panel, "🎥 开启虚拟摄像头", self.toggle_virtual_camera, "virtual_start")
        self.btn_virtual_camera.pack(**pad)

        _make_btn(panel, "🚪 退出", self.close_application, "exit").pack(**pad)

        # 弹性空间
        tk.Frame(panel, bg=BG_PANEL).pack(fill="both", expand=True)

    # ── 视频线程 ──────────────────────────────────────────────

    def _start_thread(self, camera_paused=False):
        self.thread = VideoThread(self.background_changer)
        if camera_paused:
            self.thread.close_camera()
        self.thread.start()
        self._schedule_poll()

    def _schedule_poll(self):
        if self._poll_job is None:
            # 每 33ms 轮询一次帧队列（~30fps）
            self._poll_job = self.root.after(33, self._poll_frame)

    def _cancel_poll(self):
        if self._poll_job is not None:
            try:
                self.root.after_cancel(self._poll_job)
            except Exception:
                pass
            self._poll_job = None

    def _poll_frame(self):
        """主线程定时拉取视频帧并更新显示"""
        self._poll_job = None

        try:
            frame = self.thread.frame_queue.get_nowait()
            self.current_frame = frame.copy()
            self._display_frame(frame)
        except Exception:
            pass

        try:
            orig = self.thread.original_queue.get_nowait()
            self.original_frame = orig.copy()
        except Exception:
            pass

        self._sync_virtual_camera_status()

        if self._running:
            self._schedule_poll()

    @property
    def _running(self):
        return self.thread._run_flag

    def _display_frame(self, frame: np.ndarray):
        """将 BGR numpy 帧转换并显示到 Label"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            # 保持宽高比缩放到 Label 实际大小
            lw = self.video_label.winfo_width()
            lh = self.video_label.winfo_height()
            if lw > 1 and lh > 1:
                img.thumbnail((lw, lh), Image.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=photo)
            self._photo = photo     # 防止 GC
        except Exception:
            pass

    # ── 算法切换 ──────────────────────────────────────────────

    def _on_algo_changed(self, event=None):
        if self._switching_algorithm:
            self._set_algorithm_combo(self.current_algorithm_id)
            return

        idx = self.algo_combo.current()
        if idx < 0 or idx >= len(ALGORITHM_OPTIONS):
            return

        algorithm_id, algorithm_name = ALGORITHM_OPTIONS[idx]
        if algorithm_id == self.current_algorithm_id:
            return

        self.show_status(f"正在切换到 {algorithm_name}...", "info")
        self._switching_algorithm = True
        self._algorithm_switch_token += 1
        switch_token = self._algorithm_switch_token
        self.algo_combo.config(state="disabled")

        camera_was_paused = self.thread.camera_paused
        virtual_was_requested = self.virtual_camera_requested
        current_backgrounds = list(self.background_changer.backgrounds)
        current_bg_index = self.background_changer.current_background_index

        threading.Thread(
            target=self._create_algorithm_in_background,
            args=(
                switch_token,
                algorithm_id,
                algorithm_name,
                current_backgrounds,
                current_bg_index,
                camera_was_paused,
                virtual_was_requested,
            ),
            daemon=True,
        ).start()

    def _create_algorithm_in_background(
        self,
        switch_token,
        algorithm_id,
        algorithm_name,
        current_backgrounds,
        current_bg_index,
        camera_was_paused,
        virtual_was_requested,
    ):
        new_changer = None
        error = None
        try:
            new_changer = create_background_changer(algorithm_id=algorithm_id)
            new_changer.backgrounds = current_backgrounds
            new_changer.current_background_index = (
                min(current_bg_index, len(current_backgrounds) - 1)
                if current_backgrounds else 0
            )
        except Exception as e:
            error = e

        try:
            self.root.after(
                0,
                lambda: self._finish_algorithm_switch(
                    switch_token,
                    algorithm_name,
                    new_changer,
                    error,
                    camera_was_paused,
                    virtual_was_requested,
                ),
            )
        except RuntimeError:
            pass

    def _finish_algorithm_switch(
        self,
        switch_token,
        algorithm_name,
        new_changer,
        error,
        camera_was_paused,
        virtual_was_requested,
    ):
        if switch_token != self._algorithm_switch_token:
            return

        self._switching_algorithm = False
        self.algo_combo.config(state="readonly")

        if error is not None or new_changer is None:
            self._set_algorithm_combo(self.current_algorithm_id)
            self.show_status(f"切换到 {algorithm_name} 失败: {error}", "error")
            return

        self.background_changer = new_changer
        self.current_algorithm_id = self.background_changer.algorithm_id
        self.thread.set_background_changer(self.background_changer)
        self.root.title(f"背景切换上位机 - {self.background_changer.algorithm_name}")

        self._set_algorithm_combo(self.current_algorithm_id)
        self.update_background_list()

        if camera_was_paused:
            self.thread.request_paused_frame_refresh()

        if self.is_recording:
            self.show_status(
                f"已切换到 {self.background_changer.algorithm_name}，录屏继续进行",
                "success",
            )
        elif virtual_was_requested:
            self.show_status(
                f"已切换到 {self.background_changer.algorithm_name}，虚拟摄像头继续输出",
                "success",
            )
        else:
            self.show_status(f"已切换到 {self.background_changer.algorithm_name}", "success")

    def _set_algorithm_combo(self, algorithm_id):
        if not hasattr(self, "algo_combo"):
            return
        idx = next(
            (i for i, (option_id, _) in enumerate(ALGORITHM_OPTIONS)
             if option_id == algorithm_id),
            0,
        )
        self.algo_combo.current(idx)

    def _reset_record_button(self):
        self.btn_record.config(
            text="⏺️ 开始录屏",
            bg=BTN_COLORS["rec_start"][0],
        )
        self.btn_record._normal_bg = BTN_COLORS["rec_start"][0]

    # ── 背景管理 ──────────────────────────────────────────────

    def load_backgrounds(self):
        count = self.background_changer.load_backgrounds(BACKGROUND_DIR)
        self.update_background_list()
        if count > 0:
            self.show_status("背景加载完成", "info")
        else:
            self.show_status("未找到背景图片", "warning")

    def update_background_list(self):
        supported = get_supported_image_formats()
        files = []
        if os.path.exists(BACKGROUND_DIR):
            for f in os.listdir(BACKGROUND_DIR):
                if os.path.splitext(f)[1].lower() in supported:
                    files.append(f)
        self.bg_combo["values"] = files
        if files:
            self.bg_combo.current(
                min(self.background_changer.current_background_index, len(files) - 1)
            )

    def _on_bg_changed(self, event=None):
        idx = self.bg_combo.current()
        if 0 <= idx < len(self.background_changer.backgrounds):
            self.background_changer.current_background_index = idx
            if self.thread.camera_paused:
                self.thread.request_paused_frame_refresh()
            self.show_status(f"已切换到背景: {self.bg_combo.get()}", "info")

    def upload_background(self):
        file_path = filedialog.askopenfilename(
            title="选择背景图片",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return
        try:
            if not os.path.exists(file_path):
                self.show_status("文件路径不存在", "error")
                return

            ensure_directory(BACKGROUND_DIR)
            filename = os.path.basename(file_path)
            ext = os.path.splitext(filename)[1].lower()
            supported = get_supported_image_formats()
            if ext not in supported:
                self.show_status(f"不支持的格式: {ext}", "error")
                return

            unique_name = get_unique_filename(BACKGROUND_DIR, filename)
            dest = os.path.join(BACKGROUND_DIR, unique_name)

            if copy_file_safely(file_path, dest):
                bg = load_image_safely(dest)
                if bg is not None and bg.size > 0:
                    self.background_changer.backgrounds.append(bg)
                    self.update_background_list()
                    new_index = len(self.background_changer.backgrounds) - 1
                    self.background_changer.current_background_index = new_index
                    self.bg_combo.current(new_index)
                    if self.thread.camera_paused:
                        self.thread.request_paused_frame_refresh()
                    self.show_status("背景上传成功！", "success")
                else:
                    self.show_status("无法加载图片文件", "error")
            else:
                self.show_status("文件复制失败", "error")
        except Exception as e:
            self.show_status(f"上传失败: {e}", "error")

    # ── 摄像头控制 ────────────────────────────────────────────

    def open_camera(self):
        self.thread.open_camera()
        self.btn_open_cam.config(state="disabled")
        self.btn_close_cam.config(state="normal")
        self.show_status("摄像头已开启", "success")

    def close_camera(self):
        self.thread.close_camera()
        self.btn_open_cam.config(state="normal")
        self.btn_close_cam.config(state="disabled")
        self.show_status("摄像头已关闭", "info")

    # ── 截图 ──────────────────────────────────────────────────

    def take_screenshot(self):
        try:
            ensure_directory(SCREENSHOT_DIR)
            filename = os.path.join(SCREENSHOT_DIR, get_screenshot_filename())
            if self.current_frame is not None and self.current_frame.size > 0:
                if save_image_safely(self.current_frame, filename):
                    self.show_status(f"截图已保存: {filename}", "success")
                else:
                    self.show_status("截图保存失败", "error")
            else:
                self.show_status("无法获取当前帧", "error")
        except Exception as e:
            self.show_status(f"截图失败: {e}", "error")

    # ── 录屏 ──────────────────────────────────────────────────

    def toggle_recording(self):
        if not self.is_recording:
            ensure_directory(RECORDING_DIR)
            output_path = os.path.join(RECORDING_DIR, get_recording_filename())
            if self.thread.start_recording(output_path):
                self.is_recording = True
                self.btn_record.config(text="⏹️ 停止录屏",
                                       bg=BTN_COLORS["rec_stop"][0])
                self.btn_record._normal_bg = BTN_COLORS["rec_stop"][0]
                self.show_status("开始录屏...", "info")
            else:
                self.show_status("无法开始录屏", "error")
        else:
            if self.thread.stop_recording():
                self.is_recording = False
                self._reset_record_button()
                self.show_status("录屏已停止并保存", "success")
            else:
                self.show_status("无法停止录屏", "error")

    # ── 虚拟摄像头 ────────────────────────────────────────────

    def toggle_virtual_camera(self):
        if not self.virtual_camera_requested:
            if self.thread.start_virtual_camera():
                self.virtual_camera_requested = True
                self._virtual_camera_announced_device = ""
                self.btn_virtual_camera.config(
                    text="🎥 停止虚拟摄像头",
                    bg=BTN_COLORS["virtual_stop"][0]
                )
                self.btn_virtual_camera._normal_bg = BTN_COLORS["virtual_stop"][0]
                self.show_status("正在启动虚拟摄像头输出...", "info")
            else:
                status = self.thread.get_virtual_camera_status()
                self.show_status(
                    status.get("error") or "无法启动虚拟摄像头", "error"
                )
        else:
            self.thread.stop_virtual_camera()
            self.virtual_camera_requested = False
            self._virtual_camera_announced_device = ""
            self._reset_virtual_camera_button()
            self.show_status("虚拟摄像头输出已停止", "info")

    def _sync_virtual_camera_status(self):
        if not self.virtual_camera_requested:
            return

        status = self.thread.get_virtual_camera_status()
        if not status["enabled"]:
            self.virtual_camera_requested = False
            self._virtual_camera_announced_device = ""
            self._reset_virtual_camera_button()
            self.show_status(
                status.get("error") or "虚拟摄像头输出已停止", "error"
            )
            return

        device = status.get("device") or ""
        if status["active"] and device != self._virtual_camera_announced_device:
            self._virtual_camera_announced_device = device
            if device:
                self.show_status(f"虚拟摄像头已启动: {device}", "success")
            else:
                self.show_status("虚拟摄像头已启动", "success")

    def _reset_virtual_camera_button(self):
        self.btn_virtual_camera.config(
            text="🎥 开启虚拟摄像头",
            bg=BTN_COLORS["virtual_start"][0]
        )
        self.btn_virtual_camera._normal_bg = BTN_COLORS["virtual_start"][0]

    # ── 状态栏 ──────────────────────────────────────────────

    def show_status(self, message: str, msg_type: str = "info"):
        color_map = {
            "success": COLOR_SUCCESS,
            "warning": COLOR_WARNING,
            "error":   COLOR_ERROR,
            "info":    COLOR_INFO,
        }
        self.status_label.config(fg=color_map.get(msg_type, COLOR_INFO))
        self.status_var.set(message)

    # ── 退出 ──────────────────────────────────────────────────

    def close_application(self):
        self._cancel_poll()
        self.thread.stop_virtual_camera()
        self.thread.stop()
        self.root.destroy()
