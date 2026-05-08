import cv2
import numpy as np
import threading
import queue
import time
import os

from config.constants import CAMERA_FLIP_HORIZONTAL, VIRTUAL_CAMERA_FPS


class VideoThread(threading.Thread):
    """视频处理线程（tkinter 版，使用 queue 传帧）"""

    def __init__(self, background_changer):
        super().__init__(daemon=True)   # daemon=True：主线程退出时自动结束
        self.background_changer = background_changer
        self._changer_lock = threading.RLock()
        self._run_flag = True
        self.camera_paused = False
        self._last_frame = None
        self._camera_error_reported = False

        # 帧队列：视频线程生产，主线程消费
        # maxsize=2：最多缓存2帧，防止内存堆积，保持实时性
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.original_queue: queue.Queue = queue.Queue(maxsize=2)

        # 录屏
        self.recording = False
        self.video_writer = None
        self.recording_fps = 30
        self.frame_size = (640, 480)

        # 虚拟摄像头输出
        self.virtual_camera_enabled = False
        self.virtual_camera = None
        self.virtual_camera_size = None
        self.virtual_camera_device = ""
        self.virtual_camera_error = ""
        self.virtual_camera_fps = VIRTUAL_CAMERA_FPS
        self._virtual_camera_module = None
        self._virtual_camera_lock = threading.Lock()

        self.show_original = False

    def run(self):
        cap = None
        try:
            while self._run_flag:
                if self.camera_paused:
                    if cap is not None:
                        cap.release()
                        cap = None
                    paused_frame = self._get_paused_frame()
                    self._last_frame = paused_frame
                    self._put_frame(self.frame_queue, paused_frame)
                    self._send_virtual_camera_frame(paused_frame)
                    time.sleep(0.1)
                    continue

                if cap is None:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        self._print_camera_open_error_once()
                        cap.release()
                        cap = None
                        time.sleep(0.5)
                        continue
                    self._camera_error_reported = False

                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    cap = None
                    time.sleep(0.05)
                    continue

                if CAMERA_FLIP_HORIZONTAL:
                    frame = cv2.flip(frame, 1)

                self.frame_size = (frame.shape[1], frame.shape[0])

                # 发送原始帧（供截图用）
                self._put_frame(self.original_queue, frame.copy())

                if self.show_original:
                    processed_frame = frame.copy()
                else:
                    processed_frame = self.get_background_changer().process_frame(frame)

                self._last_frame = processed_frame

                # 发送处理后的帧
                self._put_frame(self.frame_queue, processed_frame)

                # 录屏
                if self.recording and self.video_writer is not None:
                    out = processed_frame
                    if out.shape[1] != self.frame_size[0] or out.shape[0] != self.frame_size[1]:
                        out = cv2.resize(out, self.frame_size)
                    self.video_writer.write(out)

                self._send_virtual_camera_frame(processed_frame)
        finally:
            if cap is not None:
                cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
            self._close_virtual_camera()

    @staticmethod
    def _put_frame(q: queue.Queue, frame: np.ndarray):
        """非阻塞放帧，队列满时丢弃旧帧保持实时性"""
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        try:
            q.put_nowait(frame)
        except queue.Full:
            pass

    def start_recording(self, output_path: str) -> bool:
        if self.recording:
            return False
        try:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.recording_fps, self.frame_size
            )
            if self.video_writer.isOpened():
                self.recording = True
                return True
            self.video_writer = None
            return False
        except Exception as e:
            print(f"录屏初始化失败: {e}")
            self.video_writer = None
            return False

    def stop_recording(self) -> bool:
        if not self.recording:
            return False
        self.recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        return True

    def start_virtual_camera(self) -> bool:
        with self._virtual_camera_lock:
            if self.virtual_camera_enabled:
                return True

            try:
                import pyvirtualcam
            except ImportError as e:
                self.virtual_camera_error = (
                    "pyvirtualcam 未安装，请重新运行 install_runtime.bat"
                )
                print(f"虚拟摄像头初始化失败: {e}")
                return False

            self._virtual_camera_module = pyvirtualcam
            self.virtual_camera_enabled = True
            self.virtual_camera_error = ""
            self.virtual_camera_device = ""
            return True

    def stop_virtual_camera(self) -> bool:
        with self._virtual_camera_lock:
            was_enabled = self.virtual_camera_enabled
            self.virtual_camera_enabled = False
            self._close_virtual_camera_locked()
            return was_enabled

    def get_virtual_camera_status(self):
        with self._virtual_camera_lock:
            return {
                "enabled": self.virtual_camera_enabled,
                "active": self.virtual_camera is not None,
                "device": self.virtual_camera_device,
                "error": self.virtual_camera_error,
            }

    def get_background_changer(self):
        with self._changer_lock:
            return self.background_changer

    def set_background_changer(self, background_changer):
        with self._changer_lock:
            self.background_changer = background_changer
        if self.camera_paused:
            self.request_paused_frame_refresh()

    def open_camera(self):
        self.camera_paused = False
        self._camera_error_reported = False

    def close_camera(self):
        self.camera_paused = True
        self.request_paused_frame_refresh()

    def request_paused_frame_refresh(self):
        if not self.camera_paused:
            return
        paused_frame = self._get_paused_frame()
        self._last_frame = paused_frame
        self._put_frame(self.frame_queue, paused_frame)
        self._send_virtual_camera_frame(paused_frame)

    def toggle_original(self):
        self.show_original = not self.show_original

    def stop(self):
        self._run_flag = False

    def _send_virtual_camera_frame(self, frame: np.ndarray):
        with self._virtual_camera_lock:
            if not self.virtual_camera_enabled:
                return

            if not self._ensure_virtual_camera_locked(frame):
                return

            try:
                out = self._fit_virtual_camera_frame_locked(frame)
                self.virtual_camera.send(out)
            except Exception as e:
                self.virtual_camera_error = self._format_virtual_camera_error(e)
                print(f"虚拟摄像头发送失败: {e}")
                self.virtual_camera_enabled = False
                self._close_virtual_camera_locked()

    def _ensure_virtual_camera_locked(self, frame: np.ndarray) -> bool:
        if self.virtual_camera is not None:
            return True

        if self._virtual_camera_module is None:
            return False

        height, width = frame.shape[:2]
        try:
            self.virtual_camera = self._virtual_camera_module.Camera(
                width=width,
                height=height,
                fps=self.virtual_camera_fps,
                fmt=self._virtual_camera_module.PixelFormat.BGR,
                print_fps=False,
            )
            self.virtual_camera_size = (width, height)
            self.virtual_camera_device = self.virtual_camera.device
            self.virtual_camera_error = ""
            print(f"虚拟摄像头已启动: {self.virtual_camera_device}")
            return True
        except Exception as e:
            self.virtual_camera_error = self._format_virtual_camera_error(e)
            print(f"虚拟摄像头初始化失败: {e}")
            self.virtual_camera_enabled = False
            self._close_virtual_camera_locked()
            return False

    @staticmethod
    def _format_virtual_camera_error(error: Exception) -> str:
        raw = str(error)
        missing_backend_markers = (
            "OBS Virtual Camera device not found",
            "No camera registered",
            "backend",
        )
        if any(marker in raw for marker in missing_backend_markers):
            return (
                "未检测到系统虚拟摄像头设备。请先安装 OBS Studio/OBS Virtual Camera "
                "或 Unity Capture，重启上位机后再开启虚拟摄像头。"
            )
        return raw

    def _fit_virtual_camera_frame_locked(self, frame: np.ndarray) -> np.ndarray:
        width, height = self.virtual_camera_size
        if frame.shape[1] == width and frame.shape[0] == height:
            return np.ascontiguousarray(frame)
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(resized)

    def _close_virtual_camera(self):
        with self._virtual_camera_lock:
            self._close_virtual_camera_locked()

    def _close_virtual_camera_locked(self):
        if self.virtual_camera is not None:
            try:
                self.virtual_camera.close()
            except Exception as e:
                print(f"虚拟摄像头关闭失败: {e}")
        self.virtual_camera = None
        self.virtual_camera_size = None
        self.virtual_camera_device = ""

    def _get_paused_frame(self) -> np.ndarray:
        width, height = self.frame_size
        try:
            changer = self.get_background_changer()
            backgrounds = changer.backgrounds
            current_index = changer.current_background_index
            if backgrounds and 0 <= current_index < len(backgrounds):
                frame = backgrounds[current_index]
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                return frame.copy()
        except Exception as e:
            print(f"生成摄像头关闭预览失败: {e}")
        return np.zeros((height, width, 3), dtype=np.uint8)

    def _print_camera_open_error_once(self):
        if self._camera_error_reported:
            return
        self._camera_error_reported = True
        print("=" * 60)
        print("错误: 无法打开摄像头!")
        print("可能的原因:")
        print("1. 摄像头被其他程序占用")
        print("2. 摄像头驱动未安装或损坏")
        print("3. OpenCV 无法访问默认摄像头")
        print("=" * 60)
