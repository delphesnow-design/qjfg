import cv2
import numpy as np
import threading
import queue
import time
import os

from config.constants import CAMERA_FLIP_HORIZONTAL


class VideoThread(threading.Thread):
    """视频处理线程（tkinter 版，使用 queue 传帧）"""

    def __init__(self, background_changer):
        super().__init__(daemon=True)   # daemon=True：主线程退出时自动结束
        self.background_changer = background_changer
        self._run_flag = True
        self.camera_paused = False
        self._last_frame = None

        # 帧队列：视频线程生产，主线程消费
        # maxsize=2：最多缓存2帧，防止内存堆积，保持实时性
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.original_queue: queue.Queue = queue.Queue(maxsize=2)

        # 录屏
        self.recording = False
        self.video_writer = None
        self.recording_fps = 30
        self.frame_size = (640, 480)

        self.show_original = False

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("=" * 60)
            print("错误: 无法打开摄像头!")
            print("可能的原因:")
            print("1. 摄像头被其他程序占用")
            print("2. 摄像头驱动未安装或损坏")
            print("3. OpenCV 无法访问默认摄像头")
            print("=" * 60)
            return

        while self._run_flag:
            # 摄像头关闭时保持最后一帧
            if self.camera_paused:
                if self._last_frame is not None:
                    self._put_frame(self.frame_queue, self._last_frame)
                time.sleep(0.033)
                continue

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            if CAMERA_FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            self.frame_size = (frame.shape[1], frame.shape[0])

            # 发送原始帧（供截图用）
            self._put_frame(self.original_queue, frame.copy())

            if self.show_original:
                processed_frame = frame.copy()
            else:
                processed_frame = self.background_changer.process_frame(frame)

            self._last_frame = processed_frame

            # 发送处理后的帧
            self._put_frame(self.frame_queue, processed_frame)

            # 录屏
            if self.recording and self.video_writer is not None:
                out = processed_frame
                if out.shape[1] != self.frame_size[0] or out.shape[0] != self.frame_size[1]:
                    out = cv2.resize(out, self.frame_size)
                self.video_writer.write(out)

        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()

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

    def open_camera(self):
        self.camera_paused = False

    def close_camera(self):
        self.camera_paused = True

    def toggle_original(self):
        self.show_original = not self.show_original

    def stop(self):
        self._run_flag = False
