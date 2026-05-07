"""
传统计算机视觉背景替换算法
支持: MOG2, KNN, GrabCut, LOBSTER, SuBSENSE
"""
import os
from typing import Optional

import cv2
import numpy as np

from config.constants import BACKGROUND_DIR


# ─────────────────────────────────────────────────────────────
#  通用后处理工具
# ─────────────────────────────────────────────────────────────

def _fill_holes(mask: np.ndarray, ksize: int = 19) -> np.ndarray:
    """
    形态学闭运算填补内部空洞。
    原理：先膨胀把空洞"堵住"，再腐蚀还原轮廓大小。
    ksize 越大，能填补的空洞越大。
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def _keep_largest_components(mask: np.ndarray,
                              max_keep: int = 3,
                              min_area_frac: float = 0.003) -> np.ndarray:
    """
    连通域分析：只保留面积最大的 max_keep 个区域，丢弃孤立噪点。
    原理：前景物体（人）通常是画面中最大的连通区域；
         细小噪点面积极小，按面积阈值过滤即可去除。
    """
    binary = (mask > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return mask

    min_area = max(100, int(mask.shape[0] * mask.shape[1] * min_area_frac))
    # 按面积降序排列（跳过 label 0 = 背景）
    areas = sorted(
        [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)],
        reverse=True,
    )
    result = np.zeros_like(mask)
    for area, lid in areas[:max_keep]:
        if area >= min_area:
            result[labels == lid] = mask[labels == lid]
    return result


def _morph_erode(mask: np.ndarray, iterations: int, ksize: int = 5) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(mask, k, iterations=iterations)


def _morph_dilate(mask: np.ndarray, iterations: int, ksize: int = 5) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask, k, iterations=iterations)


def _smooth_alpha(mask: np.ndarray, blur: int) -> np.ndarray:
    blur_k = max(3, blur | 1)
    return cv2.GaussianBlur(mask, (blur_k, blur_k), 0)


def _alpha_blend(frame: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    bg_r = cv2.resize(bg, (w, h))
    if alpha.ndim == 2:
        alpha = np.stack([alpha] * 3, axis=-1)
    blended = frame.astype(np.float32) * alpha + bg_r.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
#  LBSP 工具（LOBSTER / SuBSENSE 共用）
# ─────────────────────────────────────────────────────────────

_LBSP_RING: np.ndarray = np.array(
    [
        (-1, -1), (-1,  0), (-1,  1), ( 0, -1),
        ( 0,  1), ( 1, -1), ( 1,  0), ( 1,  1),
        (-2, -2), (-2,  0), (-2,  2), ( 0, -2),
        ( 0,  2), ( 2, -2), ( 2,  0), ( 2,  2),
    ],
    dtype=np.int32,
)


def _lbsp_desc(gray: np.ndarray, t: int = 20) -> np.ndarray:
    g = gray.astype(np.int16)
    desc = np.zeros(gray.shape, dtype=np.uint16)
    for i, (dy, dx) in enumerate(_LBSP_RING):
        shifted = np.roll(np.roll(gray, int(dy), axis=0), int(dx), axis=1).astype(np.int16)
        bit = (np.abs(g - shifted) < t).astype(np.uint16)
        desc |= bit << np.uint16(i)
    return desc


def _hamming16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xor = (a ^ b).astype(np.uint16)
    hi = (xor >> np.uint16(8)).astype(np.uint8)
    lo = (xor & np.uint16(0xFF)).astype(np.uint8)
    cnt_hi = np.unpackbits(hi[..., np.newaxis], axis=-1).sum(axis=-1)
    cnt_lo = np.unpackbits(lo[..., np.newaxis], axis=-1).sum(axis=-1)
    return (cnt_hi + cnt_lo).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
#  MOG2 / KNN
# ─────────────────────────────────────────────────────────────

class _SubtractorProcessor:
    """
    优化点：
    1. 闭运算填补人体内部空洞（皮肤反光/低纹理区域易被误判背景）
    2. 连通域过滤去除孤立噪点颗粒
    3. 膨胀系数加大以覆盖人像边缘缺失区域
    """
    def __init__(self, algorithm="MOG2"):
        self.algorithm = algorithm
        self.var_threshold = 16
        self.dist_threshold = 400
        self.erode_iter  = 1
        self.dilate_iter = 3   # 加大膨胀，补全人像边缘缺失
        self.blur_size   = 21
        self._build()

    def _build(self):
        if self.algorithm == "MOG2":
            self._sub = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=self.var_threshold, detectShadows=True
            )
        else:
            self._sub = cv2.createBackgroundSubtractorKNN(
                history=500, dist2Threshold=float(self.dist_threshold), detectShadows=True
            )

    def reset(self):
        self._build()

    def get_alpha(self, frame: np.ndarray) -> np.ndarray:
        raw = self._sub.apply(frame)
        _, mask = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)

        # ① 闭运算：填补人体内部空洞（解决颗粒遮挡）
        mask = _fill_holes(mask, ksize=19)

        # ② 腐蚀去噪 → 膨胀恢复（erode_iter < dilate_iter 使整体偏向扩张）
        if self.erode_iter > 0:
            mask = _morph_erode(mask, self.erode_iter)
        if self.dilate_iter > 0:
            mask = _morph_dilate(mask, self.dilate_iter)

        # ③ 连通域过滤：丢弃孤立噪点小区域（解决颗粒问题）
        mask = _keep_largest_components(mask)

        # ④ 高斯模糊柔化边缘
        mask = _smooth_alpha(mask, self.blur_size)
        return mask.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────
#  GrabCut
# ─────────────────────────────────────────────────────────────

class _GrabCutProcessor:
    """
    优化点：
    1. 迭代次数 2→5：更多迭代让图割能量收敛，减少边界误判
    2. 闭运算填补人像内部孔洞（人体内部颜色差异大时易被误割）
    3. 额外腐蚀收缩边界：GrabCut 矩形内的背景像素会被误标前景，
       腐蚀可以将轮廓向内收紧，去掉外围误判的背景条带
    4. 连通域过滤去除矩形边角区域的杂散前景块
    """
    def __init__(self):
        self.iterations      = 5    # 原 2 → 5，能量收敛更好
        self.rect_ratio      = 0.10
        self.erode_iter      = 2    # 原 1 → 2，收缩外扩的背景误判
        self.dilate_iter     = 1
        self.blur_size       = 15
        self.process_max_side = 420
        self.run_every_n_frames = 3
        self._skip = 0
        self._last_alpha: Optional[np.ndarray] = None

    def reset(self):
        self._skip = 0
        self._last_alpha = None

    def get_alpha(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._skip += 1
        if (
            self._skip % self.run_every_n_frames != 0
            and self._last_alpha is not None
            and self._last_alpha.shape[:2] == (h, w)
        ):
            return self._last_alpha

        m = max(h, w)
        if m > self.process_max_side:
            scale = self.process_max_side / m
            sw = max(1, int(round(w * scale)))
            sh = max(1, int(round(h * scale)))
            small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        shs, sws = small.shape[:2]

        rx = max(1, int(sws * self.rect_ratio))
        ry = max(1, int(shs * self.rect_ratio))
        rw, rh = sws - 2 * rx, shs - 2 * ry
        if rw < 8 or rh < 8:
            return np.zeros((h, w), np.float32)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        mask_gc   = np.zeros((shs, sws), np.uint8)
        try:
            cv2.grabCut(small, mask_gc, (rx, ry, rw, rh),
                        bgd_model, fgd_model,
                        self.iterations, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return np.zeros((h, w), np.float32)

        fg_mask = np.where(
            (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # ① 闭运算：填补人像内部因颜色差异被割掉的孔洞
        fg_mask = _fill_holes(fg_mask, ksize=15)

        # ② 腐蚀收缩：去掉轮廓外沿被误判的背景条带
        if self.erode_iter > 0:
            fg_mask = _morph_erode(fg_mask, self.erode_iter)
        if self.dilate_iter > 0:
            fg_mask = _morph_dilate(fg_mask, self.dilate_iter)

        # ③ 连通域过滤：去掉矩形角落的杂散前景块
        fg_mask = _keep_largest_components(fg_mask)

        if fg_mask.shape[1] != w or fg_mask.shape[0] != h:
            fg_mask = cv2.resize(fg_mask, (w, h), interpolation=cv2.INTER_LINEAR)

        alpha = fg_mask.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (self.blur_size | 1, self.blur_size | 1), 0)
        self._last_alpha = alpha
        return alpha


# ─────────────────────────────────────────────────────────────
#  LOBSTER
# ─────────────────────────────────────────────────────────────

class _LOBSTERProcessor:
    """
    优化点：
    1. T_COLOR 30→40：人体皮肤纹理少，略微放宽颜色匹配容忍度，
       减少皮肤区域被误判为背景导致的轮廓不完整问题
    2. 闭运算填孔 + 连通域过滤，同 MOG2
    3. 膨胀系数加大以补全人像缺失轮廓
    """
    N_SAMPLES        = 16
    K_MIN            = 2
    T_COLOR          = 40   # 原 30 → 40，降低对皮肤区域的误判
    T_LBSP           = 4
    T_LBSP_SINGLE    = 20
    LEARN_RATE       = 16
    PROCESS_MAX_SIDE = 360

    def __init__(self):
        self.erode_iter  = 1
        self.dilate_iter = 3   # 原 2 → 3
        self.blur_size   = 15
        self._bgr:  Optional[np.ndarray] = None
        self._lbsp: Optional[np.ndarray] = None
        self._ready = False
        self._rng   = np.random.default_rng(42)

    def reset(self):
        self._bgr   = None
        self._lbsp  = None
        self._ready = False

    def _init_model(self, frame: np.ndarray):
        H, W = frame.shape[:2]
        N = self.N_SAMPLES
        self._bgr  = np.empty((H, W, N, 3), dtype=np.uint8)
        self._lbsp = np.empty((H, W, N),    dtype=np.uint16)
        for k in range(N):
            noise = self._rng.integers(-5, 6, (H, W, 3), dtype=np.int16)
            s_bgr = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            self._bgr[:, :, k, :] = s_bgr
            s_gray = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2GRAY)
            self._lbsp[:, :, k] = _lbsp_desc(s_gray, self.T_LBSP_SINGLE)
        self._ready = True

    def _classify(self, frame, cur_lbsp):
        f_exp = frame[:, :, np.newaxis, :].astype(np.int16)
        color_dist = np.sum(np.abs(self._bgr.astype(np.int16) - f_exp), axis=3)
        color_ok   = color_dist < self.T_COLOR
        lbsp_exp   = cur_lbsp[:, :, np.newaxis]
        hamming    = _hamming16(self._lbsp, lbsp_exp)
        lbsp_ok    = hamming < self.T_LBSP
        match_cnt  = (color_ok & lbsp_ok).sum(axis=2)
        is_bg      = match_cnt >= self.K_MIN
        fg_mask    = (~is_bg).astype(np.uint8) * 255
        return fg_mask, is_bg

    def _update_model(self, frame, cur_lbsp, is_bg):
        H, W   = frame.shape[:2]
        N, R   = self.N_SAMPLES, self.LEARN_RATE
        draw   = self._rng.integers(0, R, (H, W))
        update = (draw == 0) & is_bg
        ys, xs = np.nonzero(update)
        if len(ys) == 0:
            return
        slots = self._rng.integers(0, N, len(ys))
        self._bgr [ys, xs, slots, :] = frame[ys, xs]
        self._lbsp[ys, xs, slots]    = cur_lbsp[ys, xs]
        dy = self._rng.integers(-1, 2, len(ys))
        dx = self._rng.integers(-1, 2, len(ys))
        ny = np.clip(ys + dy, 0, H - 1)
        nx = np.clip(xs + dx, 0, W - 1)
        s2 = self._rng.integers(0, N, len(ys))
        self._bgr [ny, nx, s2, :] = frame[ys, xs]
        self._lbsp[ny, nx, s2]    = cur_lbsp[ys, xs]

    def get_alpha(self, frame_orig: np.ndarray) -> np.ndarray:
        H0, W0 = frame_orig.shape[:2]
        m = max(H0, W0)
        if m > self.PROCESS_MAX_SIDE:
            scale = self.PROCESS_MAX_SIDE / m
            W1 = max(1, int(round(W0 * scale)))
            H1 = max(1, int(round(H0 * scale)))
            frame = cv2.resize(frame_orig, (W1, H1), interpolation=cv2.INTER_AREA)
        else:
            frame = frame_orig
        H, W = frame.shape[:2]

        if not self._ready or self._bgr.shape[:2] != (H, W):
            self._init_model(frame)

        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_lbsp = _lbsp_desc(gray, self.T_LBSP_SINGLE)
        fg_mask, is_bg = self._classify(frame, cur_lbsp)
        self._update_model(frame, cur_lbsp, is_bg)

        # ① 闭运算填孔
        fg_mask = _fill_holes(fg_mask, ksize=15)
        # ② 腐蚀去边噪 → 膨胀补全
        if self.erode_iter > 0:
            fg_mask = _morph_erode(fg_mask, self.erode_iter)
        if self.dilate_iter > 0:
            fg_mask = _morph_dilate(fg_mask, self.dilate_iter)
        # ③ 连通域过滤孤立颗粒
        fg_mask = _keep_largest_components(fg_mask)

        if fg_mask.shape != (H0, W0):
            fg_mask = cv2.resize(fg_mask, (W0, H0), interpolation=cv2.INTER_LINEAR)

        fg_mask = _smooth_alpha(fg_mask, self.blur_size)
        return fg_mask.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────
#  SuBSENSE
# ─────────────────────────────────────────────────────────────

class _SuBSENSEProcessor:
    """
    优化点：
    1. T_COLOR_MIN 20→35：初始颜色匹配阈值过低时，稍有光照变化
       就触发前景报警，背景边缘会被误判为人像。提高后，背景像素
       更容易与样本库匹配，从而减少假阳性。
    2. T_LBSP_MIN 2→3：同样的道理放宽纹理匹配。
    3. 额外腐蚀 2 次：SuBSENSE 边界天然偏外扩，腐蚀把人像轮廓
       外沿的背景误判区域收回去。
    4. 连通域过滤：去除人像周围散落的孤立前景小块。
    """
    N_SAMPLES        = 16
    K_MIN            = 2
    T_LBSP_SINGLE    = 20
    LEARN_RATE       = 16
    PROCESS_MAX_SIDE = 360
    T_COLOR_MIN  = 35.0   # 原 20 → 35，降低背景边缘的假阳性
    T_COLOR_MAX  = 80.0
    T_LBSP_MIN   = 3.0    # 原 2 → 3
    T_LBSP_MAX   = 8.0
    B_MAX        = 64
    ALPHA_C      = 0.703125   # (80-35)/64
    ALPHA_L      = 0.078125   # (8-3)/64

    def __init__(self):
        self.erode_iter  = 3   # 原 1 → 3，主动收缩外沿背景误判
        self.dilate_iter = 1
        self.blur_size   = 15
        self._bgr     = None
        self._lbsp    = None
        self._t_color = None
        self._t_lbsp  = None
        self._blink   = None
        self._prev_fg = None
        self._ready   = False
        self._rng = np.random.default_rng(0)

    def reset(self):
        self._bgr = self._lbsp = self._t_color = self._t_lbsp = None
        self._blink = self._prev_fg = None
        self._ready = False

    def _init_model(self, frame: np.ndarray):
        H, W = frame.shape[:2]
        N = self.N_SAMPLES
        self._bgr  = np.empty((H, W, N, 3), dtype=np.uint8)
        self._lbsp = np.empty((H, W, N),    dtype=np.uint16)
        for k in range(N):
            noise = self._rng.integers(-5, 6, (H, W, 3), dtype=np.int16)
            s_bgr = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            self._bgr[:, :, k, :] = s_bgr
            s_gray = cv2.cvtColor(s_bgr, cv2.COLOR_BGR2GRAY)
            self._lbsp[:, :, k] = _lbsp_desc(s_gray, self.T_LBSP_SINGLE)
        self._t_color = np.full((H, W), self.T_COLOR_MIN, dtype=np.float32)
        self._t_lbsp  = np.full((H, W), self.T_LBSP_MIN,  dtype=np.float32)
        self._blink   = np.zeros((H, W), dtype=np.int16)
        self._prev_fg = np.zeros((H, W), dtype=bool)
        self._ready   = True

    def _classify(self, frame, cur_lbsp):
        f_exp = frame[:, :, np.newaxis, :].astype(np.int16)
        color_dist = np.sum(np.abs(self._bgr.astype(np.int16) - f_exp), axis=3)
        color_ok   = color_dist < self._t_color[:, :, np.newaxis]
        lbsp_exp   = cur_lbsp[:, :, np.newaxis]
        hamming    = _hamming16(self._lbsp, lbsp_exp)
        lbsp_ok    = hamming < self._t_lbsp[:, :, np.newaxis]
        match_cnt  = (color_ok & lbsp_ok).sum(axis=2)
        is_bg      = match_cnt >= self.K_MIN
        fg_mask    = (~is_bg).astype(np.uint8) * 255
        return fg_mask, is_bg

    def _update_thresholds(self, cur_fg: np.ndarray):
        flipped = cur_fg ^ self._prev_fg
        self._blink = np.clip(
            self._blink + np.where(flipped, 1, -1).astype(np.int16),
            0, self.B_MAX,
        ).astype(np.int16)
        b = self._blink.astype(np.float32)
        self._t_color = np.clip(
            self.T_COLOR_MIN + b * self.ALPHA_C,
            self.T_COLOR_MIN, self.T_COLOR_MAX,
        ).astype(np.float32)
        self._t_lbsp = np.clip(
            self.T_LBSP_MIN + b * self.ALPHA_L,
            self.T_LBSP_MIN, self.T_LBSP_MAX,
        ).astype(np.float32)
        self._prev_fg[:] = cur_fg

    def _update_model(self, frame, cur_lbsp, is_bg):
        H, W   = frame.shape[:2]
        N, R   = self.N_SAMPLES, self.LEARN_RATE
        draw   = self._rng.integers(0, R, (H, W))
        update = (draw == 0) & is_bg
        ys, xs = np.nonzero(update)
        if len(ys) == 0:
            return
        slots = self._rng.integers(0, N, len(ys))
        self._bgr [ys, xs, slots, :] = frame[ys, xs]
        self._lbsp[ys, xs, slots]    = cur_lbsp[ys, xs]
        dy = self._rng.integers(-1, 2, len(ys))
        dx = self._rng.integers(-1, 2, len(ys))
        ny = np.clip(ys + dy, 0, H - 1)
        nx = np.clip(xs + dx, 0, W - 1)
        s2 = self._rng.integers(0, N, len(ys))
        self._bgr [ny, nx, s2, :] = frame[ys, xs]
        self._lbsp[ny, nx, s2]    = cur_lbsp[ys, xs]

    def get_alpha(self, frame_orig: np.ndarray) -> np.ndarray:
        H0, W0 = frame_orig.shape[:2]
        m = max(H0, W0)
        if m > self.PROCESS_MAX_SIDE:
            scale = self.PROCESS_MAX_SIDE / m
            W1 = max(1, int(round(W0 * scale)))
            H1 = max(1, int(round(H0 * scale)))
            frame = cv2.resize(frame_orig, (W1, H1), interpolation=cv2.INTER_AREA)
        else:
            frame = frame_orig
        H, W = frame.shape[:2]

        if not self._ready or self._bgr.shape[:2] != (H, W):
            self._init_model(frame)

        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_lbsp = _lbsp_desc(gray, self.T_LBSP_SINGLE)
        fg_mask, is_bg = self._classify(frame, cur_lbsp)
        self._update_thresholds(~is_bg)
        self._update_model(frame, cur_lbsp, is_bg)

        # ① 腐蚀收缩：去掉轮廓外沿背景误判区域（erode_iter=3）
        if self.erode_iter > 0:
            fg_mask = _morph_erode(fg_mask, self.erode_iter)
        # ② 轻度膨胀恢复人像轮廓
        if self.dilate_iter > 0:
            fg_mask = _morph_dilate(fg_mask, self.dilate_iter)
        # ③ 连通域过滤：去掉人像周围散落的孤立前景小块
        fg_mask = _keep_largest_components(fg_mask, max_keep=2)

        if fg_mask.shape != (H0, W0):
            fg_mask = cv2.resize(fg_mask, (W0, H0), interpolation=cv2.INTER_LINEAR)

        fg_mask = _smooth_alpha(fg_mask, self.blur_size)
        return fg_mask.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────
#  统一背景替换类（兼容项目接口）
# ─────────────────────────────────────────────────────────────

class CVClassicBackgroundChanger:
    """
    传统 CV 背景替换器，兼容项目的 BackgroundChanger 接口。
    method: "MOG2" | "KNN" | "GrabCut" | "LOBSTER" | "SuBSENSE"
    """

    def __init__(self, method: str = "MOG2"):
        self.backgrounds = []
        self.current_background_index = 0
        self._method = method

        if method in ("MOG2", "KNN"):
            self._processor = _SubtractorProcessor(algorithm=method)
        elif method == "GrabCut":
            self._processor = _GrabCutProcessor()
        elif method == "LOBSTER":
            self._processor = _LOBSTERProcessor()
        elif method == "SuBSENSE":
            self._processor = _SuBSENSEProcessor()
        else:
            raise ValueError(f"未知方法: {method}")

        print(f"已初始化 {method} 背景替换器")

    def load_backgrounds(self, folder_path: str) -> int:
        self.backgrounds = []
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
        supported_formats = [".jpg", ".jpeg", ".png", ".bmp"]
        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_formats:
                file_path = os.path.join(folder_path, filename)
                bg = self._load_image(file_path)
                if bg is not None:
                    self.backgrounds.append(bg)
                    print(f"已加载背景: {filename}")
        print(f"成功加载 {len(self.backgrounds)} 个背景图片")
        return len(self.backgrounds)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if len(self.backgrounds) == 0:
            return frame
        try:
            alpha = self._processor.get_alpha(frame)
            bg    = self.backgrounds[self.current_background_index]
            return _alpha_blend(frame, bg, alpha)
        except Exception as e:
            print(f"[{self._method}] 处理帧出错: {e}")
            return frame

    def next_background(self) -> bool:
        if len(self.backgrounds) > 0:
            self.current_background_index = (
                self.current_background_index + 1
            ) % len(self.backgrounds)
            return True
        return False

    def get_current_background_name(self, backgrounds_folder: str = BACKGROUND_DIR) -> str:
        if not self.backgrounds:
            return "无背景"
        supported_formats = [".jpg", ".jpeg", ".png", ".bmp"]
        files = []
        if os.path.exists(backgrounds_folder):
            for f in os.listdir(backgrounds_folder):
                if os.path.splitext(f)[1].lower() in supported_formats:
                    files.append(f)
        if self.current_background_index < len(files):
            return files[self.current_background_index]
        return f"背景 {self.current_background_index + 1}"

    def _load_image(self, file_path: str) -> Optional[np.ndarray]:
        bg = cv2.imread(file_path)
        if bg is not None:
            return bg
        try:
            from PIL import Image as PILImage
            pil = PILImage.open(file_path).convert("RGB")
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
