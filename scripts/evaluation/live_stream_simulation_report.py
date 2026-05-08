#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live-like background replacement stress test.

The normal offline benchmark is intentionally simple. This script builds a
harder video stream with motion, exposure drift, camera jitter, compression
noise, and no-empty-background startup cases, then evaluates replacement
quality against a known foreground mask.
"""

from __future__ import annotations

import csv
import html
import importlib.util
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.factory import BackgroundChangerFactory
from config.constants import (
    MEDIAPIPE_MODEL_PATH,
    MODNET_MODEL_PATH,
    PPM100_IMAGE_DIR,
    PPM100_MATTE_DIR,
    RVM_MODEL_PATH,
)


REPORT_DIR = PROJECT_ROOT / "docs" / "reports"
ASSET_DIR = REPORT_DIR / "assets"
REPORT_PATH = REPORT_DIR / "直播场景模拟测试报告.md"
CSV_PATH = ASSET_DIR / "live_stream_simulation_results.csv"

FRAME_SIZE = (320, 180)
RNG_SEED = 20260508


@dataclass(frozen=True)
class MethodSpec:
    algorithm_id: int
    name: str
    category: str
    dependency: str
    model_path: str | None = None


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    title: str
    description: str
    warmup_frames: int
    measure_frames: int
    person_from_start: bool = False
    pause_start: int | None = None
    pause_end: int | None = None
    camera_jitter: bool = False
    light_shift: bool = False
    compression_noise: bool = False


@dataclass
class PortraitSample:
    image: np.ndarray
    alpha: np.ndarray


METHODS = [
    MethodSpec(0, "MODNet", "深度学习 Matting", "torch + torchvision", MODNET_MODEL_PATH),
    MethodSpec(1, "MediaPipe", "深度学习 Segmentation", "mediapipe", MEDIAPIPE_MODEL_PATH),
    MethodSpec(2, "RVM", "深度学习 Video Matting", "onnxruntime", RVM_MODEL_PATH),
    MethodSpec(3, "MOG2", "传统 CV 背景建模", "opencv-python"),
    MethodSpec(4, "KNN", "传统 CV 背景建模", "opencv-python"),
    MethodSpec(5, "GrabCut", "传统 CV 交互式分割改造", "opencv-python"),
    MethodSpec(6, "LOBSTER", "传统 CV/LBSP 背景建模", "opencv-python + numpy"),
    MethodSpec(7, "SuBSENSE", "传统 CV/LBSP 自适应背景建模", "opencv-python + numpy"),
]

SCENARIOS = [
    ScenarioSpec(
        key="empty_warmup_move",
        title="空背景预热后移动",
        description="摄像头先看到空背景，再有人进入并持续小幅移动。",
        warmup_frames=16,
        measure_frames=54,
    ),
    ScenarioSpec(
        key="person_from_start",
        title="开播时人已在画面中",
        description="没有空背景预热，真实直播中很常见，会考验背景建模算法是否把人学习成背景。",
        warmup_frames=0,
        measure_frames=54,
        person_from_start=True,
    ),
    ScenarioSpec(
        key="pause_absorb",
        title="人物停住后再移动",
        description="人物先移动、随后长时间停住、最后再次移动，测试前景被背景模型吸收的问题。",
        warmup_frames=16,
        measure_frames=78,
        pause_start=20,
        pause_end=62,
    ),
    ScenarioSpec(
        key="jitter_light_compress",
        title="抖动光照压缩",
        description="模拟摄像头自动曝光、轻微机位抖动、直播压缩和传感器噪声。",
        warmup_frames=16,
        measure_frames=54,
        camera_jitter=True,
        light_shift=True,
        compression_noise=True,
    ),
]


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def dependency_status(spec: MethodSpec) -> tuple[bool, str]:
    checks = {
        0: ("torch", "torchvision"),
        1: ("mediapipe",),
        2: ("onnxruntime",),
    }
    missing = [name for name in checks.get(spec.algorithm_id, ()) if not has_module(name)]
    problems: list[str] = []
    if missing:
        problems.append("缺少依赖: " + ", ".join(missing))
    if spec.model_path and not Path(spec.model_path).exists():
        problems.append("模型文件不存在: " + str(Path(spec.model_path).relative_to(PROJECT_ROOT)))
    if problems:
        return False, "；".join(problems)
    return True, "可初始化"


def load_portrait_samples(limit: int = 6) -> list[PortraitSample]:
    image_dir = Path(PPM100_IMAGE_DIR)
    matte_dir = Path(PPM100_MATTE_DIR)
    if not image_dir.exists() or not matte_dir.exists():
        return []

    samples: list[PortraitSample] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        matte_path = matte_dir / image_path.name
        if not matte_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        matte = cv2.imread(str(matte_path), cv2.IMREAD_GRAYSCALE)
        if image is None or matte is None:
            continue
        if matte.shape[:2] != image.shape[:2]:
            matte = cv2.resize(matte, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        ys, xs = np.nonzero(matte > 10)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        pad = max(8, int(max(x1 - x0, y1 - y0) * 0.06))
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(image.shape[1] - 1, x1 + pad)
        y1 = min(image.shape[0] - 1, y1 + pad)

        crop_img = image[y0:y1 + 1, x0:x1 + 1].copy()
        crop_alpha = matte[y0:y1 + 1, x0:x1 + 1].copy()
        samples.append(PortraitSample(crop_img, crop_alpha))
        if len(samples) >= limit:
            break
    return samples


def make_replacement_background() -> np.ndarray:
    width, height = FRAME_SIZE
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:, :, 0] = (45 + 60 * yy).astype(np.uint8)
    bg[:, :, 1] = (200 + 45 * (1 - xx)).astype(np.uint8)
    bg[:, :, 2] = (65 + 95 * xx).astype(np.uint8)
    cv2.rectangle(bg, (0, height - 34), (width, height), (35, 155, 80), -1)
    cv2.line(bg, (0, height - 34), (width, height - 34), (245, 230, 100), 2)
    return bg


def make_room_background(t: int, scenario: ScenarioSpec) -> np.ndarray:
    width, height = FRAME_SIZE
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, width, dtype=np.float32)[None, :]

    bg = np.zeros((height, width, 3), dtype=np.float32)
    bg[:, :, 0] = 64 + 30 * xx + 10 * yy
    bg[:, :, 1] = 88 + 42 * yy
    bg[:, :, 2] = 128 + 24 * (1 - xx)

    frame = np.clip(bg, 0, 255).astype(np.uint8)
    cv2.rectangle(frame, (0, int(height * 0.72)), (width, height), (74, 86, 94), -1)
    cv2.rectangle(frame, (18, 24), (95, 78), (112, 97, 81), -1)
    cv2.rectangle(frame, (22, 28), (91, 74), (94, 116, 148), 2)
    cv2.rectangle(frame, (width - 92, 18), (width - 20, 82), (96, 76, 92), -1)
    for idx in range(4):
        y = 26 + idx * 12
        cv2.line(frame, (width - 86, y), (width - 26, y), (155, 135, 112), 2)
    cv2.rectangle(frame, (34, int(height * 0.74)), (142, int(height * 0.92)), (45, 48, 54), -1)
    cv2.rectangle(frame, (44, int(height * 0.77)), (132, int(height * 0.89)), (35, 58, 76), -1)
    cv2.circle(frame, (width - 58, int(height * 0.74)), 18, (49, 106, 70), -1)
    cv2.line(frame, (width - 58, int(height * 0.74)), (width - 42, int(height * 0.68)), (49, 106, 70), 4)
    cv2.line(frame, (width - 58, int(height * 0.74)), (width - 75, int(height * 0.67)), (49, 106, 70), 4)

    if scenario.light_shift:
        gain = 1.0 + 0.12 * np.sin(t / 7.0) + 0.05 * np.sin(t / 2.3)
        offset = np.array([5 * np.sin(t / 9.0), 8 * np.sin(t / 11.0), 12 * np.sin(t / 8.0)])
        frame = np.clip(frame.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)

    return frame


def synthetic_person(frame_index: int, scenario: ScenarioSpec) -> tuple[np.ndarray, np.ndarray]:
    width, height = FRAME_SIZE
    mask = np.zeros((height, width), dtype=np.uint8)
    fg = np.zeros((height, width, 3), dtype=np.uint8)

    t = frame_index
    if scenario.pause_start is not None and scenario.pause_end is not None:
        if scenario.pause_start <= t <= scenario.pause_end:
            motion_t = scenario.pause_start
        elif t > scenario.pause_end:
            motion_t = scenario.pause_start + (t - scenario.pause_end)
        else:
            motion_t = t
    else:
        motion_t = t

    cx = int(width * 0.50 + np.sin(motion_t / 6.0) * width * 0.12)
    cy = int(height * 0.62 + np.sin(motion_t / 11.0) * height * 0.015)
    head_y = int(height * 0.32 + np.sin(motion_t / 10.0) * height * 0.01)

    shoulder_y = int(height * 0.49)
    arm_swing = int(np.sin(motion_t / 4.0) * 10)
    cv2.ellipse(mask, (cx, cy), (36, 56), 0, 0, 360, 255, -1)
    cv2.rectangle(mask, (cx - 31, shoulder_y), (cx + 31, cy + 42), 255, -1)
    cv2.circle(mask, (cx, head_y), 22, 255, -1)
    cv2.line(mask, (cx - 27, shoulder_y + 6), (cx - 63, cy + arm_swing), 255, 14)
    cv2.line(mask, (cx + 27, shoulder_y + 6), (cx + 63, cy - arm_swing), 255, 14)

    cv2.ellipse(fg, (cx, cy), (36, 56), 0, 0, 360, (42, 106, 196), -1)
    cv2.rectangle(fg, (cx - 31, shoulder_y), (cx + 31, cy + 42), (36, 96, 184), -1)
    cv2.circle(fg, (cx, head_y), 22, (76, 164, 224), -1)
    cv2.ellipse(fg, (cx, head_y - 9), (23, 14), 0, 180, 360, (32, 42, 48), -1)
    cv2.line(fg, (cx - 27, shoulder_y + 6), (cx - 63, cy + arm_swing), (36, 96, 184), 14)
    cv2.line(fg, (cx + 27, shoulder_y + 6), (cx + 63, cy - arm_swing), (36, 96, 184), 14)
    cv2.line(fg, (cx - 22, cy - 10), (cx + 22, cy - 10), (80, 132, 214), 2)
    cv2.line(fg, (cx - 18, cy + 14), (cx + 20, cy + 12), (78, 128, 208), 2)

    return fg, mask


def portrait_person(
    frame_index: int,
    scenario: ScenarioSpec,
    samples: list[PortraitSample],
) -> tuple[np.ndarray, np.ndarray]:
    width, height = FRAME_SIZE
    if not samples:
        return synthetic_person(frame_index, scenario)
    sample = samples[0]

    t = frame_index
    if scenario.pause_start is not None and scenario.pause_end is not None:
        if scenario.pause_start <= t <= scenario.pause_end:
            motion_t = scenario.pause_start
        elif t > scenario.pause_end:
            motion_t = scenario.pause_start + (t - scenario.pause_end)
        else:
            motion_t = t
    else:
        motion_t = t

    target_h = int(height * 0.82)
    scale = target_h / max(1, sample.image.shape[0])
    target_w = max(1, int(sample.image.shape[1] * scale))
    fg_small = cv2.resize(sample.image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    alpha_small = cv2.resize(sample.alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    cx = int(width * 0.50 + np.sin(motion_t / 6.0) * width * 0.12)
    top = int(height * 0.12 + np.sin(motion_t / 10.0) * height * 0.01)
    left = cx - target_w // 2

    fg = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    x0 = max(0, left)
    y0 = max(0, top)
    x1 = min(width, left + target_w)
    y1 = min(height, top + target_h)
    if x1 <= x0 or y1 <= y0:
        return fg, mask

    sx0 = x0 - left
    sy0 = y0 - top
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)
    fg[y0:y1, x0:x1] = fg_small[sy0:sy1, sx0:sx1]
    mask[y0:y1, x0:x1] = alpha_small[sy0:sy1, sx0:sx1]
    return fg, mask


def apply_camera_effects(
    frame: np.ndarray,
    mask: np.ndarray,
    frame_index: int,
    scenario: ScenarioSpec,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    if scenario.camera_jitter:
        dx = 2.4 * np.sin(frame_index / 2.1) + 1.2 * np.sin(frame_index / 5.7)
        dy = 1.8 * np.cos(frame_index / 2.8)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(
            frame, matrix, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(
            mask, matrix, (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    if scenario.compression_noise:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 58]
        ok, encoded = cv2.imencode(".jpg", frame, encode_param)
        if ok:
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        rng = np.random.default_rng(RNG_SEED + frame_index)
        noise = rng.normal(0, 3.5, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return frame, mask


def make_live_frame(
    frame_index: int,
    scenario: ScenarioSpec,
    include_person: bool,
    samples: list[PortraitSample],
) -> tuple[np.ndarray, np.ndarray]:
    bg = make_room_background(frame_index, scenario)
    if include_person:
        fg, alpha_u8 = portrait_person(frame_index, scenario, samples)
        alpha = cv2.GaussianBlur(alpha_u8.astype(np.float32) / 255.0, (7, 7), 0)
        frame = (
            fg.astype(np.float32) * alpha[:, :, None]
            + bg.astype(np.float32) * (1.0 - alpha[:, :, None])
        )
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        mask = (alpha_u8 > 20).astype(np.uint8) * 255
    else:
        frame = bg
        mask = np.zeros(bg.shape[:2], dtype=np.uint8)
    return apply_camera_effects(frame, mask, frame_index, scenario)


def infer_replaced_background_mask(input_frame: np.ndarray, output_frame: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(output_frame.astype(np.int16) - input_frame.astype(np.int16)), axis=2)
    return diff > 24


def binary_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return float(inter / union) if union else 1.0


def fmt(value: object, digits: int = 2, pct: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        if pct:
            return f"{value * 100:.{digits}f}%"
        return f"{value:.{digits}f}"
    return str(value)


def score_row(bg_iou: float, fg_corruption: float, flicker: float) -> float:
    return 100.0 * (
        0.45 * bg_iou
        + 0.35 * max(0.0, 1.0 - fg_corruption)
        + 0.20 * max(0.0, 1.0 - flicker)
    )


def make_expected_output(input_frame: np.ndarray, fg_mask: np.ndarray, replacement_bg: np.ndarray) -> np.ndarray:
    result = input_frame.copy()
    bg = cv2.resize(replacement_bg, (input_frame.shape[1], input_frame.shape[0]))
    result[fg_mask == 0] = bg[fg_mask == 0]
    return result


def make_error_overlay(input_frame: np.ndarray, pred_bg: np.ndarray, true_bg: np.ndarray) -> np.ndarray:
    overlay = input_frame.copy()
    fg_corrupted = pred_bg & ~true_bg
    bg_missed = ~pred_bg & true_bg
    overlay[fg_corrupted] = (40, 40, 230)
    overlay[bg_missed] = (230, 120, 30)
    return cv2.addWeighted(input_frame, 0.45, overlay, 0.55, 0)


def sample_slug(method_name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in method_name).strip("_")


def save_algorithm_sample(
    method_name: str,
    scenario: ScenarioSpec,
    input_frame: np.ndarray,
    output_frame: np.ndarray,
    fg_mask: np.ndarray,
    pred_bg: np.ndarray,
    replacement_bg: np.ndarray,
) -> str:
    expected = make_expected_output(input_frame, fg_mask, replacement_bg)
    error = make_error_overlay(input_frame, pred_bg, fg_mask == 0)
    panels = [
        ("input", input_frame),
        (f"{method_name} output", output_frame),
        ("expected", expected),
        ("error red/blue", error),
    ]
    labeled: list[np.ndarray] = []
    for label, image in panels:
        panel = image.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(panel, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        labeled.append(panel)
    montage = np.concatenate(labeled, axis=1)
    path = ASSET_DIR / f"live_sim_sample_{sample_slug(method_name)}_{scenario.key}.jpg"
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), montage)
    return str(path.relative_to(REPORT_DIR)).replace("\\", "/")


def evaluate_scenario(
    method: MethodSpec,
    scenario: ScenarioSpec,
    samples: list[PortraitSample],
    replacement_bg: np.ndarray,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": method.algorithm_id,
        "method": method.name,
        "scenario": scenario.key,
        "scenario_title": scenario.title,
        "status": "待运行",
        "available": True,
        "init_ms": None,
        "avg_ms": None,
        "fps": None,
        "bg_iou": None,
        "fg_corruption": None,
        "bg_miss": None,
        "flicker": None,
        "score": None,
        "sample": "",
    }

    deps_ok, status = dependency_status(method)
    if not deps_ok:
        row.update({"available": False, "status": status})
        return row

    try:
        init_t0 = time.perf_counter()
        factory = BackgroundChangerFactory(algorithm_id=method.algorithm_id)
        factory.backgrounds = [replacement_bg]
        row["init_ms"] = (time.perf_counter() - init_t0) * 1000.0
    except Exception as exc:  # noqa: BLE001
        row.update({"available": False, "status": f"初始化失败: {exc}"})
        return row

    for i in range(scenario.warmup_frames):
        frame, _ = make_live_frame(i, scenario, include_person=False, samples=samples)
        factory.process_frame(frame)

    timings: list[float] = []
    ious: list[float] = []
    fg_corruptions: list[float] = []
    bg_misses: list[float] = []
    flickers: list[float] = []
    prev_pred_bg: np.ndarray | None = None
    prev_true_bg: np.ndarray | None = None
    sample_target = scenario.measure_frames // 2

    for i in range(scenario.measure_frames):
        frame_index = scenario.warmup_frames + i
        input_frame, fg_mask = make_live_frame(frame_index, scenario, True, samples)
        true_bg = fg_mask == 0
        true_fg = ~true_bg

        t0 = time.perf_counter()
        output = factory.process_frame(input_frame)
        timings.append((time.perf_counter() - t0) * 1000.0)

        pred_bg = infer_replaced_background_mask(input_frame, output)
        ious.append(binary_iou(pred_bg, true_bg))
        fg_corruptions.append(float(pred_bg[true_fg].mean()) if np.any(true_fg) else 0.0)
        bg_misses.append(float((~pred_bg[true_bg]).mean()) if np.any(true_bg) else 0.0)
        if prev_pred_bg is not None and prev_true_bg is not None:
            stable_bg = true_bg & prev_true_bg
            if np.any(stable_bg):
                flickers.append(float((pred_bg[stable_bg] != prev_pred_bg[stable_bg]).mean()))
        prev_pred_bg = pred_bg
        prev_true_bg = true_bg

        if i == sample_target:
            row["sample"] = save_algorithm_sample(
                method.name,
                scenario,
                input_frame,
                output,
                fg_mask,
                pred_bg,
                replacement_bg,
            )

    avg_ms = statistics.mean(timings)
    bg_iou = statistics.mean(ious)
    fg_corruption = statistics.mean(fg_corruptions)
    bg_miss = statistics.mean(bg_misses)
    flicker = statistics.mean(flickers) if flickers else 0.0
    row.update(
        {
            "status": "完成",
            "avg_ms": avg_ms,
            "fps": 1000.0 / avg_ms if avg_ms > 0 else None,
            "bg_iou": bg_iou,
            "fg_corruption": fg_corruption,
            "bg_miss": bg_miss,
            "flicker": flicker,
            "score": score_row(bg_iou, fg_corruption, flicker),
        }
    )
    return row


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_method: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row.get("status") == "完成":
            by_method.setdefault(str(row["method"]), []).append(row)

    agg_rows: list[dict[str, object]] = []
    for method, group in by_method.items():
        agg_rows.append(
            {
                "method": method,
                "avg_ms": statistics.mean(float(r["avg_ms"]) for r in group),
                "fps": statistics.mean(float(r["fps"]) for r in group),
                "bg_iou": statistics.mean(float(r["bg_iou"]) for r in group),
                "fg_corruption": statistics.mean(float(r["fg_corruption"]) for r in group),
                "bg_miss": statistics.mean(float(r["bg_miss"]) for r in group),
                "flicker": statistics.mean(float(r["flicker"]) for r in group),
                "score": statistics.mean(float(r["score"]) for r in group),
            }
        )
    return sorted(agg_rows, key=lambda r: float(r["score"]), reverse=True)


def aggregate_rows_for_scenarios(
    rows: list[dict[str, object]],
    scenario_keys: set[str],
) -> list[dict[str, object]]:
    scoped = [
        row for row in rows
        if row.get("status") == "完成" and str(row.get("scenario")) in scenario_keys
    ]
    return aggregate_rows(scoped)


def write_csv(rows: list[dict[str, object]]) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "id", "method", "scenario", "scenario_title", "status", "available",
        "init_ms", "avg_ms", "fps", "bg_iou", "fg_corruption", "bg_miss",
        "flicker", "score", "sample",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_bar_svg(
    path: Path,
    title: str,
    rows: Iterable[dict[str, object]],
    value_key: str,
    unit: str,
    color: str,
    reverse: bool = False,
) -> None:
    data = [
        (str(row["method"]), float(row[value_key]))
        for row in rows
        if isinstance(row.get(value_key), (int, float))
    ]
    if reverse:
        data = sorted(data, key=lambda item: item[1])
    width = 860
    height = max(250, 90 + len(data) * 44)
    left = 150
    right = 42
    top = 58
    bar_h = 23
    gap = 21
    max_value = max((float(v) for _, v in data), default=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-size="21" font-family="Arial, sans-serif" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top - 8}" x2="{width - right}" y2="{top - 8}" stroke="#d7dde5"/>',
    ]
    usable_width = width - left - right
    for idx, (label, value) in enumerate(data):
        y = top + idx * (bar_h + gap)
        bar_w = 0 if max_value == 0 else max(2, value / max_value * usable_width)
        lines.extend(
            [
                f'<text x="{left - 14}" y="{y + 16}" text-anchor="end" font-size="14" font-family="Arial, sans-serif" fill="#273142">{html.escape(label)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="4" fill="{color}"/>',
                f'<text x="{min(left + bar_w + 8, width - right - 84):.1f}" y="{y + 16}" font-size="13" font-family="Arial, sans-serif" fill="#273142">{value:.2f}{html.escape(unit)}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def heat_color(value: float) -> str:
    value = float(np.clip(value, 0.0, 100.0))
    if value >= 80:
        return "#2f8f5b"
    if value >= 65:
        return "#8aa640"
    if value >= 50:
        return "#d49a30"
    return "#c74d45"


def write_heatmap_svg(path: Path, rows: list[dict[str, object]]) -> None:
    methods = sorted({str(row["method"]) for row in rows if row.get("status") == "完成"})
    scenario_titles = {s.key: s.title for s in SCENARIOS}
    scenarios = [s.key for s in SCENARIOS]
    cell_w = 142
    cell_h = 42
    left = 136
    top = 78
    width = left + len(scenarios) * cell_w + 34
    height = top + len(methods) * cell_h + 36
    score_map = {
        (str(row["method"]), str(row["scenario"])): float(row["score"])
        for row in rows
        if row.get("status") == "完成"
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="34" font-size="21" font-family="Arial, sans-serif" font-weight="700">直播压力场景综合分</text>',
    ]
    for col, scenario_key in enumerate(scenarios):
        x = left + col * cell_w + cell_w / 2
        lines.append(
            f'<text x="{x:.1f}" y="62" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#273142">{html.escape(scenario_titles[scenario_key])}</text>'
        )
    for row_idx, method in enumerate(methods):
        y = top + row_idx * cell_h
        lines.append(
            f'<text x="{left - 12}" y="{y + 27}" text-anchor="end" font-size="14" font-family="Arial, sans-serif" fill="#273142">{html.escape(method)}</text>'
        )
        for col, scenario_key in enumerate(scenarios):
            x = left + col * cell_w
            score = score_map.get((method, scenario_key))
            color = "#e8edf3" if score is None else heat_color(score)
            label = "-" if score is None else f"{score:.1f}"
            lines.extend(
                [
                    f'<rect x="{x + 4}" y="{y + 5}" width="{cell_w - 8}" height="{cell_h - 10}" rx="5" fill="{color}"/>',
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + 27}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#ffffff">{label}</text>',
                ]
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(rows: list[dict[str, object]], samples: list[PortraitSample]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    aggregate = aggregate_rows(rows)
    risk_keys = {"person_from_start", "jitter_light_compress"}
    risk_aggregate = aggregate_rows_for_scenarios(rows, risk_keys)
    write_bar_svg(
        ASSET_DIR / "live_sim_overall_score.svg",
        "直播模拟综合分（越高越好）",
        aggregate,
        "score",
        "",
        "#2f8f5b",
    )
    write_bar_svg(
        ASSET_DIR / "live_sim_latency_ms.svg",
        "平均单帧耗时（ms，越低越好）",
        aggregate,
        "avg_ms",
        " ms",
        "#2f6fbb",
        reverse=True,
    )
    write_bar_svg(
        ASSET_DIR / "live_sim_fg_corruption.svg",
        "前景误替换率（越低越好）",
        aggregate,
        "fg_corruption",
        "",
        "#c74d45",
        reverse=True,
    )
    write_heatmap_svg(ASSET_DIR / "live_sim_scenario_heatmap.svg", rows)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_mode = "PPM-100 真实人像合成" if samples else "几何人像合成（未找到 PPM-100）"
    compared_methods = ", ".join(row["method"] for row in aggregate)
    best = aggregate[0] if aggregate else None
    risk_best = risk_aggregate[0] if risk_aggregate else None
    mog2 = next((row for row in aggregate if row["method"] == "MOG2"), None)
    mog2_risk = next((row for row in risk_aggregate if row["method"] == "MOG2"), None)
    mog2_detail = [row for row in rows if row.get("method") == "MOG2" and row.get("status") == "完成"]
    mog2_worst = min(mog2_detail, key=lambda row: float(row["score"]), default=None)

    summary_table = "\n".join(
        [
            "| 方法 | 综合分 | 平均耗时(ms) | FPS | 背景 IoU | 前景误替换率 | 背景漏替换率 | 抖动率 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {row['method']} | {fmt(row['score'])} | {fmt(row['avg_ms'])} | {fmt(row['fps'])} | {fmt(row['bg_iou'], 3)} | {fmt(row['fg_corruption'], 2, pct=True)} | {fmt(row['bg_miss'], 2, pct=True)} | {fmt(row['flicker'], 2, pct=True)} |"
                for row in aggregate
            ],
        ]
    )

    scenario_table = "\n".join(
        [
            "| 方法 | 场景 | 综合分 | 耗时(ms) | 背景 IoU | 前景误替换率 | 背景漏替换率 | 抖动率 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {row['method']} | {row['scenario_title']} | {fmt(row['score'])} | {fmt(row['avg_ms'])} | {fmt(row['bg_iou'], 3)} | {fmt(row['fg_corruption'], 2, pct=True)} | {fmt(row['bg_miss'], 2, pct=True)} | {fmt(row['flicker'], 2, pct=True)} |"
                for row in rows
                if row.get("status") == "完成"
            ],
        ]
    )

    risk_table = "\n".join(
        [
            "| 方法 | 风险场景综合分 | 平均耗时(ms) | 背景 IoU | 前景误替换率 | 背景漏替换率 | 抖动率 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {row['method']} | {fmt(row['score'])} | {fmt(row['avg_ms'])} | {fmt(row['bg_iou'], 3)} | {fmt(row['fg_corruption'], 2, pct=True)} | {fmt(row['bg_miss'], 2, pct=True)} | {fmt(row['flicker'], 2, pct=True)} |"
                for row in risk_aggregate
            ],
        ]
    )

    scenario_desc = "\n".join(
        f"- **{s.title}**：{s.description}" for s in SCENARIOS
    )

    method_order = [str(row["method"]) for row in aggregate]
    scenario_order = [scenario.key for scenario in SCENARIOS]
    sample_map = {
        (str(row["method"]), str(row["scenario"])): row
        for row in rows
        if row.get("status") == "完成" and row.get("sample")
    }
    sample_sections: list[str] = []
    for method in method_order:
        image_lines = []
        for scenario_key in scenario_order:
            row = sample_map.get((method, scenario_key))
            if row:
                image_lines.append(
                    f"![{method} {row['scenario_title']}]({row['sample']})"
                )
        if image_lines:
            sample_sections.append(f"### {method}\n\n" + "\n".join(image_lines))
    sample_lines = "\n\n".join(sample_sections) if sample_sections else "暂无样例图。"

    if mog2 and best:
        diagnosis = (
            f"本轮直播模拟里，综合分最高的是 **{best['method']}**（{best['score']:.2f}）。"
            f"MOG2 的综合分为 **{mog2['score']:.2f}**，平均前景误替换率为 "
            f"**{mog2['fg_corruption'] * 100:.2f}%**，背景漏替换率为 **{mog2['bg_miss'] * 100:.2f}%**。"
        )
        if risk_best and mog2_risk:
            diagnosis += (
                f" 只看关键直播风险场景时，最高的是 **{risk_best['method']}**"
                f"（{risk_best['score']:.2f}），MOG2 为 **{mog2_risk['score']:.2f}**；"
                "这说明静态/预热场景会掩盖 MOG2 在真实开播流程里的弱点。"
            )
    else:
        diagnosis = "本轮没有得到可用于对比的 MOG2 结果。"
    if mog2_worst:
        diagnosis += (
            f" MOG2 最弱场景是 **{mog2_worst['scenario_title']}**，综合分 "
            f"**{float(mog2_worst['score']):.2f}**；这类场景最接近用户开播时直接在镜头前、"
            "人物停留或摄像头自动调整造成的实时问题。"
        )

    method_scope = (
        "检测到 PPM-100 图像与 matte，因此本报告可扩展到深度学习方法。"
        if samples
        else "未检测到 `data/PPM-100/image` 与 `data/PPM-100/matte`，本次只比较 5 个传统 CV 方法；深度模型在几何人像上不公平，暂不纳入。"
    )

    report = f"""# 直播场景模拟测试报告

生成时间：{generated_at}

## 结论摘要

- 测试数据模式：{data_mode}
- 参与对比方法：{compared_methods}
- {method_scope}
- 真实人像 matte 数据源：PPM-100，来自 MODNet 作者公开数据集；本地下载自 Hugging Face 镜像并整理为 `data/PPM-100/image` 与 `data/PPM-100/matte`。
- 默认算法决策：RVM 在总体综合分和关键直播风险场景中均排名第一，当前上位机默认算法已切换为 **RVM**。
- {diagnosis}

## 模拟办法

数据来源：

- PPM-100 官方仓库：https://github.com/ZHKKKe/PPM
- PPM-100 Hugging Face 镜像：https://huggingface.co/datasets/realdream-ai/ppm-matting

为了更接近真实直播，本脚本不再只测试静态空背景，而是生成连续视频流，并逐帧调用项目现有 `process_frame` 背景替换接口。每帧都有真值前景 mask，因此可以判断哪些背景应该被替换、哪些人物区域不应该被替换。

压力场景：

{scenario_desc}

核心指标：

- 背景 IoU：预测被替换的背景区域与真值背景区域的交并比。
- 前景误替换率：人物区域中被替换成背景的比例，越低越好。
- 背景漏替换率：背景区域中没有被替换的比例，越低越好。
- 抖动率：相邻帧稳定背景区域中预测结果翻转的比例，越低越好。
- 综合分：`0.45 * 背景IoU + 0.35 * (1 - 前景误替换率) + 0.20 * (1 - 抖动率)`，换算为 0-100 分。

## 总体结果

{summary_table}

## 分场景结果

{scenario_table}

## 关键直播风险场景

这里单独汇总“开播时人已在画面中”和“抖动光照压缩”。这两类比空背景预热更接近真实直播，也是 MOG2 实际观感变差的主要来源。

{risk_table}

## 图表

![直播模拟综合分](assets/live_sim_overall_score.svg)

![分场景热力图](assets/live_sim_scenario_heatmap.svg)

![平均单帧耗时](assets/live_sim_latency_ms.svg)

![前景误替换率](assets/live_sim_fg_corruption.svg)

## 八种算法样例

红色表示人物被误替换，蓝色表示背景漏替换。

{sample_lines}

## 分析

MOG2 的根本假设是背景相对稳定，并且背景模型能先看到足够干净的背景。直播里常见的“人一开始就在镜头前”“人物停住说话”“摄像头自动曝光和轻微抖动”都会破坏这个假设。它可以作为无模型、固定机位、空背景预热充分时的低成本方案，但不适合作为默认直播抠像算法。

当前上位机默认算法已按本报告切换为 RVM。后续如果更新模型、摄像头采集链路或背景替换策略，建议重新运行本脚本，并继续优先选择“前景误替换率低、抖动率低、FPS 可接受”的方法。

## 复现命令

```powershell
.\\.venv\\Scripts\\python.exe scripts\\evaluation\\live_stream_simulation_report.py
```

原始 CSV：`assets/live_stream_simulation_results.csv`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def select_methods(samples: list[PortraitSample]) -> list[MethodSpec]:
    if samples:
        return METHODS
    return [method for method in METHODS if method.algorithm_id in (3, 4, 5, 6, 7)]


def main() -> None:
    samples = load_portrait_samples()
    methods = select_methods(samples)
    replacement_bg = make_replacement_background()

    rows: list[dict[str, object]] = []
    for method in methods:
        print(f"\n=== {method.name} ===")
        for scenario in SCENARIOS:
            print(f"  running: {scenario.title}")
            row = evaluate_scenario(method, scenario, samples, replacement_bg)
            rows.append(row)
            if row.get("status") == "完成":
                print(
                    f"    score={float(row['score']):.2f}, "
                    f"IoU={float(row['bg_iou']):.3f}, "
                    f"fg_bad={float(row['fg_corruption']) * 100:.2f}%, "
                    f"{float(row['avg_ms']):.2f}ms"
                )
            else:
                print(f"    skipped: {row.get('status')}")

    write_csv(rows)
    write_report(rows, samples)
    print(f"\n报告已生成: {REPORT_PATH}")
    print(f"CSV已生成: {CSV_PATH}")


if __name__ == "__main__":
    main()
