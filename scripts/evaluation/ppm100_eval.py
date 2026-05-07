#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPM-100 客观评估 - 直接获取掩模版"""

import os, sys, cv2, numpy as np, csv, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

for lib in ['torch', 'mediapipe', 'onnxruntime']:
    try: __import__(lib)
    except ImportError: pass

from algorithms.factory import BackgroundChangerFactory
from config.constants import (
    EXPERIMENT_RESULT_DIR,
    GREEN_BACKGROUND_DIR,
    PPM100_IMAGE_DIR,
    PPM100_MATTE_DIR,
)

PPM_IMAGE_DIR = PPM100_IMAGE_DIR
PPM_MATTE_DIR = PPM100_MATTE_DIR
RESULT_DIR    = os.path.join(EXPERIMENT_RESULT_DIR, "ppm100")
os.makedirs(RESULT_DIR, exist_ok=True)

GREEN_BG_DIR  = GREEN_BACKGROUND_DIR
os.makedirs(GREEN_BG_DIR, exist_ok=True)
GREEN_BG_PATH = os.path.join(GREEN_BG_DIR, "green.jpg")
if not os.path.exists(GREEN_BG_PATH):
    g = np.zeros((480, 640, 3), dtype=np.uint8)
    g[:] = (0, 255, 0)
    cv2.imwrite(GREEN_BG_PATH, g)


def compute_metrics(pred, gt):
    p   = pred.astype(np.float32) / 255.0
    g   = gt.astype(np.float32)   / 255.0
    mad = float(np.mean(np.abs(p - g)))
    mse = float(np.mean((p - g) ** 2))
    pb  = (p > 0.5).astype(np.uint8)
    gb  = (g > 0.5).astype(np.uint8)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    iou   = float(inter / union) if union > 0 else 1.0
    return mad, mse, iou


def get_mask_mediapipe(changer, image):
    import mediapipe as mp
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = changer.segmenter.segment(mp_image)

    # confidence_masks是长度6的列表
    # 索引0=背景，索引1-5=前景各类别
    # 取索引1到5的最大值作为前景置信度
    masks = result.confidence_masks
    fg = np.zeros_like(masks[0].numpy_view(), dtype=np.float32)
    for i in range(1, len(masks)):   # 索引1到5
        fg = np.maximum(fg, masks[i].numpy_view())

    # 二值化，阈值0.5
    return (fg > 0.5).astype(np.uint8) * 255


def get_mask_modnet(changer, image):
    """
    直接调用MODNet内部推理，返回Alpha Matte
    通过process_frame + 已知绿色背景反推（MODNet Alpha Matte精度高，反推可靠）
    """
    # MODNet在第一次运行已验证数据可靠（IoU=0.9628），直接用process_frame反推
    h, w = image.shape[:2]
    green = np.zeros((h, w, 3), dtype=np.uint8)
    green[:] = (0, 255, 0)
    changer.backgrounds              = [green]
    changer.current_background_index = 0
    result = changer.process_frame(image)

    # MODNet输出Alpha Matte，合成公式P=F*alpha+G*(1-alpha)
    # 因此 alpha = (P_g - G_g) / (F_g - G_g) 在绿色通道上
    f  = image[:, :, 1].astype(np.float32)   # 原图绿色通道
    p  = result[:, :, 1].astype(np.float32)  # 合成图绿色通道
    g  = 255.0                                # 背景绿色通道=255

    denom = f - g
    denom = np.where(np.abs(denom) < 1e-3, 1e-3, denom)
    alpha = (p - g) / denom
    alpha = np.clip(alpha, 0.0, 1.0)
    return (alpha * 255).astype(np.uint8)


def get_mask_rvm(changer, image):
    """
    直接调用RVM的_run_rvm_inference私有方法
    """
    # 对齐到32的倍数
    h, w = image.shape[:2]
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    img   = cv2.resize(image, (new_w, new_h), cv2.INTER_AREA) \
            if (new_h != h or new_w != w) else image.copy()

    try:
        pha = changer._run_rvm_inference(img)  # 返回Alpha Matte float32 [0,1]
        if pha is None:
            return None
        # resize回原尺寸
        if pha.shape[:2] != (h, w):
            pha = cv2.resize(pha, (w, h), cv2.INTER_LINEAR)
        return (np.clip(pha, 0, 1) * 255).astype(np.uint8)
    except Exception as e:
        print(f"    _run_rvm_inference失败: {e}")
        return None


def evaluate_model(algo_id, algo_name, image_files):
    print(f"\n{'='*55}")
    print(f"评估: {algo_name}  (algorithm_id={algo_id})")
    print(f"{'='*55}")

    factory = BackgroundChangerFactory(algorithm_id=algo_id)
    factory.load_backgrounds(GREEN_BG_DIR)
    changer = factory.changer

    mad_list, mse_list, iou_list, ms_list = [], [], [], []
    rows = []

    for i, fname in enumerate(image_files):
        img_path   = os.path.join(PPM_IMAGE_DIR, fname)
        matte_path = os.path.join(PPM_MATTE_DIR, fname)

        if not os.path.exists(matte_path):
            continue

        image = cv2.imread(img_path)
        gt    = cv2.imread(matte_path, cv2.IMREAD_GRAYSCALE)
        if image is None or gt is None:
            continue

        # 缩放至720p
        h, w = image.shape[:2]
        if max(h, w) > 720:
            scale  = 720 / max(h, w)
            new_wh = (int(w * scale), int(h * scale))
            image  = cv2.resize(image, new_wh, cv2.INTER_AREA)
            gt     = cv2.resize(gt,    new_wh, cv2.INTER_AREA)

        try:
            t0 = time.time()

            if algo_name == "MediaPipe":
                pred = get_mask_mediapipe(changer, image)
            elif algo_name == "MODNet":
                pred = get_mask_modnet(changer, image)
            elif algo_name == "RVM":
                pred = get_mask_rvm(changer, image)
            else:
                pred = None

            t1 = time.time()
        except Exception as e:
            print(f"  [{i+1:03d}] 失败: {e}")
            continue

        if pred is None:
            print(f"  [{i+1:03d}] 掩模为None，跳过")
            continue

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        mad, mse, iou = compute_metrics(pred, gt)
        elapsed_ms    = (t1 - t0) * 1000

        mad_list.append(mad)
        mse_list.append(mse)
        iou_list.append(iou)
        ms_list.append(elapsed_ms)
        rows.append([fname, f"{mad:.4f}", f"{mse:.4f}",
                     f"{iou:.4f}", f"{elapsed_ms:.1f}"])

        print(f"  [{i+1:03d}/100] MAD={mad:.4f}  MSE={mse:.4f}"
              f"  IoU={iou:.4f}  {elapsed_ms:.0f}ms")

    if not mad_list:
        print(f"  {algo_name} 无有效结果")
        return None

    avg_mad = float(np.mean(mad_list))
    avg_mse = float(np.mean(mse_list))
    avg_iou = float(np.mean(iou_list))
    avg_ms  = float(np.mean(ms_list))

    print(f"\n{algo_name} 平均:  MAD={avg_mad:.4f}  "
          f"MSE={avg_mse:.4f}  IoU={avg_iou:.4f}  {avg_ms:.1f}ms/帧")

    csv_path = os.path.join(RESULT_DIR, f"{algo_name}_detail.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["文件名", "MAD", "MSE", "IoU", "推理时间(ms)"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["平均", f"{avg_mad:.4f}", f"{avg_mse:.4f}",
                    f"{avg_iou:.4f}", f"{avg_ms:.1f}"])
    print(f"  已保存: {csv_path}")

    return {"algorithm": algo_name,
            "MAD": avg_mad, "MSE": avg_mse,
            "IoU": avg_iou, "ms": avg_ms}


if __name__ == "__main__":
    image_files = sorted([
        f for f in os.listdir(PPM_IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"找到 {len(image_files)} 张图像")

    # 三个模型全部重跑
    #models = [(1, "MediaPipe"), (0, "MODNet"), (2, "RVM")]
    #models = [(1, "MediaPipe")]
    models = [(1, "MediaPipe"), (0, "MODNet"), (2, "RVM")]
    all_results = []

    for algo_id, algo_name in models:
        r = evaluate_model(algo_id, algo_name, image_files)
        if r:
            all_results.append(r)
        print(f"\n按回车继续...")
        input()

    print("\n" + "="*60)
    print("PPM-100 客观评估汇总")
    print("="*60)
    print(f"{'算法':<14}{'MAD(↓)':<10}{'MSE(↓)':<10}{'IoU(↑)':<10}{'时间(ms)'}")
    print("-"*60)
    for r in all_results:
        print(f"{r['algorithm']:<14}{r['MAD']:<10.4f}"
              f"{r['MSE']:<10.4f}{r['IoU']:<10.4f}{r['ms']:.1f}")

    summary = os.path.join(RESULT_DIR, "ppm100_summary.csv")
    with open(summary, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["算法", "MAD(↓)", "MSE(↓)", "IoU(↑)", "平均推理时间(ms)"])
        for r in all_results:
            w.writerow([r['algorithm'], f"{r['MAD']:.4f}",
                        f"{r['MSE']:.4f}", f"{r['IoU']:.4f}",
                        f"{r['ms']:.1f}"])
    print(f"\n汇总CSV: {summary}")
    print("实验完成！")
