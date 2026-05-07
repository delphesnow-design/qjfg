import csv
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import EXPERIMENT_RESULT_DIR

RESULT_DIR = os.path.join(EXPERIMENT_RESULT_DIR, "ppm100")

results = []
for algo_name, ms in [("MediaPipe", 189.3), ("MODNet", 274.9), ("RVM", 29.6)]:
    csv_path = os.path.join(RESULT_DIR, f"{algo_name}_detail.csv")
    if not os.path.exists(csv_path):
        print(f"找不到: {csv_path}")
        continue

    mad_list, mse_list, iou_list = [], [], []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            if len(row) < 4 or row[0] == '平均' or row[0] == '':
                continue
            try:
                mad_list.append(float(row[1]))
                mse_list.append(float(row[2]))
                iou_list.append(float(row[3]))
            except:
                continue

    if not mad_list:
        print(f"{algo_name}: 无有效数据")
        continue

    avg_mad = float(np.mean(mad_list))
    avg_mse = float(np.mean(mse_list))
    avg_iou = float(np.mean(iou_list))
    n       = len(mad_list)

    print(f"{algo_name:<14} n={n:3d}  MAD={avg_mad:.4f}  "
          f"MSE={avg_mse:.4f}  IoU={avg_iou:.4f}  {ms:.1f}ms")
    results.append([algo_name, f"{avg_mad:.4f}", f"{avg_mse:.4f}",
                    f"{avg_iou:.4f}", f"{ms:.1f}", str(n)])

# 写汇总
summary_path = os.path.join(RESULT_DIR, "ppm100_summary_final.csv")
with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(["算法", "MAD(↓)", "MSE(↓)", "IoU(↑)", "推理时间(ms)", "有效样本数"])
    w.writerows(results)
print(f"\n已保存: {summary_path}")
