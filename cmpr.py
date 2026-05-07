# check_mediapipe_result.py
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for lib in ['torch', 'mediapipe', 'onnxruntime']:
    try: __import__(lib)
    except ImportError: pass

import mediapipe as mp
from algorithms.factory import BackgroundChangerFactory

factory = BackgroundChangerFactory(algorithm_id=1)
changer = factory.changer

# 用一张真实PPM-100图测试
img_path = r"C:\Users\19800\Desktop\MY\cs\qianjingfengge\PPM-100\image\13179159164_1a4ae8d085_o.jpg"
image = cv2.imread(img_path)
h, w = image.shape[:2]
scale = 720 / max(h, w)
image = cv2.resize(image, (int(w*scale), int(h*scale)), cv2.INTER_AREA)

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = changer.segmenter.segment(mp_image)

print("result类型:", type(result))
print("result所有属性:", [x for x in dir(result) if not x.startswith('__')])
print()

# 逐一检查每个属性的值
for attr in [x for x in dir(result) if not x.startswith('__')]:
    try:
        val = getattr(result, attr)
        if callable(val):
            continue
        print(f"{attr}: {type(val).__name__} = ", end="")
        if val is None:
            print("None")
        elif hasattr(val, '__len__'):
            print(f"长度{len(val)}", end="")
            if len(val) > 0:
                print(f", 第一个元素类型: {type(val[0]).__name__}", end="")
                if hasattr(val[0], 'numpy_view'):
                    arr = val[0].numpy_view()
                    print(f", shape={arr.shape}, dtype={arr.dtype}, max={arr.max():.3f}", end="")
            print()
        else:
            print(val)
    except Exception as e:
        print(f"{attr}: 读取失败 {e}")