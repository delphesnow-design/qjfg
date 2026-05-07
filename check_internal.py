# check_internal.py  放项目根目录运行
import sys, os, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for lib in ['torch', 'mediapipe', 'onnxruntime']:
    try: __import__(lib)
    except ImportError: pass

from algorithms.factory import BackgroundChangerFactory
import inspect

# 检查MediaPipe内部changer
factory = BackgroundChangerFactory(algorithm_id=1)

print("=== MediaPipe changer 所有方法 ===")
changer = factory.changer
for name in dir(changer):
    if not name.startswith('__'):
        attr = getattr(changer, name)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
                print(f"  {name}{sig}")
            except:
                print(f"  {name}()")
        else:
            print(f"  [属性] {name} = {type(attr).__name__}")

# 尝试直接调用可能的掩模获取方法
print("\n=== 尝试获取掩模 ===")
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
test_img[100:380, 160:480] = (200, 150, 100)  # 模拟人像区域

for method_name in ['get_alpha', 'get_mask', 'segment',
                    'get_foreground_mask', 'process', '_get_mask']:
    if hasattr(changer, method_name):
        print(f"找到方法: {method_name}")
        try:
            result = getattr(changer, method_name)(test_img)
            print(f"  返回类型: {type(result)}, shape: {getattr(result, 'shape', 'N/A')}")
        except Exception as e:
            print(f"  调用失败: {e}")

print("\n=== 检查RVM changer ===")
factory2 = BackgroundChangerFactory(algorithm_id=2)
changer2 = factory2.changer
print("RVM changer方法:", [x for x in dir(changer2) if not x.startswith('__')])