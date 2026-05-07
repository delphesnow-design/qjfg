#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于真实项目代码的实验测试脚本
所有截图和数据保存到 experiment_results/ 目录
"""

# ─── 导入标准库 ───
import sys          # sys模块提供对Python解释器的访问，sys.path用于管理模块搜索路径
import os           # os模块提供操作系统接口，os.path用于路径操作，os.makedirs创建目录
import time         # time模块提供时间相关函数，time.time()返回当前Unix时间戳（浮点秒）
import csv          # csv模块读写逗号分隔值文件，用于保存实验数据表格

# ─── 导入第三方库 ───
import cv2          # OpenCV图像处理库，提供摄像头操作、图像显示、形态学处理等功能
import numpy as np  # NumPy数值计算库，提供多维数组ndarray和数学运算，as np是别名约定
import matplotlib.pyplot as plt
# matplotlib是Python绘图库，pyplot子模块提供类似MATLAB的绘图API，as plt是别名约定
import psutil       # psutil（process and system utilities）库，用于获取CPU/内存占用率
# 安装命令：pip install psutil

# ─── 项目路径配置 ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# sys.path是Python搜索模块的目录列表
# insert(0, ...)把项目根目录插入列表第0位（最高优先级），确保import能找到algorithms包
# os.path.abspath(__file__)：__file__是当前脚本的相对路径，abspath转为绝对路径
# os.path.dirname(...)：取上一级目录，即项目根目录

# ─── DLL预加载（必须在PyQt5之前，与main.py保持完全一致）───
try:
    import torch        # PyTorch深度学习框架，MODNet依赖
except ImportError:     # 如果torch未安装则跳过，不报错退出
    pass                # pass是空语句，什么都不做，用于语法要求有语句但逻辑为空的情况

try:
    import mediapipe    # Google MediaPipe框架
except ImportError:
    pass

try:
    import onnxruntime  # ONNX Runtime推理引擎，RVM依赖
except ImportError:
    pass


# ─── 全局结果保存目录配置 ───
RESULTS_DIR = "experiment_results"
# 字符串字面量赋值给变量，RESULTS_DIR是目录名，全大写是Python常量命名惯例（PEP8规范）

def ensure_results_dir(subdir: str = "") -> str:
    """
    创建实验结果保存目录，返回最终路径
    subdir: 子目录名，用于按算法分类存放
    -> str: 函数返回值类型注解，说明返回字符串
    """
    if subdir:
        # if subdir: 等价于 if subdir != ""，非空字符串在布尔上下文中为True
        path = os.path.join(RESULTS_DIR, subdir)
        # os.path.join用操作系统正确的分隔符拼接路径
        # Windows上是反斜杠\，Linux/Mac上是正斜杠/，join自动处理
    else:
        path = RESULTS_DIR
        # 没有子目录时直接用根结果目录

    os.makedirs(path, exist_ok=True)
    # os.makedirs递归创建多级目录（mkdir只能创建单层）
    # exist_ok=True：如果目录已存在不抛出FileExistsError，直接忽略
    # 相当于Linux的 mkdir -p 命令

    return path
    # return语句返回函数结果给调用者


def save_frame_pair(frame_orig: np.ndarray,
                    frame_result: np.ndarray,
                    algo_name: str,
                    frame_idx: int,
                    save_dir: str) -> None:
    """
    同时保存原始帧和处理结果帧，用于论文对比图
    frame_orig: np.ndarray  ← 类型注解，说明参数是NumPy数组
    frame_result: np.ndarray
    algo_name: str          ← 算法名称字符串
    frame_idx: int          ← 帧序号整数
    save_dir: str           ← 保存目录字符串
    -> None                 ← 返回值为None（无返回值）
    """
    orig_path   = os.path.join(save_dir, f"orig_{algo_name}_{frame_idx:04d}.png")
    result_path = os.path.join(save_dir, f"result_{algo_name}_{frame_idx:04d}.png")
    # f-string（格式化字符串字面量）：f"..."中{}内的表达式会被求值替换
    # {frame_idx:04d}：将frame_idx格式化为4位十进制整数，不足4位前补0
    # 例如frame_idx=5 → "0005"，确保文件名按字典序排列与时间顺序一致

    cv2.imwrite(orig_path, frame_orig)
    # cv2.imwrite(路径, 图像数组)：将NumPy数组保存为图片文件
    # OpenCV自动根据文件扩展名选择编码格式，.png是无损压缩，适合论文截图

    cv2.imwrite(result_path, frame_result)


def get_cpu_usage() -> float:
    """获取当前CPU占用率（百分比）"""
    return psutil.cpu_percent(interval=0.1)
    # psutil.cpu_percent(interval=N)：等待N秒后采样，返回0.0-100.0的浮点数
    # interval=0.1表示等待0.1秒，避免瞬时采样误差过大


def get_memory_usage() -> float:
    """获取当前内存占用（MB）"""
    process = psutil.Process(os.getpid())
    # psutil.Process(pid)：获取指定进程ID的Process对象
    # os.getpid()：返回当前进程的PID（Process ID，进程标识符）
    return process.memory_info().rss / 1024 / 1024
    # memory_info().rss：RSS（Resident Set Size，常驻内存大小），单位是字节（Bytes）
    # 除以1024得KB，再除以1024得MB
    # / 是Python的真除法，结果为浮点数（Python3特性，Python2中/是整除）


def run_experiment(algorithm_id: int,
                   duration_sec: int = 60,
                   auto_screenshot_sec: int = 8) -> dict:
    """
    对指定算法进行完整实验测试
    返回包含所有实验数据的字典
    """
    # ─── 算法名称映射字典 ───
    algo_names = {
        0: "MODNet",   1: "MediaPipe", 2: "RVM",
        3: "MOG2",     4: "KNN",       5: "GrabCut",
        6: "LOBSTER",  7: "SuBSENSE"
    }
    # dict（字典）是Python内置数据结构，键值对映射，{}语法创建
    # 整数键映射到字符串值，通过algo_names[0]取值得"MODNet"

    name = algo_names.get(algorithm_id, "Unknown")
    # dict.get(key, default)：取key对应的值，key不存在时返回default而非抛KeyError
    # 比algo_names[algorithm_id]更安全

    # ─── 创建本算法专属保存目录 ───
    save_dir = ensure_results_dir(name)
    # 每个算法在experiment_results/下有独立子目录
    # 例如：experiment_results/MediaPipe/，experiment_results/MODNet/

    print(f"\n{'='*50}")
    # f-string中 {'='*50} 是字符串重复运算：'='乘以50产生50个等号
    # 用于在终端输出分隔线，提升可读性
    print(f"开始测试: {name}  保存目录: {save_dir}")
    print(f"{'='*50}")

    # ─── 初始化算法（通过工厂类，与项目完全一致）───
    from algorithms.factory import BackgroundChangerFactory
    # from...import语句：从指定模块导入特定名称
    # 延迟导入（在函数内而非文件顶部）：确保DLL预加载已完成后再导入

    factory = BackgroundChangerFactory(algorithm_id=algorithm_id)
    # 实例化工厂类，algorithm_id=algorithm_id是关键字参数传递方式
    # 关键字参数：显式指定参数名，顺序可以不固定，代码可读性更高

    factory.load_backgrounds("backgrounds")
    # 调用实例方法，加载backgrounds/目录下的背景图片
    # 注意：backgrounds/目录必须存在且不为空，否则process_frame直接返回原帧

    # ─── 打开摄像头 ───
    cap = cv2.VideoCapture(0)
    # cv2.VideoCapture(index)：打开视频捕获设备
    # index=0：系统默认摄像头（第一个摄像头）
    # index=1：第二个摄像头（如外接USB摄像头）
    # 返回VideoCapture对象，用于后续读取帧

    if not cap.isOpened():
        # cap.isOpened()：返回bool，True表示摄像头成功打开
        # not取反：如果未成功打开则进入if块
        print("错误：摄像头无法打开")
        return {}
        # 返回空字典，调用方通过检查返回值是否为空来判断是否成功

    # ─── 实验数据收集容器 ───
    fps_list      = []   # list（列表）：动态数组，[]创建空列表，append()追加元素
    cpu_list      = []   # 每帧的CPU占用率
    mem_list      = []   # 每帧的内存占用（MB）
    frame_count   = 0    # 计数器变量，整数，初始化为0
    screenshot_done = False  # bool标志位，False/True，控制是否已保存截图
    start_time    = time.time()
    # time.time()返回自1970-01-01 00:00:00 UTC以来的秒数（Unix时间戳）
    # 浮点数，精度可达微秒级

    print(f"测试进行中，共 {duration_sec} 秒。按 'q' 退出，按 's' 手动截图")

    # ─── 主循环：逐帧处理 ───
    while True:
        # while True: 无限循环，依靠内部break语句退出
        # Python的循环没有do-while语法，while True + break是惯用替代

        elapsed = time.time() - start_time
        # 当前经过时间 = 当前时间戳 - 开始时间戳，单位秒
        if elapsed >= duration_sec:
            # 超过设定时长则退出循环
            break
            # break：立即终止最近一层循环

        # ─── 读取一帧 ───
        t_frame_start = time.time()
        # 记录本帧处理开始时刻（包含cap.read的IO时间）

        ret, frame = cap.read()
        # cap.read()：从摄像头读取下一帧
        # 返回元组(bool, ndarray)，Python支持多返回值自动解包
        # ret：True表示成功读取，False表示读取失败（如摄像头断开）
        # frame：BGR格式的NumPy数组，shape为(height, width, 3)

        if not ret:
            # 读取失败时退出循环
            break

        # ─── 记录处理前的系统状态 ───
        cpu_before = get_cpu_usage()
        # 调用前面定义的函数，获取当前CPU占用率

        # ─── 核心：调用真实项目的处理函数 ───
        result = factory.process_frame(frame)
        # 这一行与GUI中的调用方式完全相同
        # factory.process_frame内部根据algorithm_id调用对应算法的process_frame
        # 输入：BGR格式的NumPy数组（H,W,3）
        # 输出：背景替换后的BGR格式NumPy数组（H,W,3）

        # ─── 计算本帧处理时间和帧率 ───
        t_frame_end = time.time()
        frame_time  = t_frame_end - t_frame_start
        # 单帧耗时（秒），浮点数

        fps = 1.0 / max(frame_time, 1e-6)
        # fps = 1 / 耗时（秒）
        # max(frame_time, 1e-6)：防止frame_time为0时发生除零错误
        # 1e-6是科学计数法：1×10^-6 = 0.000001秒，即1微秒
        # 这是Python浮点字面量的科学计数法写法

        fps_list.append(fps)
        # list.append(value)：在列表末尾追加元素，时间复杂度O(1)

        # ─── 记录CPU和内存 ───
        cpu_list.append(get_cpu_usage())
        mem_list.append(get_memory_usage())

        frame_count += 1
        # += 是增量赋值运算符：frame_count = frame_count + 1 的简写

        # ─── 自动截图：背景模型稳定后（elapsed > auto_screenshot_sec）───
        if elapsed > auto_screenshot_sec and not screenshot_done:
            # and 是逻辑与运算符：两个条件都为True时整体才为True
            # not screenshot_done：只截一次，避免重复保存
            save_frame_pair(frame, result, name, frame_count, save_dir)
            screenshot_done = True
            print(f"[{elapsed:.1f}s] 已自动保存对比截图到 {save_dir}/")
            # {elapsed:.1f}：格式化为保留1位小数的浮点数

        # ─── 在视频窗口叠加信息文字 ───
        display = result.copy()
        # ndarray.copy()：深拷贝，创建新数组
        # 不在result上直接修改，避免把文字叠加到保存的截图里

        info_lines = [
            f"算法: {name}",
            f"FPS: {fps:.1f}",
            f"CPU: {cpu_list[-1]:.1f}%",
            # cpu_list[-1]：Python负索引，-1表示列表最后一个元素
            f"内存: {mem_list[-1]:.1f}MB",
            f"时间: {elapsed:.0f}/{duration_sec}s",
            # {elapsed:.0f}：格式化为0位小数（即整数显示）
        ]

        for i, line in enumerate(info_lines):
            # enumerate(iterable)：同时获取索引和元素值
            # 返回(index, value)元组序列，配合for...in解包使用
            # i从0开始，line是对应的字符串

            cv2.putText(
                display,            # 目标图像（NumPy数组）
                line,               # 要绘制的文字字符串
                (10, 30 + i * 30),  # 文字起始坐标(x, y)，元组
                                    # x=10（距左边10像素），y随i增加（每行间隔30像素）
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型：Hershey简单字体
                0.7,                # 字体缩放因子（0.7倍原始大小）
                (0, 255, 0),        # 字体颜色BGR格式：(蓝,绿,红)=(0,255,0)=纯绿色
                2                   # 线条粗细（像素）
            )

        cv2.imshow(f"实验测试 - {name}", display)
        # cv2.imshow(窗口名, 图像)：在命名窗口中显示图像
        # 窗口名不同则创建多个独立窗口

        key = cv2.waitKey(1) & 0xFF
        # cv2.waitKey(delay_ms)：等待键盘输入，delay_ms毫秒后返回
        # delay_ms=1：等待1毫秒，接近无等待，保证视频流畅
        # 返回值是按键的ASCII码，未按键返回-1
        # & 0xFF：按位与运算，取低8位，消除高位平台差异（Windows和Linux返回值不同）

        if key == ord('q'):
            # ord('q')：返回字符'q'的ASCII码整数值（113）
            # 按q键退出
            break

        elif key == ord('s'):
            # elif（else if缩写）：前面条件不满足时检查此条件
            save_frame_pair(frame, result, name, frame_count, save_dir)
            print(f"[{elapsed:.1f}s] 手动截图已保存")

    # ─── 循环结束，释放资源 ───
    cap.release()
    # cap.release()：释放摄像头资源，关闭设备文件句柄
    # 不释放会导致摄像头指示灯持续亮起，其他程序无法使用摄像头

    cv2.destroyAllWindows()
    # 关闭所有由cv2.imshow创建的显示窗口

    # ─── 处理空数据边界情况 ───
    if not fps_list:
        # not fps_list：空列表在布尔上下文为False，not后变True
        # 即：如果一帧都没采集到，直接返回空字典
        return {}

    # ─── 统计计算 ───
    result_stats = {
        "algorithm":   name,
        "algorithm_id": algorithm_id,
        "avg_fps":     float(np.mean(fps_list)),
        # np.mean(array)：计算数组算术平均值
        # float()：将NumPy标量转为Python原生float，便于后续JSON序列化等操作
        "min_fps":     float(np.min(fps_list)),
        # np.min：数组最小值
        "max_fps":     float(np.max(fps_list)),
        "std_fps":     float(np.std(fps_list)),
        # np.std：标准差，反映帧率波动幅度，越小越稳定
        "avg_cpu":     float(np.mean(cpu_list)),
        "avg_mem_mb":  float(np.mean(mem_list)),
        "total_frames": frame_count,
        "fps_series":  fps_list,   # 保留完整序列，用于绘图
        "cpu_series":  cpu_list,
        "save_dir":    save_dir,
    }

    # ─── 打印汇总 ───
    print(f"\n[{name}] 测试完成:")
    print(f"  平均帧率:  {result_stats['avg_fps']:.2f} fps")
    # {:.2f}：格式化为保留2位小数的浮点数
    print(f"  最低帧率:  {result_stats['min_fps']:.2f} fps")
    print(f"  帧率标准差:{result_stats['std_fps']:.2f}（越小越稳定）")
    print(f"  平均CPU:   {result_stats['avg_cpu']:.1f}%")
    print(f"  平均内存:  {result_stats['avg_mem_mb']:.1f} MB")
    print(f"  总帧数:    {frame_count}")
    print(f"  截图保存至:{save_dir}/")

    # ─── 保存帧率CSV数据文件 ───
    csv_path = os.path.join(save_dir, f"fps_data_{name}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        # open(path, mode, ...)：打开文件，返回文件对象
        # 'w'：写模式，文件不存在则创建，存在则清空
        # newline=''：告诉csv模块自己处理换行符，避免Windows多余空行
        # encoding='utf-8'：指定文件编码

        writer = csv.writer(f)
        # csv.writer：创建CSV写入器对象

        writer.writerow(["帧序号", "帧率(fps)", "CPU占用(%)"])
        # writerow(list)：写入一行，列表元素间用逗号分隔

        for i, (fps_val, cpu_val) in enumerate(zip(fps_list, cpu_list)):
            # zip(a, b)：将两个列表"拉链"配对，返回(a[i],b[i])的迭代器
            # enumerate(zip(...))：同时获取索引
            # for i, (fps_val, cpu_val)：解包嵌套元组
            writer.writerow([i + 1, f"{fps_val:.2f}", f"{cpu_val:.1f}"])

    print(f"  CSV数据:   {csv_path}")
    return result_stats


def plot_all_results(all_results: list) -> None:
    """
    绘制所有算法的对比图表，保存到 experiment_results/ 根目录
    all_results: list  ← 包含多个result_stats字典的列表
    """
    if not all_results:
        # 边界检查：没有数据则直接返回
        print("没有实验数据，跳过绘图")
        return

    ensure_results_dir()
    # 确保根结果目录存在

    names   = [r["algorithm"] for r in all_results]
    # 列表推导式（List Comprehension）：[ 表达式 for 变量 in 可迭代对象 ]
    # 等价于：
    # names = []
    # for r in all_results:
    #     names.append(r["algorithm"])
    # 列表推导式更简洁，是Python惯用写法

    avg_fps = [r["avg_fps"] for r in all_results]
    min_fps = [r["min_fps"] for r in all_results]
    std_fps = [r["std_fps"] for r in all_results]
    avg_cpu = [r["avg_cpu"] for r in all_results]

    # ─── 创建2行2列的子图布局 ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # plt.subplots(nrows, ncols, figsize=(宽英寸, 高英寸))
    # 返回(Figure对象, Axes数组)
    # axes是2×2的NumPy数组，axes[0][0]是左上，axes[0][1]是右上，以此类推

    # 颜色列表，每个算法一个颜色
    colors = ['#E74C3C','#3498DB','#2ECC71','#F39C12',
              '#9B59B6','#1ABC9C','#E67E22','#95A5A6']
    # 十六进制颜色码，#RRGGBB格式

    plt.rcParams['font.family'] = 'SimHei'
    # rcParams：matplotlib全局配置字典
    # font.family设置字体族，SimHei（黑体）支持中文字符显示
    # 不设置中文会显示方块乱码

    # ─── 图1：平均帧率柱状图（左上）───
    ax1 = axes[0][0]
    # axes[0][0]：第0行第0列的子图，即左上角
    # 赋值给ax1是为了后续操作简洁

    bars = ax1.bar(names, avg_fps,
                   color=colors[:len(names)],  # 切片取前len(names)个颜色
                   alpha=0.85,                 # 透明度0-1，0.85略透明
                   edgecolor='black',          # 柱子边框颜色
                   linewidth=0.8)              # 边框线宽

    ax1.axhline(y=25, color='red', linestyle='--', linewidth=1.5,
                label='实时基准(25fps)')
    # axhline：在y=25处画水平参考线
    # linestyle='--'：虚线样式

    ax1.set_title("各算法平均帧率对比", fontsize=13)
    ax1.set_ylabel("帧率 (fps)")
    ax1.tick_params(axis='x', rotation=30)
    # tick_params：坐标轴刻度标签样式，rotation=30表示x轴标签旋转30度防重叠
    ax1.legend()
    # legend()：显示图例（label对应的图例项）

    for bar, val in zip(bars, avg_fps):
        # 在每个柱子顶部标注数值
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 # bar.get_x()：柱子左边x坐标；get_width()/2：柱子宽度一半
                 # 两者相加得柱子中心x坐标
                 bar.get_height() + 0.5,
                 # 柱子顶部y坐标 + 0.5像素偏移，避免文字覆盖柱子
                 f"{val:.1f}",       # 文字内容
                 ha='center',        # 水平对齐：居中
                 va='bottom',        # 垂直对齐：底部
                 fontsize=9)

    # ─── 图2：帧率标准差（右上，反映稳定性）───
    ax2 = axes[0][1]
    ax2.bar(names, std_fps,
            color=colors[:len(names)], alpha=0.85, edgecolor='black', linewidth=0.8)
    ax2.set_title("帧率标准差（越小越稳定）", fontsize=13)
    ax2.set_ylabel("标准差 (fps)")
    ax2.tick_params(axis='x', rotation=30)

    # ─── 图3：CPU占用柱状图（左下）───
    ax3 = axes[1][0]
    ax3.bar(names, avg_cpu,
            color=colors[:len(names)], alpha=0.85, edgecolor='black', linewidth=0.8)
    ax3.set_title("各算法CPU占用率对比", fontsize=13)
    ax3.set_ylabel("CPU占用率 (%)")
    ax3.tick_params(axis='x', rotation=30)

    # ─── 图4：帧率随时间变化折线图（右下）───
    ax4 = axes[1][1]
    for i, r in enumerate(all_results):
        series = r["fps_series"]
        # series是该算法所有帧的fps列表

        # 最多取300个点，太多点折线图会太密
        step = max(1, len(series) // 300)
        # //是整除运算符，结果为整数
        # max(1, ...)确保step至少为1

        sampled = series[::step]
        # 切片语法 [start:stop:step]，start和stop省略表示全部
        # series[::2]表示每隔2个取一个元素（下采样）

        ax4.plot(sampled, label=r["algorithm"],
                 color=colors[i], alpha=0.8, linewidth=1.2)
        # plot(y值列表)：绘制折线

    ax4.axhline(y=25, color='red', linestyle='--', linewidth=1.5)
    ax4.set_title("帧率随时间变化", fontsize=13)
    ax4.set_xlabel("帧序号（采样）")
    ax4.set_ylabel("帧率 (fps)")
    ax4.legend()

    plt.tight_layout()
    # tight_layout()：自动调整子图间距，避免标题/标签重叠

    save_path = os.path.join(RESULTS_DIR, "comparison_chart.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # savefig(路径, dpi=分辨率)：保存图表为图片文件
    # dpi=150：每英寸150点，论文图片建议≥150
    # bbox_inches='tight'：裁剪掉图表外的多余空白

    print(f"\n对比图表已保存: {save_path}")
    plt.show()
    # plt.show()：在屏幕上弹出交互式图表窗口


def save_summary_csv(all_results: list) -> None:
    """将所有算法汇总数据保存为CSV，方便填写论文表格"""
    if not all_results:
        return

    ensure_results_dir()
    csv_path = os.path.join(RESULTS_DIR, "experiment_summary.csv")

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        # encoding='utf-8-sig'：UTF-8带BOM签名
        # Excel打开UTF-8文件需要BOM才能正确识别中文，-sig自动添加

        writer = csv.writer(f)
        writer.writerow(["算法", "平均fps", "最低fps", "最高fps",
                          "帧率标准差", "平均CPU%", "平均内存MB"])
        # 表头行

        for r in all_results:
            writer.writerow([
                r["algorithm"],
                f"{r['avg_fps']:.2f}",
                f"{r['min_fps']:.2f}",
                f"{r['max_fps']:.2f}",
                f"{r['std_fps']:.2f}",
                f"{r['avg_cpu']:.1f}",
                f"{r['avg_mem_mb']:.1f}",
            ])

    print(f"汇总CSV已保存: {csv_path}（可直接用Excel打开）")


# ─── 主程序入口 ───
if __name__ == "__main__":
    # if __name__ == "__main__"：Python惯用语
    # __name__是内置变量：直接运行此文件时值为"__main__"
    # 被其他文件import时值为模块名
    # 此判断确保以下代码只在直接运行时执行，import时不执行

    print("实验结果将保存至:", os.path.abspath(RESULTS_DIR))
    # os.path.abspath：将相对路径转为绝对路径，显示完整保存位置

    # ========== 修改此处选择测试算法 ==========
    # 第一步对比（全部）：test_ids = [3, 4, 5, 6, 7, 0, 1, 2]
    # 第二步对比（仅深度学习）：
    test_ids = [0, 1, 2]   # MODNet=0, MediaPipe=1, RVM=2

    all_results = []
    # 初始化空列表，收集所有算法的实验结果

    for algo_id in test_ids:
        # for...in循环：依次遍历test_ids列表中的每个元素
        result = run_experiment(
            algorithm_id=algo_id,
            duration_sec=60,          # 每个算法测试60秒
            auto_screenshot_sec=8,    # 8秒后自动截图（背景模型稳定后）
        )

        if result:
            # 非空字典为True，空字典为False
            all_results.append(result)

        input("\n准备好后按回车键继续测试下一个算法...")
        # input()：等待用户键盘输入，返回输入的字符串
        # 这里不使用返回值，仅作暂停用途，让用户有时间查看当前结果

    # ─── 所有测试完成，汇总输出 ───
    if all_results:
        print("\n" + "=" * 60)
        print("全部实验完成，正在生成图表和汇总...")
        print("=" * 60)

        plot_all_results(all_results)
        save_summary_csv(all_results)

        print(f"\n所有文件已保存至: {os.path.abspath(RESULTS_DIR)}/")
        print("目录结构:")
        for r in all_results:
            print(f"  {RESULTS_DIR}/{r['algorithm']}/  ← 截图和帧率CSV")
        print(f"  {RESULTS_DIR}/comparison_chart.png  ← 对比图表（论文用）")
        print(f"  {RESULTS_DIR}/experiment_summary.csv ← 汇总表（Excel可打开）")