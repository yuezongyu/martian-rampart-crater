import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager as fm
myriad_path = r"/public2/yuezy/Apollo/Myriad_Pro/MyriadPro-Regular.otf"
# 1. 将字体文件添加到 Matplotlib 的字体管理器中
fm.fontManager.addfont(myriad_path)

# 2. 获取该字体文件在系统内部的名称（通常是 'Myriad Pro'）
prop = fm.FontProperties(fname=myriad_path)
font_name = prop.get_name()

# 3. 修改全局默认字体
plt.rcParams['font.family'] = font_name

# 如果你的坐标轴需要显示负号，通常还要加上这一行防止乱码
plt.rcParams['axes.unicode_minus'] = False

# =====================================
# 模块级常量
# =====================================
G_MARS = 3.721
MAX_RADIUS = 30
CELL_SIZE = 10
CELL_VOL = CELL_SIZE ** 3
CACHE_FILENAME = "ejecta_volume_rigorous.pkl"  # 修改文件名以区分旧算法

# =====================================
# 数学工具：向量化计算点到椭圆最短距离
# =====================================
def get_distance_to_ellipse_vectorized(x, y, a, b):
    """
    使用牛顿迭代法向量化计算点 (x, y) 到椭圆 (x/a)^2 + (y/b)^2 = 1 的最短几何距离。
    """
    # 1. 初始猜测角度 t
    # 使用 atan2(a*y, b*x) 得到非常接近投影点的椭圆参数角
    t = np.arctan2(a * y, b * x)
    
    # 2. 牛顿迭代求解 f'(t) = 0
    # 目标是最小化距离平方 D^2。
    
    a2 = a**2
    b2 = b**2
    c_val = b2 - a2
    
    # 迭代 5 次足以达到 float64 精度
    for _ in range(5):
        st = np.sin(t)
        ct = np.cos(t)
        
        # 一阶导数 f' (实际上是 d(D^2)/dt 的一半)
        f_prime = c_val * st * ct + a * x * st - b * y * ct
        
        # 二阶导数 f'' (用于牛顿法分母)
        f_double_prime = c_val * (ct**2 - st**2) + a * x * ct + b * y * st
        
        # 避免除以零
        f_double_prime = np.maximum(f_double_prime, 1e-10) 
        
        # 更新 t
        dt = f_prime / f_double_prime
        t = t - dt

    # 3. 计算椭圆上的投影点
    x_ellipse = a * np.cos(t)
    y_ellipse = b * np.sin(t)
    
    # 4. 计算欧几里得距离
    dist = np.hypot(x - x_ellipse, y - y_ellipse)
    return dist

# =====================================
# 核心处理函数
# =====================================
def process_file_vectorized(filepath, off, a_major, b_minor):
    """
    使用 pandas 和 numpy 批量处理单个文件。
    已集成严谨的距离计算。
    """
    try:
        # 读取数据 (engine='python' 容错性更好)
        df = pd.read_csv(filepath, sep=r'[,\s]+', header=None, 
                         usecols=[2, 5, 6, 7, 10, 11, 12], engine='python')
        
        # 强制转换为数值类型，错误转为 NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        # 丢弃 NaN 行
        df.dropna(inplace=True)
        # 转为 float64 numpy 数组
        data = df.values.astype(np.float64)

    except Exception:
        return np.empty((0, 3))

    if data.shape[0] == 0:
        return np.empty((0, 3))

    # 解包数据
    lbl = np.abs(data[:, 0].astype(int))
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    vx, vy, vz = data[:, 4], data[:, 5], data[:, 6]

    # 1. 初始空间筛选
    mask = (x >= -3000.0) & (x <= 2000.0)
    x_corrected = x - off

    # 2. 物理弹道计算
    D = vz**2 + 2 * G_MARS * z
    mask &= (D >= 0)
    D_safe = np.where(D >= 0, D, 0)
    
    t = (vz + np.sqrt(D_safe)) / G_MARS
    mask &= (t >= 0)

    x_land = x_corrected + vx * t
    y_land = y + vy * t

    # 3. 椭圆内部过滤 (Inside Check)
    inside = (x_land**2)/(a_major**2) + (y_land**2)/(b_minor**2) <= 1
    mask &= (~inside)
    
    # === 应用 Mask 减少牛顿迭代计算量 ===
    x_land = x_land[mask]
    y_land = y_land[mask]
    lbl = lbl[mask]
    
    if len(x_land) == 0:
        return np.empty((0, 3))

    # 4. 严谨距离计算 (Newton Method)
    # 计算点到椭圆边界的最短欧氏距离
    dist_from_rim = get_distance_to_ellipse_vectorized(x_land, y_land, a_major, b_minor)
    
    # 5. 最大半径过滤
    Rad = 0.5 * (a_major + b_minor)
    # 注意：这里的 dist_from_rim 是几何最短距离
    mask_dist = (dist_from_rim <= MAX_RADIUS * Rad)
    
    return np.column_stack((x_land[mask_dist], y_land[mask_dist], lbl[mask_dist]))


def process_folder(base_dir, offset, major, minor, subdir_name):
    """
    处理文件夹。强制在 base_dir/ejecta 目录下寻找缓存和原始数据。
    """
    info = {'folder': subdir_name}
    target_dir = os.path.join(base_dir, "ejecta")
    datafile = os.path.join(target_dir, CACHE_FILENAME)
    
    landed_data = [] 

    # --- 阶段 1: 获取数据 ---
    if os.path.exists(datafile) and os.path.getsize(datafile) > 0:
        try:
            with open(datafile, "rb") as f:
                raw_load = pickle.load(f)
                if len(raw_load) > 0:
                    landed_data = np.array(raw_load)
                else:
                    landed_data = np.empty((0, 3))
            info['source'] = 'Cache (.pkl)'
            info['path'] = datafile 
        except Exception as e:
            info['error'] = f"Cache corrupt: {e}"
            return {}, info
    else:
        # 读取原始文件
        if not os.path.exists(target_dir):
            info['error'] = f"Dir not found: {target_dir}"
            return {}, info
            
        info['source'] = 'Raw Data (.ejecta)'
        info['path'] = target_dir
        pattern = re.compile(r'^bm\.proc\d{1,4}\.(\d{4})\.ejecta$')
        
        try:
            files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) 
                     if pattern.match(f) and os.path.getsize(os.path.join(target_dir, f)) > 0]
        except Exception as e:
             info['error'] = f"Listdir fail: {e}"
             return {}, info

        info['total_files'] = len(files)
        
        if len(files) == 0:
            info['error'] = f"No .ejecta files found"
            return {}, info
        
        # 批量处理
        results_list = []
        for fp in files:
            res = process_file_vectorized(fp, offset, major, minor)
            if res.shape[0] > 0:
                results_list.append(res)
        
        if results_list:
            landed_data = np.vstack(results_list)
        else:
            landed_data = np.empty((0, 3))
            
        # 保存缓存
        try:
            with open(datafile, "wb") as f:
                pickle.dump(landed_data, f)
        except Exception:
            pass
    
    info['points_count'] = landed_data.shape[0]

    # --- 阶段 2: 方位角统计 (Volume Binning) ---
    if landed_data.shape[0] == 0:
        return {}, info

    x = landed_data[:, 0]
    y = landed_data[:, 1]
    labels = landed_data[:, 2].astype(int)

    # 计算角度
    angles_deg = np.degrees(np.arctan2(y, x)) % 360
    
    # Binning
    # 0度中心: 337.5 ~ 22.5
    bin_indices = np.floor((angles_deg + 22.5) / 45.0).astype(int)
    bin_indices[bin_indices == 8] = 0 

    unique_labels = np.unique(labels)
    sector_volume = {}
    
    for lbl in unique_labels:
        lbl_mask = (labels == lbl)
        lbl_bins = bin_indices[lbl_mask]
        
        if len(lbl_bins) > 0:
            # 统计每个 Bin 有多少个点
            counts = np.bincount(lbl_bins, minlength=9) 
            # 乘以单个点的体积 (CELL_VOL)
            sector_volume[lbl] = counts[:8] * CELL_VOL
        else:
             sector_volume[lbl] = np.zeros(8)

    return sector_volume, info


# ========== 主程序入口 ==========
if __name__ == "__main__":
    os.chdir("/public2/yuezy/Apollo")
    # 1. 配置参数
    folder = [
        ['Rampart-job2002','Rampart-job2003','Rampart-job2004','Rampart-job2005','Rampart-job2006','Rampart-job2007'],
        ['Rampart-job2008','Rampart-job2016','Rampart-job2075','Rampart-job2055','Rampart-job2056','Rampart-job2061'],
        ['Rampart-job2114','Rampart-job2116','Rampart-job2119','Rampart-job2120','Rampart-job2121','Rampart-job2145']
    ]

    offset = [
        [0,-148,-265,-407,-482,-657],
        [0,-184,-252,-344,-502,-602],
        [0,-160,-335,-525,-625,-630]
    ]

    major = [
        [1180,1180,1250,1200,1125,1180],
        [1269,1259,1327,1329,1327,1697],
        [1470,1459,1525,1425,1425,1100]
    ]

    minor = [
        [1180,1125,1200,1200,1100,970],
        [1269,1267,1260,1267,1270,1120],
        [1470,1467,1467,1497,1446,980]
    ]

    angle = [90, 75, 60, 45, 30, 15]
    
    color_map = {1: 'red', 2: 'blue', 3: 'green'}
    label_map = {1: 'proj', 2: 'ice', 3: 'basalt'}
    color_map1 = {1: 'red', 2: 'green'} 
    label_map1 = {1: 'proj', 2: 'basalt'}

    OUTPUT_DIR = "Figures"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 循环处理
    print("开始计算方位角体积分布 (使用严谨距离算法)...")
    fig, axes = plt.subplots(3, 6, figsize=(18.4, 10), sharey=True, sharex=True)
    angle_bins = np.arange(0, 360, 45)

    for i in range(3):
        for j in range(6):
            subdir = folder[i][j]
            fpath = os.path.join('.', subdir) 
            
            data, info = process_folder(fpath, offset[i][j], major[i][j], minor[i][j], subdir)
            
            # 状态打印
            source = info.get('source', 'Unknown')
            pts = info.get('points_count', 0)
            status_msg = info.get('error', 'OK')
            path_used = info.get('path', 'Unknown')
            
            if data:
                print(f"[{i+1},{j+1}] {subdir}: Src=[{source}] Pts=[{pts}] Path=[{path_used}]")
            else:
                print(f"[{i+1},{j+1}] {subdir}: ERROR -> {status_msg}")

            ax = axes[i, j]
            ax.set_ylim(top=1.5)
            
            if not data:
                ax.set_title(f"{subdir}\n(No data)", fontsize=8)
                ax.axis('off')
                continue

            bottom = np.zeros(8)
            
            if i == 0:
                curr_color_map = color_map1
                curr_label_map = label_map1
            else:
                curr_color_map = color_map
                curr_label_map = label_map

            for label in sorted(data.keys()):
                c = curr_color_map.get(label, 'gray')
                l = curr_label_map.get(label, str(label))
                
                vals = data[label]
                if len(vals) < 8:
                    vals = np.pad(vals, (0, 8-len(vals)))
                vals = vals/1e8
                # width=40 保证 bar 之间有微小空隙
                ax.bar(angle_bins, vals, bottom=bottom, width=40,
                       color=c, label=l, alpha=0.9)
                bottom += vals

            ax.set_xticks(angle_bins[::2]) 
            ax.grid(True, linestyle='--', alpha=0.5)
            
            if i == 0:
                ax.set_title(f"Angle={angle[j]}°", fontsize=15)
                
            
            if j == 0:
                ax.legend(fontsize=13, loc='upper left')
                ax.tick_params(axis='y', labelsize=15)
            
            if i == 2:
                ax.set_xlabel("Azimuth (°)", fontsize=15)
                ax.tick_params(axis='x', labelsize=15)
            
            if j == 0:
                ax.set_ylabel(r'Volume ($10^8$ m$^3$)', fontsize=15)

    plt.tight_layout()
    
    # 保存图片
    figname = os.path.join(OUTPUT_DIR, "Fig3_Azimuthal_Volume_Distribution_Rigorous")
    print(f"正在保存结果到: {figname} (.png/.pdf)")
    plt.savefig(f"{figname}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{figname}.pdf", format='pdf', bbox_inches='tight')
    
    plt.show()
    print("完成。")
