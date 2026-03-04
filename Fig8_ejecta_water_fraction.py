import os
import re
import math
import pickle
import copy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm, ListedColormap

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

# ========== 模块级常量 (物理/几何参数) ==========
G_MARS = 3.721
MAX_RADIUS = 30 
CACHE_FILENAME = "landed_points_pressure_gpa_id_neg3.pkl" 

# ========== 熔融计算参数 (可在此修改) ==========
PRESSURE_CRITICAL = 3.0   # Pc (GPa): 完全熔融压强
PRESSURE_INCIPIENT = 1.0  # Pi (GPa): 开始熔融压强

# =====================================
# 数学工具：向量化计算点到椭圆最短距离
# =====================================
def get_distance_to_ellipse_vectorized(x, y, a, b):
    t = np.arctan2(a * y, b * x)
    a2 = a**2
    b2 = b**2
    c_val = b2 - a2
    for _ in range(5):
        st = np.sin(t)
        ct = np.cos(t)
        f = c_val * st * ct + a * x * st - b * y * ct
        f_prime = c_val * (ct**2 - st**2) + a * x * ct + b * y * st
        f_prime = np.maximum(f_prime, 1e-10)
        dt = f / f_prime
        t = t - dt
    x_ellipse = a * np.cos(t)
    y_ellipse = b * np.sin(t)
    dist = np.hypot(x - x_ellipse, y - y_ellipse)
    return dist

# ========== 核心处理函数 (读取原始数据) ==========
def process_file_vectorized(filepath, off, a_major, b_minor):
    try:
        # col 2 (ID), 5(x), 6(y), 7(z), 8(P), 9(T), 10,11,12(v)
        df = pd.read_csv(filepath, sep=r'[,\s]+', header=None, 
                         usecols=[2, 5, 6, 7, 8, 9, 10, 11, 12], engine='python')
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)
        data = df.values.astype(np.float64)
    except Exception:
        return np.empty((0, 4))

    if data.shape[0] == 0:
        return np.empty((0, 4))

    mat_id = data[:, 0]
    id_mask = (mat_id == -2)
    
    if not np.any(id_mask):
        return np.empty((0, 4))
        
    data = data[id_mask]
    
    # 重新解包
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    p_val   = data[:, 4] / 1.0e9 # GPa
    t_val   = data[:, 5]
    vx, vy, vz = data[:, 6], data[:, 7], data[:, 8]

    # 1. 坐标筛选与位移修正
    mask = x >= -3000.0
    x_corrected = x - off 
    
    # 2. 物理弹道计算
    D = vz**2 + 2 * G_MARS * z
    mask &= (D >= 0)
    
    D_safe = np.where(D >= 0, D, 0)
    t = (vz + np.sqrt(D_safe)) / G_MARS
    mask &= (t >= 0)
    
    x_land = x_corrected + vx * t
    y_land = y + vy * t
    
    # 3. 椭圆内部过滤
    inside = (x_land**2)/(a_major**2) + (y_land**2)/(b_minor**2) <= 1
    mask &= (~inside) 
    
    x_land = x_land[mask]
    y_land = y_land[mask]
    p_val  = p_val[mask]
    t_val  = t_val[mask]
    
    if len(x_land) == 0:
        return np.empty((0, 4))

    dist_rigorous = get_distance_to_ellipse_vectorized(x_land, y_land, a_major, b_minor)
    Rad = (a_major + b_minor) * 0.5
    mask_dist = (dist_rigorous <= MAX_RADIUS * Rad)
    
    return np.column_stack((x_land[mask_dist], y_land[mask_dist], p_val[mask_dist], t_val[mask_dist]))

def process_folder(input_dir, off, a_major, b_minor):
    info = {'folder': input_dir}
    Rad = (a_major + b_minor) * 0.5
    
    datafile = os.path.join(input_dir, CACHE_FILENAME)
    target_dir = os.path.join(input_dir, "ejecta")
    landed_data = [] 
    
    # 读取缓存 (缓存里存的还是原始数据：x, y, P, T)
    if os.path.exists(datafile) and os.path.getsize(datafile) > 0:
        try:
            with open(datafile, "rb") as f:
                raw_load = pickle.load(f)
                landed_data = np.array(raw_load) if len(raw_load) > 0 else np.empty((0, 4))
            info['source'] = 'Cache (.pkl)'
            info['path'] = datafile
        except Exception as e:
            info['error'] = f"Cache corrupt: {e}"
            landed_data = []
            
    if len(landed_data) == 0:
        if not os.path.exists(target_dir):
            info['error'] = f"Dir not found: {target_dir}"
            return (None, None, None, None, None, None, info)
            
        info['source'] = 'Raw Data (.ejecta)'
        info['path'] = target_dir
        
        pattern = re.compile(r'^bm\.proc\d{1,4}\.(\d{4})\.ejecta$')
        files_to_process = []
        try:
            for fn in os.listdir(target_dir):
                if pattern.match(fn):
                    path = os.path.join(target_dir, fn)
                    if os.path.getsize(path) > 0:
                        files_to_process.append(path)
        except Exception as e:
             info['error'] = f"listdir error: {e}"
             return (None, None, None, None, None, None, info)
        
        info['total_files'] = len(files_to_process)
        all_points_list = []
        for fp in files_to_process:
            pts = process_file_vectorized(fp, off, a_major, b_minor)
            if pts.shape[0] > 0:
                all_points_list.append(pts)
        if all_points_list:
            landed_data = np.vstack(all_points_list)
        else:
            landed_data = np.empty((0, 4))
        try:
            with open(datafile, "wb") as f:
                pickle.dump(landed_data, f)
        except Exception:
            pass

    if len(landed_data) == 0:
        info['error'] = "no landed points"
        return (None, None, None, None, None, None, info)

    points = np.array(landed_data)
    X = points[:, 0] / Rad
    Y = points[:, 1] / Rad
    p_pts = points[:, 2] # 这是压强 P
    t_pts = points[:, 3] 
    
    # ========== 熔融计算逻辑 (修改部分) ==========
    # 公式: Fraction = (P - Pi) / (Pc - Pi)
    # 避免除以零（虽然参数里不太可能相等，但加个保险）
    denom = PRESSURE_CRITICAL - PRESSURE_INCIPIENT
    if denom == 0: denom = 1e-10

    melt_fraction = (p_pts - PRESSURE_INCIPIENT) / denom
    
    # 边界条件:
    # 1. P < Pi -> Fraction < 0 -> clip to 0
    # 2. P > Pc -> Fraction > 1 -> clip to 1
    melt_fraction = np.clip(melt_fraction, 0.0, 1.0)

    # 返回值中的第三项从压强改为了熔融分数
    # min/max 直接设为 0 和 1 即可，因为我们要画的是归一化比例
    return (X, Y, melt_fraction, t_pts, 0.0, 1.0, info)


# ========== 主程序入口 ==========
if __name__ == "__main__":
    os.chdir("/public2/yuezy/Apollo")
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
    
    OUTPUT_DIR = "Figures"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"开始处理数据 (Melt Fraction Calculation)...")
    print(f"Parameters: Pi={PRESSURE_INCIPIENT} GPa, Pc={PRESSURE_CRITICAL} GPa")

    # results 保存为 3x6 结构
    results = [[None]*6 for _ in range(3)]

    # 读取数据：只读 i=1 和 i=2
    for i in range(1, 3): 
        for j in range(6):
            subdir = folder[i][j]
            # 注意：process_folder 现在返回的是 Melt Fraction 而不是 Pressure
            res = process_folder(subdir, offset[i][j], major[i][j], minor[i][j])
            results[i][j] = res
            
            info_dict = res[6]
            source_info = info_dict.get('source', 'Unknown')
            status = "OK" if res[0] is not None else info_dict.get('error', 'Error')
            print(f"[{i+1},{j+1}] {subdir}: Source=[{source_info}] Status=[{status}]")

    # --- 颜色映射设置 (针对熔融分数 0.0 ~ 1.0) ---
    # 使用 Jet 或者其他连续色谱
    try:
        cmap = mpl.colormaps['jet']
    except AttributeError:
        cmap = plt.cm.get_cmap("jet")
    
    # 因为熔融分数是 0 到 1 的连续值，使用 Normalize 效果更好
    norm = Normalize(vmin=0.0, vmax=1.0)

    # --- 绘图：2行6列 ---
    fig, axes = plt.subplots(2, 6, figsize=(18.4, 7), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.08, wspace=0.1, hspace=0.2)

    for i_plot in range(2):
        i_data = i_plot + 1 
        
        for j in range(6):
            ax = axes[i_plot][j]
            res = results[i_data][j]
            
            if res is None or res[0] is None or len(res[0]) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_axis_off()
                continue
            
            # 这里解包出的 P 其实是 Melt Fraction
            X, Y, MeltFrac, T, _, _, _ = res

            hb = ax.hexbin(X, Y, C=MeltFrac, 
                           gridsize=60, 
                           cmap=cmap, 
                           norm=norm, 
                           reduce_C_function=np.mean, # 计算每个网格内的平均熔融分数
                           extent=(-5, 5, -5, 5),
                           mincnt=1,
                           linewidths=0.1, 
                           edgecolors='none'
                           )
            
            ax.set_aspect('equal')
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)

            if i_plot == 0:
                ax.set_title(f"Angle={angle[j]}°", fontsize=15)
            
            if j == 0:
                ax.set_ylabel(r"$Y/R_t$", fontsize=15)
                ax.set_yticks([-4, -2, 0, 2, 4])
                ax.tick_params(axis='y', labelsize=15)
            
            if i_plot == 1:
                ax.set_xlabel(r"$X/R_t$", fontsize=15)
                ax.set_xticks([-4, -2, 0, 2, 4])
                ax.tick_params(axis='x', labelsize=15)

    # --- Colorbar ---
    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
        cax=cbar_ax,
        ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0] # 设置刻度
    )
    # cbar.set_label(f'Ice Melt Fraction\n(Pi={PRESSURE_INCIPIENT} GPa, Pc={PRESSURE_CRITICAL} GPa)', fontsize=14)
    cbar.set_label('Ice Melt Fraction', fontsize=15)
    cbar.ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    cbar.ax.tick_params(labelsize=15)

    figname_map = os.path.join(OUTPUT_DIR, "Fig8_Melt_Fraction")
    print(f"正在保存分布图到: {figname_map} (.png/.pdf)")
    plt.savefig(f"{figname_map}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{figname_map}.pdf", format='pdf', bbox_inches='tight')
    
    plt.show()