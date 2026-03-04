import os
import re
import math
import pickle
import copy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
CACHE_FILENAME = "landed_points_rigorous.pkl" 

# =====================================
# 数学工具：向量化计算点到椭圆最短距离
# =====================================
def get_distance_to_ellipse_vectorized(x, y, a, b):
    """
    使用牛顿迭代法向量化计算点 (x, y) 到椭圆 (x/a)^2 + (y/b)^2 = 1 的最短几何距离。
    """
    # 1. 初始猜测角度 t
    t = np.arctan2(a * y, b * x)
    
    a2 = a**2
    b2 = b**2
    c_val = b2 - a2
    
    # 2. 牛顿迭代求解 f'(t) = 0
    for _ in range(5):
        st = np.sin(t)
        ct = np.cos(t)
        
        f = c_val * st * ct + a * x * st - b * y * ct
        f_prime = c_val * (ct**2 - st**2) + a * x * ct + b * y * st
        f_prime = np.maximum(f_prime, 1e-10)
        
        dt = f / f_prime
        t = t - dt

    # 3. 计算椭圆上的投影点坐标
    x_ellipse = a * np.cos(t)
    y_ellipse = b * np.sin(t)
    
    # 4. 计算欧几里得距离
    dist = np.hypot(x - x_ellipse, y - y_ellipse)
    return dist


# ========== 核心处理函数 ==========

def process_file_vectorized(filepath, off, a_major, b_minor):
    """
    使用 pandas 和 numpy 批量处理单个文件中的数据。
    返回符合条件的落点坐标数组 (N, 2)。
    """
    try:
        # 【修复 1】使用 engine='python' 和更通用的正则分隔符 r'[,\s]+' 以兼容逗号或空格
        df = pd.read_csv(filepath, sep=r'[,\s]+', header=None, 
                         usecols=[5, 6, 7, 10, 11, 12], engine='python')
        
        # 【修复 2】强制转换为数值类型 (errors='coerce' 会把非数字转为 NaN)
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # 【修复 3】丢弃包含 NaN 的行 (即丢弃了坏数据)
        df.dropna(inplace=True)
        
        # 【修复 4】确保数据类型为 float64
        data = df.values.astype(np.float64)
        
    except Exception:
        # 文件为空或读取失败
        return np.empty((0, 2))

    if data.shape[0] == 0:
        return np.empty((0, 2))

    # 解包数据列
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    vx, vy, vz = data[:, 3], data[:, 4], data[:, 5]

    # 1. 坐标筛选与位移修正 (此时 x 保证是 float，比较不会报错)
    mask = x >= -3000.0
    x_corrected = x - off 
    
    # 2. 物理弹道计算 (向量化)
    D = vz**2 + 2 * G_MARS * z
    mask &= (D >= 0)
    
    D_safe = np.where(D >= 0, D, 0)
    t = (vz + np.sqrt(D_safe)) / G_MARS
    mask &= (t >= 0)
    
    # 计算落点
    x_land = x_corrected + vx * t
    y_land = y + vy * t
    
    # 3. 椭圆内部过滤
    inside = (x_land**2)/(a_major**2) + (y_land**2)/(b_minor**2) <= 1
    mask &= (~inside) 
    
    # 应用初步掩码，减少后续重型计算
    x_land = x_land[mask]
    y_land = y_land[mask]
    
    if len(x_land) == 0:
        return np.empty((0, 2))

    # 4. 最大半径过滤 (使用严谨算法)
    dist_rigorous = get_distance_to_ellipse_vectorized(x_land, y_land, a_major, b_minor)
    
    Rad = (a_major + b_minor) * 0.5
    mask_dist = (dist_rigorous <= MAX_RADIUS * Rad)
    
    return np.column_stack((x_land[mask_dist], y_land[mask_dist]))


def process_folder(input_dir, off, a_major, b_minor):
    """
    处理单个文件夹：读取/生成数据并计算网格厚度。
    """
    info = {'folder': input_dir}
    Rad = (a_major + b_minor) * 0.5
    
    datafile = os.path.join(input_dir, CACHE_FILENAME)
    landed_points = []
    
    # --- 阶段 1: 获取落点数据 ---
    if os.path.exists(datafile) and os.path.getsize(datafile) > 0:
        try:
            with open(datafile, "rb") as f:
                content = pickle.load(f)
                landed_points = np.array(content)
            info['source'] = 'Cache (.pkl)'
            info['path'] = datafile
        except Exception as e:
            info['error'] = f"Cache corrupt: {e}"
            return (None, None, None, None, None, info)
    else:
        data_subdir = os.path.join(input_dir, "ejecta")
        if not os.path.exists(data_subdir):
            info['error'] = f"subdir not found: {data_subdir}"
            return (None, None, None, None, None, info)
        
        info['source'] = 'Raw Data (.ejecta)'
        info['path'] = data_subdir
        
        pattern = re.compile(r'^bm\.proc\d{1,4}\.(\d{4})\.ejecta$')
        files_to_process = []
        
        try:
            for fn in os.listdir(data_subdir):
                if pattern.match(fn):
                    path = os.path.join(data_subdir, fn)
                    if os.path.getsize(path) > 0:
                        files_to_process.append(path)
        except Exception as e:
             info['error'] = f"listdir error: {e}"
             return (None, None, None, None, None, info)
        
        info['total_files'] = len(files_to_process)
        
        all_points_list = []
        for fp in files_to_process:
            pts = process_file_vectorized(fp, off, a_major, b_minor)
            if pts.shape[0] > 0:
                all_points_list.append(pts)
        
        if all_points_list:
            landed_points = np.vstack(all_points_list)
        else:
            landed_points = np.empty((0, 2))

        try:
            with open(datafile, "wb") as f:
                pickle.dump(landed_points, f)
        except Exception:
            pass

    # --- 阶段 2: 统计网格厚度 ---
    if len(landed_points) == 0:
        info['error'] = "no landed points"
        return (None, None, None, None, None, info)

    points = np.array(landed_points)
    x_pts, y_pts = points[:, 0], points[:, 1]

    grid_size = 300.0
    cube_volume = 10.0 ** 3
    cell_area = grid_size ** 2

    # 确定网格范围
    x_min, x_max = np.min(x_pts), np.max(x_pts)
    y_min, y_max = np.min(y_pts), np.max(y_pts)
    
    x0 = np.floor(x_min / grid_size) * grid_size
    x1 = np.ceil(x_max / grid_size) * grid_size
    y0 = np.floor(y_min / grid_size) * grid_size
    y1 = np.ceil(y_max / grid_size) * grid_size
    
    bins_x = np.arange(x0, x1 + grid_size, grid_size)
    bins_y = np.arange(y0, y1 + grid_size, grid_size)
    
    H, xedges, yedges = np.histogram2d(x_pts, y_pts, bins=[bins_x, bins_y])
    
    thickness = (H.T * cube_volume) / cell_area
    X, Y = np.meshgrid(xedges, yedges)
    
    thickness_masked = np.where(thickness > 0, thickness, np.nan)
    
    valid_t = thickness_masked[np.isfinite(thickness_masked)]
    if len(valid_t) == 0:
        info['error'] = "no positive thickness"
        return (None, None, None, None, None, info)
        
    vmin, vmax = np.min(valid_t), np.max(valid_t)
    
    return (X / Rad, Y / Rad, thickness_masked, vmin, vmax, info)


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
    row_labels = ["Pure basalt", "Ice on surface", "Ice in middle"]
    
    OUTPUT_DIR = "Figures"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("开始处理数据 (Rigorous Distance Calculation)...")
    global_vmin, global_vmax = np.inf, -np.inf
    results = [[None]*6 for _ in range(3)]

    for i in range(3):
        for j in range(6):
            subdir = folder[i][j]
            res = process_folder(subdir, offset[i][j], major[i][j], minor[i][j])
            results[i][j] = res
            
            if res[3] is not None:
                global_vmin = min(global_vmin, res[3])
                global_vmax = max(global_vmax, res[4])
            
            info_dict = res[5]
            source_info = info_dict.get('source', 'Unknown')
            path_info = info_dict.get('path', '')
            status = "OK" if res[3] is not None else info_dict.get('error', 'Error')
            
            print(f"[{i+1},{j+1}] {subdir}: Source=[{source_info}] Status=[{status}] Path=[{path_info}]")

    if global_vmin == np.inf:
        print("Error: No valid data found in any folder.")
        exit()

    print(f"数据处理完成。全局 Vmin: {global_vmin:.4f}, Vmax: {global_vmax:.4f}")

    # 3. 绘图：厚度分布
    fig, axes = plt.subplots(3, 6, figsize=(18.4, 10))
    plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.07, wspace=0.25, hspace=0.3)

    try:
        cmap = copy.copy(mpl.colormaps['coolwarm'])
    except AttributeError:
        cmap = copy.copy(plt.cm.get_cmap("coolwarm"))
    cmap.set_bad('white')
    norm = LogNorm(vmin=global_vmin, vmax=global_vmax)

    for i in range(3):
        for j in range(6):
            ax = axes[i][j]
            res = results[i][j]
            X, Y, T, _, _, _ = res
            
            if X is None:
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                ax.set_axis_off()
                continue
            
            mesh = ax.pcolormesh(X, Y, T, cmap=cmap, norm=norm, shading='flat', rasterized=True)
            ax.set_aspect('equal')
            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)
            
            if i == 0:
                ax.set_title(f"Angle={angle[j]}°", fontsize=15)
            
            # if j == 0:
            #     ax.set_ylabel(row_labels[i], fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(r"$y/R_t$", fontsize=15)
                ax.tick_params(axis='y', labelsize=15)
                
            if i == 2:
                ax.set_xlabel(r"$x/R_t$", fontsize=15)
                ax.tick_params(axis='x', labelsize=15)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('Thickness (m)', fontsize=15)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    figname_map = os.path.join(OUTPUT_DIR, "Fig2_Ejecta_Thickness_Rigorous")
    print(f"正在保存分布图到: {figname_map} (.png/.pdf)")
    plt.savefig(f"{figname_map}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{figname_map}.pdf", format='pdf', bbox_inches='tight')

    # 4. 绘图：Offset 折线图
    fig2 = plt.figure(figsize=(8, 6))
    markers = ['o', 's', '^']

    for i, label in enumerate(row_labels):
        plt.plot(angle, offset[i], marker=markers[i], linewidth=2, markersize=8, label=label)

    plt.xlabel("Impact Angle (°)", fontsize=12)
    plt.ylabel("Offset (m)", fontsize=12)
    plt.title("Offset vs Impact Angle", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()

    figname_line = os.path.join(OUTPUT_DIR, "Offset_Analysis")
    print(f"正在保存折线图到: {figname_line} (.png/.pdf)")
    plt.savefig(f"{figname_line}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{figname_line}.pdf", format='pdf', bbox_inches='tight')

    plt.show()
    print("所有任务完成。")