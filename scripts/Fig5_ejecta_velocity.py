import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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
# 全局常量配置
# =====================================
G_MARS = 3.721
MAX_RADIUS_FACTOR = 30   # 对应 total_width (30 * Rad)
BIN_WIDTH_FACTOR = 0.2   # 对应 first_width (0.2 * Rad)
CACHE_FILENAME = "ejecta_vr_rigorous_filtered.pkl" # 修改缓存名以区分新逻辑
OUTPUT_DIR = "Figures"

# === 新增：最小统计阈值 ===
MIN_PARTICLE_THRESHOLD = 10

# =====================================
# 数学工具：向量化计算点到椭圆最短距离
# =====================================
def get_distance_to_ellipse_vectorized(x, y, a, b):
    """
    使用牛顿迭代法向量化计算点 (x, y) 到椭圆 (x/a)^2 + (y/b)^2 = 1 的最短几何距离。
    """
    t = np.arctan2(a * y, b * x)
    
    a2 = a**2
    b2 = b**2
    c_val = b2 - a2 
    
    for _ in range(5):
        st = np.sin(t)
        ct = np.cos(t)
        
        f_prime = c_val * st * ct + a * x * st - b * y * ct
        f_double_prime = c_val * (ct**2 - st**2) + a * x * ct + b * y * st
        f_double_prime = np.maximum(f_double_prime, 1e-10) 
        dt = f_prime / f_double_prime
        t = t - dt

    x_ellipse = a * np.cos(t)
    y_ellipse = b * np.sin(t)
    dist = np.hypot(x - x_ellipse, y - y_ellipse)
    return dist

# =====================================
# 核心处理函数
# =====================================
def process_file_vectorized(filepath, off, a_major, b_minor):
    """
    处理单个文件，返回: (dist_from_rim, azimuth, vr)
    """
    try:
        df = pd.read_csv(filepath, sep=r'[,\s]+', header=None, 
                         usecols=[5, 6, 7, 10, 11, 12], engine='python')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        data = df.values.astype(np.float64)
    except Exception:
        return None

    if data.shape[0] == 0:
        return None

    x0, y0, z0 = data[:, 0], data[:, 1], data[:, 2]
    vx, vy, vz = data[:, 3], data[:, 4], data[:, 5]

    # --- 1. 基础筛选 ---
    mask = (x0 >= -3000.0)
    x_corrected = x0 - off

    # --- 2. 物理弹道 ---
    D = vz**2 + 2 * G_MARS * z0
    mask &= (D >= 0)
    D_safe = np.where(D >= 0, D, 0)
    t = (vz + np.sqrt(D_safe)) / G_MARS
    mask &= (t >= 0)

    x_land = x_corrected + vx * t
    y_land = y0 + vy * t
    
    # --- 3. 几何计算 ---
    inside = (x_land**2)/(a_major**2) + (y_land**2)/(b_minor**2) <= 1
    mask &= (~inside)
    
    x_land = x_land[mask]
    y_land = y_land[mask]
    vx = vx[mask]
    vy = vy[mask]
    
    if len(x_land) == 0:
        return None

    # 精确计算距离
    dist_from_rim = get_distance_to_ellipse_vectorized(x_land, y_land, a_major, b_minor)
    
    # 远场筛选
    Rad = 0.5 * (a_major + b_minor)
    total_width = MAX_RADIUS_FACTOR * Rad
    valid_dist = (dist_from_rim <= total_width) & (dist_from_rim >= 0)
    
    x_final = x_land[valid_dist]
    y_final = y_land[valid_dist]
    vx_final = vx[valid_dist]
    vy_final = vy[valid_dist]
    d_final = dist_from_rim[valid_dist]
    
    # 径向速度 Vr
    r_center = np.hypot(x_final, y_final)
    # 避免除以0
    r_center = np.maximum(r_center, 1e-6)
    vr = (x_final * vx_final + y_final * vy_final) / r_center
    
    # 方位角
    azimuth = np.degrees(np.arctan2(y_final, x_final)) % 360

    return np.column_stack((d_final, azimuth, vr))


def process_folder(base_dir, offset, major, minor, subdir_name):
    """
    处理文件夹，聚合、缓存、分箱
    """
    info = {'folder': subdir_name}
    Rad = 0.5 * (major + minor)
    
    target_dir = os.path.join(base_dir, "ejecta")
    datafile = os.path.join(target_dir, CACHE_FILENAME)
    
    combined_data = None

    # --- 读取缓存 ---
    if os.path.exists(datafile) and os.path.getsize(datafile) > 0:
        try:
            with open(datafile, "rb") as f:
                combined_data = pickle.load(f)
            info['source'] = 'Cache (.pkl)'
            info['path'] = datafile
        except:
            pass

    # --- 重新计算 ---
    if combined_data is None:
        if not os.path.exists(target_dir):
            return None, Rad, info
        
        info['source'] = 'Raw Data (.ejecta)'
        info['path'] = target_dir
        
        pattern = re.compile(r'^bm\.proc\d{1,4}\.(\d{4})\.ejecta$')
        files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) 
                 if pattern.match(f) and os.path.getsize(os.path.join(target_dir, f)) > 0]
        
        results_list = []
        for fp in files:
            res = process_file_vectorized(fp, offset, major, minor)
            if res is not None and res.shape[0] > 0:
                results_list.append(res)
        
        if results_list:
            combined_data = np.vstack(results_list)
            try:
                with open(datafile, "wb") as f:
                    pickle.dump(combined_data, f)
            except:
                pass
        else:
            return None, Rad, info

    info['points'] = combined_data.shape[0] if combined_data is not None else 0

    # --- 分箱统计 (Binning) ---
    dist = combined_data[:, 0]
    azimuth = combined_data[:, 1]
    vr_vals = combined_data[:, 2]

    # 方位角
    angle_indices = np.floor((azimuth + 22.5) / 45.0).astype(int)
    angle_indices[angle_indices == 8] = 0

    # 径向距离
    bin_width = BIN_WIDTH_FACTOR * Rad
    max_bins = int(MAX_RADIUS_FACTOR / BIN_WIDTH_FACTOR)
    radial_indices = np.floor(dist / bin_width).astype(int)
    
    df = pd.DataFrame({'sector': angle_indices, 'radial_bin': radial_indices, 'vr': vr_vals})
    df = df[df['radial_bin'] < max_bins]
    
    # === 修改核心逻辑：同时计算 Mean 和 Count ===
    # agg(['mean', 'count']) 用于获取均值和粒子数
    grouped = df.groupby(['sector', 'radial_bin'])['vr'].agg(['mean', 'count']).reset_index()
    grouped.rename(columns={'mean': 'vr', 'count': 'n_particles'}, inplace=True)
    
    profile_results = []
    target_bins = np.arange(max_bins)
    
    for sector_idx in range(8):
        sector_df = grouped[grouped['sector'] == sector_idx]
        
        # 补全 bin
        full_df = pd.DataFrame({'radial_bin': target_bins})
        merged = pd.merge(full_df, sector_df, on='radial_bin', how='left')
        
        # 计算 X 轴
        x_norm = target_bins * BIN_WIDTH_FACTOR + (BIN_WIDTH_FACTOR / 2.0)
        
        # 获取数据列
        y_vr = merged['vr'].values
        counts = merged['n_particles'].fillna(0).values # 填充 NaN 数量为 0
        
        # === 核心清洗步骤 ===
        
        # 1. 最小粒子数阈值过滤
        # 如果粒子数 <= Threshold (10)，则数据不可信，设为 NaN
        y_vr[counts <= MIN_PARTICLE_THRESHOLD] = np.nan
        
        # 2. 负速度过滤
        # 理论上溅射物径向速度应 > 0，负值设为 NaN
        y_vr[y_vr < 0] = np.nan
        
        profile_results.append((x_norm, y_vr))

    return profile_results, Rad, info


# =====================================
# 主程序
# =====================================
if __name__ == "__main__":
    os.chdir("/public2/yuezy/Apollo")
    # 1. 配置数据
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

    angle_list = [90, 75, 60, 45, 30, 15] 
    angle_labels_deg = np.arange(0, 360, 45)
    row_labels = ["Pure basalt", "Ice on surface", "Ice in middle"]

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 绘图循环
    fig, axes = plt.subplots(3, 6, figsize=(18.4, 10), sharex=True, sharey=True)
    colors = plt.cm.coolwarm(np.linspace(0, 1, 8))
    
    print(f"开始处理数据 (Vr Filter: N>{MIN_PARTICLE_THRESHOLD}, Vr>=0)...")

    for i in range(3):
        for j in range(6):
            subdir = folder[i][j]
            ax = axes[i, j]
            fpath = os.path.join('.', subdir)
            
            data_list, Rad, info = process_folder(
                fpath, offset[i][j], major[i][j], minor[i][j], subdir
            )
            
            # 日志输出
            src = info.get('source', 'Unknown')
            pts = info.get('points', 0)
            if data_list:
                print(f"[{i+1},{j+1}] {subdir}: Src=[{src}] Pts=[{pts}]")
            else:
                print(f"[{i+1},{j+1}] {subdir}: SKIP/ERR")
                ax.axis('off')
                continue

            # 绘图
            for k, (x_vals, y_vals) in enumerate(data_list):
                # 如果整条线都是 NaN，跳过不画
                if not np.all(np.isnan(y_vals)):
                    ax.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=0.8, 
                            color=colors[k], label=f'{angle_labels_deg[k]}°')

            ax.set_xlim(left=0, right=5)
            ax.set_ylim(bottom=0, top=200)
            ax.grid(True, which='both', linestyle='--', alpha=0.4)
            
            if i == 0:
                ax.set_title(f"Angle={angle_list[j]}°", fontsize=15)
            
            if i == 0 and j == 0:
                ax.legend(title='Azimuth', fontsize=9, loc='upper left')

            if i == 2:
                ax.set_xlabel(r"$r/R_t$", fontsize=15)
                ax.set_xticks([0, 1, 2, 3, 4, 5])
                ax.tick_params(axis='x', labelsize=15)

            # if j == 0:
            #     ax.set_ylabel(f"{row_labels[i]}\n Radial Velocity (m/s)", fontsize=11, fontweight='bold')

            if j == 0:
                ax.set_ylabel("Radial Velocity (m/s)", fontsize=15)
                ax.tick_params(axis='y', labelsize=15)

            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(axis='x', style='plain')

    plt.tight_layout()
    
    # 3. 保存文件
    save_path = os.path.join(OUTPUT_DIR, "Fig5_Ejecta_Vr")
    print(f"正在保存图像到: {save_path} ...")
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight')
    
    plt.show()
    print("完成。")
