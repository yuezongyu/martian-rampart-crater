import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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
MAX_RADIUS_FACTOR = 30   # max_radius
BIN_WIDTH_FACTOR = 0.2   # first_width = 0.2 * Rad
CACHE_FILENAME = "ejecta_ice_rigorous.pkl" # 缓存文件名
OUTPUT_DIR = "Figures"

# === 新增：最小统计阈值 ===
# 只有当某个 Bin 中的粒子总数大于此值时，才计算百分比
# 否则视为统计样本不足，置为 NaN
MIN_PARTICLE_THRESHOLD = 10 

# =====================================
# 数学工具：向量化计算点到椭圆最短距离
# =====================================
def get_distance_to_ellipse_vectorized(x, y, a, b):
    """
    使用牛顿迭代法向量化计算点 (x, y) 到椭圆 (x/a)^2 + (y/b)^2 = 1 的最短几何距离。
    """
    # 1. 初始猜测角度 t
    t = np.arctan2(a * y, b * x)
    
    # 2. 牛顿迭代求解 f'(t) = 0
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

    # 3. 计算距离
    x_ellipse = a * np.cos(t)
    y_ellipse = b * np.sin(t)
    dist = np.hypot(x - x_ellipse, y - y_ellipse)
    return dist

# =====================================
# 核心处理函数
# =====================================
def process_file_vectorized(filepath, off, a_major, b_minor):
    """
    处理单个文件，返回: (distance_from_rim, azimuth_deg, label)
    """
    try:
        df = pd.read_csv(filepath, sep=r'[,\s]+', header=None, 
                         usecols=[2, 5, 6, 7, 10, 11, 12], engine='python')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        data = df.values.astype(np.float64)
    except Exception:
        return None

    if data.shape[0] == 0:
        return None

    # 解包
    lbl = np.abs(data[:, 0].astype(int))
    x0, y0, z0 = data[:, 1], data[:, 2], data[:, 3]
    vx, vy, vz = data[:, 4], data[:, 5], data[:, 6]

    # 1. 空间与位移筛选
    mask = (x0 >= -3000.0) & (x0 <= 2000.0)
    x_corrected = x0 - off

    # 2. 物理弹道
    D = vz**2 + 2 * G_MARS * z0
    mask &= (D >= 0)
    D_safe = np.where(D >= 0, D, 0)
    t = (vz + np.sqrt(D_safe)) / G_MARS
    mask &= (t >= 0)

    # 3. 落地位置
    x_land = x_corrected + vx * t
    y_land = y0 + vy * t

    # 4. 椭圆内部过滤
    inside = (x_land**2)/(a_major**2) + (y_land**2)/(b_minor**2) <= 1
    mask &= (~inside)
    
    x_land = x_land[mask]
    y_land = y_land[mask]
    lbl = lbl[mask]
    
    if len(x_land) == 0:
        return None

    # 5. 严谨几何距离 (Newton Iteration)
    dist_from_rim = get_distance_to_ellipse_vectorized(x_land, y_land, a_major, b_minor)
    
    # 6. 远场筛选
    Rad = 0.5 * (a_major + b_minor)
    valid_dist_mask = (dist_from_rim <= MAX_RADIUS_FACTOR * Rad) & (dist_from_rim >= 0)
    
    d_final = dist_from_rim[valid_dist_mask]
    x_final = x_land[valid_dist_mask]
    y_final = y_land[valid_dist_mask]
    l_final = lbl[valid_dist_mask]

    # 7. 方位角
    azimuth = np.degrees(np.arctan2(y_final, x_final)) % 360

    return np.column_stack((d_final, azimuth, l_final))


def process_folder(base_dir, offset, major, minor, subdir_name):
    """
    处理文件夹，计算径向冰分布曲线。
    """
    info = {'folder': subdir_name}
    Rad = 0.5 * (major + minor)
    
    target_dir = os.path.join(base_dir, "ejecta")
    datafile = os.path.join(target_dir, CACHE_FILENAME)
    
    combined_data = None

    # --- 阶段 1: 获取数据 ---
    if os.path.exists(datafile) and os.path.getsize(datafile) > 0:
        try:
            with open(datafile, "rb") as f:
                combined_data = pickle.load(f)
            info['source'] = 'Cache (.pkl)'
            info['path'] = datafile
        except Exception as e:
            info['error'] = str(e)
    
    if combined_data is None:
        if not os.path.exists(target_dir):
            return None, Rad, info
        
        info['source'] = 'Raw Data (.ejecta)'
        info['path'] = target_dir
        
        pattern = re.compile(r'^bm\.proc\d{1,4}\.(\d{4})\.ejecta$')
        files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) 
                 if pattern.match(f) and os.path.getsize(os.path.join(target_dir, f)) > 0]
        
        info['total_files'] = len(files)
        
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

    # --- 阶段 2: 统计分布 (Binning) ---
    dist = combined_data[:, 0]
    azimuth = combined_data[:, 1]
    labels = combined_data[:, 2].astype(int)

    # 方位角 Binning
    angle_indices = np.floor((azimuth + 22.5) / 45.0).astype(int)
    angle_indices[angle_indices == 8] = 0

    # 径向 Binning
    bin_width = BIN_WIDTH_FACTOR * Rad
    max_bins = int(MAX_RADIUS_FACTOR / BIN_WIDTH_FACTOR)
    radial_indices = np.floor(dist / bin_width).astype(int)
    
    # 聚合
    df = pd.DataFrame({
        'sector': angle_indices,
        'radial_bin': radial_indices,
        'label': labels,
        'count': 1
    })
    
    df = df[df['radial_bin'] < max_bins]

    def count_ice(x):
        return (x == 2).sum()
    
    grouped = df.groupby(['sector', 'radial_bin'])['label'].agg(['count', count_ice]).reset_index()
    grouped.rename(columns={'count': 'total_vol', 'count_ice': 'ice_vol'}, inplace=True)
    
    # 格式化输出
    profile_results = []
    target_bins = np.arange(max_bins)
    
    for sector_idx in range(8):
        sector_df = grouped[grouped['sector'] == sector_idx].copy()
        
        # 补全 Bin，无数据处填0 (方便下面判断粒子数)
        full_df = pd.DataFrame({'radial_bin': target_bins})
        merged = pd.merge(full_df, sector_df, on='radial_bin', how='left').fillna(0)
        
        # X 轴 (归一化距离)
        radii = merged['radial_bin'] * BIN_WIDTH_FACTOR + (BIN_WIDTH_FACTOR / 2.0)
        
        # === 核心修改：基于阈值的统计 ===
        total_counts = merged['total_vol'].values
        ice_counts = merged['ice_vol'].values
        
        # 1. 初始化全为 NaN
        percents = np.full_like(total_counts, np.nan, dtype=np.float64)
        
        # 2. 找到满足阈值的有效索引 (Total > 5)
        valid_mask = total_counts > MIN_PARTICLE_THRESHOLD
        
        # 3. 仅在有效位置计算百分比
        if np.any(valid_mask):
            percents[valid_mask] = (ice_counts[valid_mask] / total_counts[valid_mask]) * 100.0
        
        # 截断 (X <= 5.5)
        mask_x = radii <= 5.5
        profile_results.append((radii[mask_x], percents[mask_x]))

    return profile_results, Rad, info


# ========== 主程序入口 ==========
if __name__ == "__main__":
    os.chdir("/public2/yuezy/Apollo")
    # 1. 参数定义
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
    row_labels = ["Ice on surface", "Ice in middle"]
    angle_labels_deg = np.arange(0, 360, 45)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 开始处理与绘图
    print(f"开始计算径向冰体积分布 (Rigorous Dist, Threshold > {MIN_PARTICLE_THRESHOLD} particles)...")
    
    fig, axes = plt.subplots(2, 6, figsize=(18.4, 6.8), sharex=True, sharey=True)
    colors = plt.cm.coolwarm(np.linspace(0, 1, 8))
    
    row_offset = 1 

    for i in range(2):
        for j in range(6):
            actual_row = i + row_offset
            subdir = folder[actual_row][j]
            
            fpath = os.path.join('.', subdir)
            data, Rad, info = process_folder(fpath, offset[actual_row][j], 
                                             major[actual_row][j], minor[actual_row][j], subdir)
            
            # 打印日志
            src = info.get('source', 'Unknown')
            pts = info.get('points', 0)
            path_used = info.get('path', '')
            if data:
                print(f"[{i+1},{j+1}] {subdir}: Src=[{src}] Pts=[{pts}] Path=[{path_used}]")
            else:
                print(f"[{i+1},{j+1}] {subdir}: SKIP/ERR -> {info.get('error','')}")

            ax = axes[i, j]

            if data is None:
                ax.set_title(f"Angle={angle[j]}°\n(No data)", fontsize=10)
                ax.axis('off')
                continue

            # 绘图
            for k, (radii, percents) in enumerate(data):
                # 检查：如果全是 NaN，则不画
                if np.all(np.isnan(percents)):
                    continue

                label_str = f'{angle_labels_deg[k]}°'
                # Matplotlib 自动处理 NaN，会在 NaN 处断开
                ax.plot(radii, percents, marker='o', markersize=3, linewidth=1, 
                        color=colors[k], label=label_str, alpha=0.8)

            ax.set_xlim(left=0, right=5)
            ax.set_ylim(0, 105)
            ax.grid(True, which='both', linestyle='--', alpha=0.5)

            if i == 0:
                ax.set_title(f"Angle={angle[j]}°", fontsize=15)

            # if j == 0:
            #     ax.set_ylabel(f"{row_labels[i]}\n %", fontsize=11, fontweight='bold')

            if j == 0:
                ax.set_ylabel("Ice fraction (%)", fontsize=15)
                ax.tick_params(axis='y', labelsize=15)

            if i == 1:
                ax.set_xlabel(r"$r/R_t$", fontsize=15)
                ax.set_xticks([0, 1, 2, 3, 4, 5])
                ax.tick_params(axis='x', labelsize=15)

            if i == 1 and j == 0:
                ax.legend(title='Azimuth', fontsize=10, loc='upper right', framealpha=0.9)

            # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.tight_layout()
    
    # 3. 保存
    figname = os.path.join(OUTPUT_DIR, "Fig4_Radial_Ice_Distribution")
    print(f"正在保存结果到: {figname} (.png/.pdf)")
    plt.savefig(f"{figname}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{figname}.pdf", format='pdf', bbox_inches='tight')
    
    plt.show()
    print("完成。")