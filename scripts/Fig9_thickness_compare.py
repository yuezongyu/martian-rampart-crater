import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import LogLocator, LogFormatterMathtext, ScalarFormatter
from scipy.optimize import least_squares

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
# 1. 全局配置与常量
# =====================================
G_MARS = 3.721
VOLUME_PER_EJECTA_3D = 1000.0  # 3D 模拟中每个粒子的体积
OUTPUT_DIR = "Figures"

# # 字体配置 (带容错处理)
# FONT_PATH = '/public2/yuezy/Apollo/TimesNewRoman.ttf'
# if os.path.exists(FONT_PATH):
#     font_manager.fontManager.addfont(FONT_PATH)
#     plt.rcParams['font.family'] = 'Times New Roman'
# else:
#     plt.rcParams['font.family'] = 'serif'

# =====================================
# 2. 理论模型与拟合函数
# =====================================
def thickness_scaling(x, p, Rad):
    """理论厚度缩放公式 (用于绘制参考线)"""
    a, b, c = p
    return (a * Rad**b * np.power(x, c))

def thickness_fit_func(x, p, Rad):
    """拟合用的简化公式: t = a * R * (x)^c """
    a, c = p
    return (a * Rad * np.power(x, c))

def residuals(p, x, y, Rad):
    """
    拟合残差函数
    【逻辑保留】: (y - model) / np.log(y)
    注意：这种加权方式倾向于拟合 y 值较大的区域（即靠近中心的厚层区域）。
    """
    model = thickness_fit_func(x, p, Rad)
    
    # 避免 log(y) 计算错误 (理论上厚度都 > 0)
    valid_mask = y > 0
    res = np.zeros_like(y)
    
    if np.any(valid_mask):
        res[valid_mask] = (y[valid_mask] - model[valid_mask]) / np.log(y[valid_mask])
        
    return res

# =====================================
# 3. 2D 数据处理工具
# =====================================
def load_paratest_step(dpath, step):
    """读取 iSALE 2D 二进制输出文件"""
    data_dict = {}
    files = ["ix", "iy", "eX", "eT", "eU", "eV", "ex", "et"]
    for key in files:
        fname = f"ParaTest.{key}.{step:04d}.bin"
        fpath = os.path.join(dpath, fname)
        if os.path.exists(fpath):
            data_dict[key] = np.fromfile(fpath, dtype="float32")
        else:
            data_dict[key] = np.array([])
    return data_dict

def calc_ejecta_volume_2d(ix, dr=5.0, dz=5.0, dtheta=1.0):
    """计算 2D 柱坐标网格体积"""
    ix = np.asarray(ix)
    r_in = ix * dr
    r_out = (ix + 1) * dr
    return 0.5 * (r_out**2 - r_in**2) * dtheta * dz

# =====================================
# 4. 3D 数据处理工具 (向量化优化)
# =====================================
def process_folder_3d(input_dir, Rad, subdir):
    """
    处理 3D Ejecta 数据文件夹。
    返回: (landed_r_array, total_files)
    """
    cache_file = os.path.join(input_dir, "compare_optimized.pkl")
    
    # --- 1. 尝试读取缓存 ---
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            print(f"{subdir} [3D]: Loaded cache, points={len(data)}")
            return data, -1 
        except Exception:
            pass

    # --- 2. 读取原始数据 ---
    if not os.path.exists(input_dir):
        print(f"Warning: Directory {input_dir} not found.")
        return np.array([]), 0

    pattern = re.compile(r'^bm\.proc\d{1,4}\.(\d{4})\.ejecta$')
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if pattern.match(f)]
    
    total_files = len(files)
    valid_points_list = []
    
    # 筛选参数
    max_radius = 30
    total_width = max_radius * Rad

    for fpath in files:
        if os.path.getsize(fpath) == 0: continue
        
        try:
            # 仅读取需要的列: 5=x, 6=y, 7=z, 10=vx, 11=vy, 12=vz
            df = pd.read_csv(fpath, sep=r'[,\s]+', header=None, 
                             usecols=[5, 6, 7, 10, 11, 12], engine='python')
            data = df.values.astype(np.float64)
        except Exception:
            continue
            
        if data.shape[0] == 0: continue

        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        vx, vy, vz = data[:, 3], data[:, 4], data[:, 5]

        # --- 向量化物理计算 ---
        mask = (x >= -3000.0)
        
        # 弹道
        D = vz**2 + 2 * G_MARS * z
        mask &= (D >= 0)
        D_safe = np.where(D >= 0, D, 0)
        t = (vz + np.sqrt(D_safe)) / G_MARS
        mask &= (t >= 0)
        
        # 落点
        x_land = x + vx * t
        y_land = y + vy * t
        dis = np.hypot(x_land, y_land)
        
        # 几何筛选: 坑内(dis/Rad <= 1) 或 极远处的点 去掉
        mask &= (dis > Rad) & (dis <= total_width + Rad)
        
        valid_r = dis[mask]
        if len(valid_r) > 0:
            valid_points_list.append(valid_r)

    if valid_points_list:
        all_r = np.concatenate(valid_points_list)
    else:
        all_r = np.array([])

    # 保存缓存
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(all_r, f)
    except:
        pass
        
    print(f"{subdir} [3D]: Processed {total_files} files, valid points={len(all_r)}")
    return all_r, total_files

# =====================================
# 5. 主程序入口
# =====================================
if __name__ == "__main__":
    os.chdir("/public2/yuezy/Apollo")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # === 参数设置 ===
    folder = [
        ['Rampart-job2002','Rampart-job2008','Rampart-job2114'], # 3D Cases
        ['Rampart-job2141','Rampart-job2142','Rampart-job2144']  # 2D Cases
    ]  
    Rad_list = [1300, 1350, 1500]
    
    # 标题定义
    row_labels = ["Pure basalt", "Ice on surface", "Ice in middle"]
    
    # 绘图初始化
    fig, axes = plt.subplots(1, 3, figsize=(18.4, 5), sharey=True)
    
    for j in range(3):
        ax = axes[j]
        current_Rad = Rad_list[j]
        
        # (A) 设置子图标题
        # ax.set_title(row_labels[j], fontsize=14, fontweight='bold')

        # (B) 定义分箱 (Binning)
        bin_width = 300.0
        max_radius = 30
        total_width = max_radius * current_Rad
        bin_edges = np.arange(current_Rad * 1.1, total_width, bin_width)
        bin_centers_all = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # 筛选绘图范围 (1.0 R - 10.0 R)
        min_rad_factor = 1.0
        max_rad_factor = 10.0
        mask_plot = (bin_centers_all >= min_rad_factor * current_Rad) & \
                    (bin_centers_all <= max_rad_factor * current_Rad)
        bin_centers_plot = bin_centers_all[mask_plot]

        # -------------------
        # 1. 处理 3D 数据
        # -------------------
        subdir_3D = folder[0][j]
        fpath_3D = os.path.join('.', subdir_3D, 'ejecta')
        
        r3D_array, _ = process_folder_3d(fpath_3D, current_Rad, subdir_3D)
        
        if len(r3D_array) > 0:
            counts_3D, _ = np.histogram(r3D_array, bins=bin_edges)
            vol_3D_bins = counts_3D * VOLUME_PER_EJECTA_3D
            area_3D_bins = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
            thickness_3D_all = vol_3D_bins / area_3D_bins
            thickness_3D_plot = thickness_3D_all[mask_plot]
            
            # 绘制 3D 散点
            ax.scatter(bin_centers_plot/current_Rad, thickness_3D_plot/current_Rad, 
                       s=15, alpha=0.9, edgecolors="none", color='black', label='SALEc')
            
            # --- 3D 拟合 ---
            # 过滤无效值
            mask_fit = (thickness_3D_plot > 0)
            if np.sum(mask_fit) > 2:
                x_fit = bin_centers_plot[mask_fit] / current_Rad
                y_fit = thickness_3D_plot[mask_fit]
                
                try:
                    # 使用 Least Squares 进行拟合
                    # Bounds: a in [0.001, 1.0], c in [-3.0, 0] (保留原程序限制)
                    res_lsq = least_squares(residuals, [0.0294, -3.0], 
                                            args=(x_fit, y_fit, current_Rad), 
                                            bounds=((0.001, -3.0), (1.0, 0)), 
                                            ftol=1e-10)
                    
                    a_fit, c_fit = res_lsq.x
                    
                    # 绘制拟合线
                    er_line = np.linspace(1, max_rad_factor, 100)
                    fit_line = thickness_fit_func(er_line, res_lsq.x, current_Rad) / current_Rad
                    # ax.plot(er_line, fit_line, "-", color="red", linewidth=2, label='Fit')
                    
                    # (C) 结果展示
                    # 1. 绘图用 LaTeX 格式
                    formula_latex = r"$t = {:.3f} R (x/R)^{{{:.2f}}}$".format(a_fit, c_fit)
                    ax.text(0.95, 0.95, formula_latex, transform=ax.transAxes, 
                            fontsize=14, color='red', verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.6, edgecolor='none'))
                    
                    # 2. 控制台打印用 普通文本 格式
                    formula_console = "t = {:.3f} * R * (x/R)^{:.2f}".format(a_fit, c_fit)
                    print(f"[{subdir_3D}] Formula: {formula_console}")
                    
                except Exception as e:
                    print(f"[{subdir_3D}] Fit failed: {e}")
            else:
                print(f"[{subdir_3D}] Not enough points for fitting.")

        else:
            print(f"[{subdir_3D}] No valid 3D data points.")

        # -------------------
        # 2. 处理 2D 数据
        # -------------------
        subdir_2D = folder[1][j]
        fpath_2D = os.path.join('/public/home/yuezy/sale2d-main/models/', subdir_2D, 'post')
        
        # 本地调试兼容
        if not os.path.exists(fpath_2D):
             fpath_2D_local = os.path.join('.', subdir_2D, 'post')
             if os.path.exists(fpath_2D_local):
                 fpath_2D = fpath_2D_local

        result_2D = load_paratest_step(fpath_2D, 500)
        
        if len(result_2D.get('eX', [])) > 0:
            eX = result_2D['eX']
            ix = result_2D['ix']
            vol_per_particle_2D = calc_ejecta_volume_2d(ix)
            vol_2D_bins, _ = np.histogram(eX, bins=bin_edges, weights=vol_per_particle_2D)
            dr_bins = bin_edges[1:] - bin_edges[:-1]
            thickness_2D_all = vol_2D_bins / (bin_centers_all * dr_bins)
            thickness_2D_plot = thickness_2D_all[mask_plot]
            
            ax.scatter(bin_centers_plot/current_Rad, thickness_2D_plot/current_Rad, 
                       s=15, alpha=0.9, edgecolors="none", color='red', marker='s', label='SALEc-2D')
        else:
            # print(f"[{subdir_2D}] No valid 2D data found.")
            pass

        ax.plot(er_line, fit_line, "-", color="red", linewidth=2, label='SALEc Fit')
        # -------------------
        # 3. 绘制参考曲线
        # -------------------
        er_ref = np.linspace(1, max_rad_factor, 50)
        refs = [
            # ([0.014, 1.01, -3.0], "purple", "Sharpton (2014)"),
            # ([0.0078, 1.0, -2.61], "orange", "Housen (1983)"),
            ([0.033, 1.0, -3.0], "blue", "Pike (1974)"),
            # ([0.14, 0.74, -3.0], "brown", "McGetchin (1973)"),
            ([0.0294, 1.0, -2.96], "green", "Li (2025)")
        ]
        
        for params, col, lab in refs:
            y_ref = thickness_scaling(er_ref, params, current_Rad) / current_Rad
            ax.plot(er_ref, y_ref, "--", color=col, linewidth=1.5, alpha=0.7, label=lab)

        # -------------------
        # 4. 图表格式化
        # -------------------
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1.0, 10.0)
        ax.set_ylim(1e-5, 1.15*1e-1)
        
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        
        ax.set_xlabel(r"$r/R$", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        if j == 0:
            ax.set_ylabel(r"$t/R$", fontsize=15)
            # 图例
            ax.legend(fontsize=14, loc='lower left')
            ax.text(1.05, 0.055, 'A', fontsize=15, fontweight='bold')

        if j == 1:
            ax.text(1.05, 0.055, 'B', fontsize=15, fontweight='bold')

        if j == 2:
            ax.text(1.05, 0.055, 'C', fontsize=15, fontweight='bold')

    plt.tight_layout()
    
    # 保存结果
    save_name = os.path.join(OUTPUT_DIR, "Fig9_Ejecta_Thickness")
    print(f"Saving figures to: {save_name}...")
    plt.savefig(f"{save_name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_name}.pdf", format='pdf', bbox_inches='tight')
    
    plt.show()
    print("完成。")
