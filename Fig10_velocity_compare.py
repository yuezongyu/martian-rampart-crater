import os
import re
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import ScalarFormatter, FuncFormatter
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
# 全局配置
# =====================================
G_MARS = 3.721
OUTPUT_DIR = "Figures"
PROJECTILE_RADIUS = 100.0  # 假设弹丸半径 r_p = 100m (用于 Maxwell 模型)


# =====================================
# 理论模型函数
# =====================================

def model_maxwell(x, p, Rad):
    """
    Maxwell Z-Model: v/sqrt(gR) vs x/R
    p = [alpha, Z]
    注意：原代码中 d0 = 0.6 * r_p / Rad
    """
    alpha, Z = p
    x_norm = x / Rad
    d0 = 0.6 * PROJECTILE_RADIUS / Rad
    # v = alpha * sqrt(gR) * (x^2 + d0^2)^(-Z/2)
    # 返回归一化速度 v/sqrt(gR)
    return alpha * np.power(x_norm**2 + d0**2, -Z / 2.0)

def model_housen(x, p, Rad):
    """
    Housen Model: v/sqrt(gR) vs x/R
    p = [alpha, mu, p_exp]
    """
    alpha, mu, p_exp = p
    x_norm = x / Rad
    # 防止 (1-x) < 0 导致报错
    term2 = np.maximum(1.0 - x_norm, 1e-10)
    return alpha * np.power(x_norm, -mu) * np.power(term2, p_exp)

def model_li(x_norm):
    """Li (2025) Model: returns v/sqrt(gR)"""
    return 0.891 * np.power(x_norm, -1.78) * np.power(1.0 - x_norm, 0.573)

def model_housen2011_complex(x_norm, U, c1, a, den0, den1, mu, nu, n2, p, R, g):
    """
    Housen & Holsapple (2011) Complete Scaling
    x_norm: x/R
    """
    dr = np.power(den0/den1, nu)
    xt = x_norm * R
    # 这里的 1.3 是原代码中的系数
    term2 = 1.0 - xt / (n2 * R / 1.3)
    # 避免无效值
    term2 = np.maximum(term2, 1e-10)
    
    Vx = U * c1 * np.power(dr * xt / a, -1.0/mu) * np.power(term2, p)
    Vref = np.sqrt(g * R)
    return Vx / Vref

# =====================================
# 数据处理：2D iSALE
# =====================================
def load_paratest_step(dpath, step):
    """读取 2D 二进制数据"""
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

# =====================================
# 数据处理：3D SPH (向量化)
# =====================================
def process_folder_3d_launch(input_dir, Rad, subdir):
    """
    计算 3D 数据的发射参数 (Launch Position & Velocity)
    返回: r_launch, v_launch
    """
    cache_file = os.path.join(input_dir, "launch_optimized.pkl")
    
    # 1. 尝试读取缓存
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            # data stored as list of tuples, convert to array
            data_arr = np.array(data)
            print(f"{subdir} [3D]: Loaded cache, points={len(data_arr)}")
            return data_arr[:, 0], data_arr[:, 1]
        except Exception:
            pass

    # 2. 读取原始数据
    if not os.path.exists(input_dir):
        print(f"Warning: Directory {input_dir} not found.")
        return np.array([]), np.array([])

    pattern = re.compile(r'^bm\.proc\d{1,4}\.(\d{4})\.ejecta$')
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if pattern.match(f)]
    
    # 筛选参数
    max_radius = 30
    total_width = max_radius * Rad
    
    results = []

    for fpath in files:
        if os.path.getsize(fpath) == 0: continue
        try:
            # 读取列: 5=x, 6=y, 7=z, 10=vx, 11=vy, 12=vz (final)
            df = pd.read_csv(fpath, sep=r'[,\s]+', header=None, 
                             usecols=[5, 6, 7, 10, 11, 12], engine='python')
            data = df.values.astype(np.float64)
        except Exception:
            continue
            
        if data.shape[0] == 0: continue

        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        vx, vy, vz_final = data[:, 3], data[:, 4], data[:, 5]

        # --- 向量化物理计算 ---
        # 1. 筛选发射点范围
        mask = (x >= -3000.0)
        
        # 2. 弹道反推
        # 落地时的垂直速度 vz_land^2 = vz_final^2 + 2gz (假设vz_final即为落地速度分量，原逻辑似乎视z为高度)
        # 原逻辑：D = vz0**2 + 2*g*z ... 这里 vz0 实际上是读入的 vz。
        # 注意：iSALE ejecta 文件通常记录的是 tracer 在该时刻的速度。如果这是落地后记录，z是高程。
        # 按照原代码逻辑：
        D = vz_final**2 + 2 * G_MARS * z
        mask &= (D >= 0)
        D_safe = np.where(D >= 0, D, 0)
        
        # 飞行时间
        t_flight = (vz_final + np.sqrt(D_safe)) / G_MARS
        mask &= (t_flight >= 0)
        
        # 落地位置
        x_land = x + vx * t_flight
        y_land = y + vy * t_flight
        dis = np.hypot(x_land, y_land)
        
        # 坑内/坑外筛选
        mask &= (dis > Rad)
        
        # 3. 反算发射参数 (Launch Conditions)
        # 落地瞬间垂直速度 (标量大小)
        vz_impact = np.sqrt(D_safe)
        
        # 发射时刻 (相对于当前记录时刻的前推)
        # t_launch_offset = (vz_impact - vz_final) / g ? 
        # 原代码：t_launch = (vz - vz0) / g，其中 vz=vz_impact, vz0=vz_final
        t_back = (vz_impact - vz_final) / G_MARS
        
        x_launch = x - vx * t_back
        y_launch = y - vy * t_back
        
        # 发射速度 (Launch Velocity)
        # v_launch = sqrt(vx^2 + vy^2 + vz_impact^2)
        v_launch = np.sqrt(vx**2 + vy**2 + vz_impact**2)
        r_launch = np.hypot(x_launch, y_launch)
        
        # 收集数据
        valid_r = r_launch[mask]
        valid_v = v_launch[mask]
        
        if len(valid_r) > 0:
            results.append(np.column_stack((valid_r, valid_v)))

    if results:
        final_data = np.vstack(results)
        r_out, v_out = final_data[:, 0], final_data[:, 1]
    else:
        r_out, v_out = np.array([]), np.array([])

    # 保存缓存 (以 list of tuples 格式兼容原逻辑，或直接存 array)
    try:
        with open(cache_file, "wb") as f:
            # 存为 list of tuples 以匹配原程序的读取习惯，或者直接存 numpy
            pickle.dump(final_data.tolist(), f)
    except:
        pass
        
    print(f"{subdir} [3D]: Processed files, valid points={len(r_out)}")
    return r_out, v_out

# =====================================
# 主程序
# =====================================
if __name__ == "__main__":
    os.chdir("/public2/yuezy/Apollo")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # === 参数设置 ===
    folder = [
        ['Rampart-job2002','Rampart-job2008','Rampart-job2114'], # 3D
        ['Rampart-job2141','Rampart-job2142','Rampart-job2144']  # 2D
    ]
    Rad_list = [1300, 1350, 1500]
    
    # 标题定义
    row_labels = ["Pure basalt", "Ice on surface", "Ice in middle"]
    
    # 筛选范围 (x/R)
    xmin_fit, xmax_fit = 0.15, 0.90

    # 绘图初始化
    fig, axes = plt.subplots(1, 3, figsize=(18.4, 6), sharey=True)
    
    # 定义参考线参数字典
    # ref_models = {
    #     "Yamamoto et al.(2005)": {"U": 240.0, "c1": 1.0, "mu": 0.45, "R": 6.95e-2, "n2": 1.3, "p": 0.3, "a": 4.9e-3, "den0": 1500.0, "den1": 970.0, "nu": 0.4, "g": 9.81, "color": "purple"},
    #     "Schmidt&Housen(1987)": {"U": 4600.0, "c1": 1.5, "mu": 0.55, "R": 14.3e-2*1.3, "n2":1.5, "p":0.5, "a":1.0e-3, "den0":1000.0, "den1":2050.0, "nu":0.4, "g":9.81, "color": "orange"},
    #     "Cintala et al.(1999)": {"U": 1920.0, "c1": 0.55, "mu": 0.41, "R": 9.1e-2, "n2":1.3, "p":0.3, "a":2.4e-3, "den0":1600.0, "den1":2400.0, "nu":0.4, "g":9.81, "color": "blue"},
    #     "Stoffler et al.(1975)": {"U": 6770.0, "c1": 0.55, "mu": 0.41, "R": 12.7e-2 * 1.3, "n2":1.3, "p":0.3, "a":3.9e-3, "den0":1510.0, "den1":1220.0, "nu":0.4, "g":9.81, "color": "green"}
    # }

    ref_models = {
        "Yamamoto et al.(2005)": {"U": 240.0, "c1": 1.0, "mu": 0.45, "R": 6.95e-2, "n2": 1.3, "p": 0.3, "a": 4.9e-3, "den0": 1500.0, "den1": 970.0, "nu": 0.4, "g": 9.81, "color": "purple"},
        "Schmidt&Housen(1987)": {"U": 4600.0, "c1": 1.5, "mu": 0.55, "R": 14.3e-2*1.3, "n2":1.5, "p":0.5, "a":1.0e-3, "den0":1000.0, "den1":2050.0, "nu":0.4, "g":9.81, "color": "orange"},
    }

    for j in range(3):
        ax = axes[j]
        current_Rad = Rad_list[j]
        
        # 设置标题
        # ax.set_title(row_labels[j], fontsize=14, fontweight='bold')

        # -------------------
        # 1. 3D 数据
        # -------------------
        subdir_3D = folder[0][j]
        fpath_3D = os.path.join('.', subdir_3D, 'ejecta')
        
        r_3D, v_3D = process_folder_3d_launch(fpath_3D, current_Rad, subdir_3D)
        
        # 筛选拟合用数据
        mask_3d = (r_3D/current_Rad > xmin_fit) & (r_3D/current_Rad < xmax_fit)
        r_fit_3D = r_3D[mask_3d]
        v_fit_3D = v_3D[mask_3d]
        
        # -------------------
        # 2. 2D 数据
        # -------------------
        subdir_2D = folder[1][j]
        fpath_2D = os.path.join('/public/home/yuezy/sale2d-main/models/', subdir_2D, 'post')
        # 本地调试路径兼容
        if not os.path.exists(fpath_2D):
             fpath_2D_local = os.path.join('.', subdir_2D, 'post')
             if os.path.exists(fpath_2D_local): fpath_2D = fpath_2D_local

        result_2D = load_paratest_step(fpath_2D, 500)
        
        r_2D = np.array([])
        v_2D = np.array([])
        
        if len(result_2D.get('ex', [])) > 0:
            r_2D = result_2D['ex']
            v_2D = np.sqrt(result_2D['eU']**2 + result_2D['eV']**2)
            mask_2d = (r_2D/current_Rad > xmin_fit) & (r_2D/current_Rad < xmax_fit)
            r_plot_2D = r_2D[mask_2d]
            v_plot_2D = v_2D[mask_2d]
        else:
            r_plot_2D, v_plot_2D = [], []

        # -------------------
        # 3. 绘图 - 散点
        # -------------------
        # 3D
        ax.scatter(r_fit_3D/current_Rad, v_fit_3D/np.sqrt(G_MARS*current_Rad), 
                   s=10, alpha=0.8, edgecolors="face", color='red', label='SALEc')
        # 2D
        if len(r_plot_2D) > 0:
            ax.scatter(r_plot_2D/current_Rad, v_plot_2D/np.sqrt(G_MARS*current_Rad), 
                       s=10, alpha=0.8, edgecolors="face", color='black', label='SALEc-2D')

        # -------------------
        # 4. 拟合与公式展示
        # -------------------
        x_eval = np.linspace(xmin_fit, xmax_fit, 100) * current_Rad
        
        if len(r_fit_3D) > 10:
            # --- (A) Maxwell Fit ---
            # Model: v/sqrt(gR) = alpha * ((x/R)^2 + d0^2)^(-Z/2)
            # p = [alpha, Z]
            # 目标函数: residual = (y_data - y_model) / y_data
            def resid_maxwell(p, y_obs, x_obs, Rad):
                y_model = model_maxwell(x_obs, p, Rad)
                return (y_obs - y_model) / y_obs

            # 观测值归一化
            y_obs_norm = v_fit_3D / np.sqrt(G_MARS * current_Rad)
            
            try:
                res_mx = least_squares(resid_maxwell, [0.5, 2.5], 
                                       args=(y_obs_norm, r_fit_3D, current_Rad),
                                       bounds=([0.0, 1.0], [np.inf, 5.0]), ftol=1e-10)
                alpha_mx, Z_mx = res_mx.x
                
                # 绘制曲线
                y_mx_plot = model_maxwell(x_eval, res_mx.x, current_Rad)
                ax.plot(x_eval/current_Rad, y_mx_plot, "-", color="red", linewidth=1.5, label="Maxwell Fit")
                
                # 公式展示 (左下角)
                # Maxwell公式通常写为 Z-model形式
                # latex_mx = r"Maxwell: $Z={:.2f}, \alpha={:.2f}$".format(Z_mx, alpha_mx)
                latex_mx = r"Maxwell: $a_1={:.2f}, Z={:.2f}$".format(alpha_mx, Z_mx)
                # latex_mx = r"$Z={:.2f}$".format(Z_mx)
                ax.text(0.05, 0.12, latex_mx, transform=ax.transAxes, fontsize=13, color='red',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
            except Exception as e:
                print(f"Maxwell fit failed: {e}")

            # --- (B) Housen Fit ---
            # Model: v/sqrt(gR) = alpha * (x/R)^-mu * (1-x/R)^p
            # p = [alpha, mu, p_exp]
            def resid_housen(p, y_obs, x_obs, Rad):
                y_model = model_housen(x_obs, p, Rad)
                return (y_obs - y_model) / y_obs

            try:
                res_hs = least_squares(resid_housen, [0.5, 2.0, 0.5],
                                       args=(y_obs_norm, r_fit_3D, current_Rad),
                                       bounds=([0.0, 1.0, 0.0], [np.inf, 4.0, 2.0]), ftol=1e-10)
                alpha_hs, mu_hs, p_hs = res_hs.x
                
                # 绘制曲线
                y_hs_plot = model_housen(x_eval, res_hs.x, current_Rad)
                ax.plot(x_eval/current_Rad, y_hs_plot, "-", color="blue", linewidth=1.5, label="Housen Fit")
                
                # 公式展示 (左下角，位于Maxwell下方)
                # latex_hs = r"$\mu={:.2f}, p={:.2f}$".format(mu_hs, p_hs)
                latex_hs = r"H&H: $a_1={:.2f}, Z={:.2f}, p={:.2f}$".format(alpha_hs, mu_hs, p_hs)
                ax.text(0.05, 0.05, latex_hs, transform=ax.transAxes, fontsize=13, color='blue',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
            except Exception as e:
                print(f"Housen fit failed: {e}")

        # -------------------
        # 5. 绘制参考曲线
        # -------------------
        x_ref = np.linspace(xmin_fit, xmax_fit, 50)
        
        # Li 2025
        ax.plot(x_ref, model_li(x_ref), "--", color="green", linewidth=1.5, label="Li et al.(2025)")
        
        # 其他文献 (Complex Housen Model)
        for label, params in ref_models.items():
            # 需要去掉 color 键传递给函数
            p_copy = params.copy()
            color = p_copy.pop("color")
            # 绘制
            y_ref = model_housen2011_complex(x_ref, **p_copy)
            ax.plot(x_ref, y_ref, "--", color=color, linewidth=1.5, label=label)

        # -------------------
        # 6. 坐标轴与格式
        # -------------------
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(0.15, 1.0) # 对应 xmin, xmax 附近
        ax.set_ylim(0.2, 15.0)
        
        # X轴格式
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xticks([0.2, 0.3, 0.4, 0.6, 1.0])
        
        # Y轴格式
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_yticks([0.5, 1.0, 2.0, 4.0, 6.0, 10.0])
        
        ax.set_xlabel(r"$r/R_t$", fontsize=15)
        ax.tick_params(axis='x', labelsize=15)
        
        if j == 0:
            ax.text(0.85, 10, 'A', fontsize=15, fontweight='bold')
            ax.set_ylabel(r"$v/\sqrt{gR}$", fontsize=15)
            # 图例
            ax.legend(loc="lower left",bbox_to_anchor=(0.046, 0.15), fontsize=13, framealpha=0.9)
            ax.tick_params(axis='y', labelsize=15)

        if j == 1:
            ax.text(0.85, 10, 'B', fontsize=15, fontweight='bold')

        if j == 2:
            ax.text(0.85, 10, 'C', fontsize=15, fontweight='bold')

    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(OUTPUT_DIR, "Fig10_Ejecta_Velocity")
    print(f"Saving to {save_path}...")
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight')
    
    plt.show()
    print("完成。")