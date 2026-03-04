import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import vtk
from vtk.util.numpy_support import vtk_to_numpy
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


# ==========================================
# 用户配置区
# ==========================================
FOLDERS = [
    ['Rampart-job2002','Rampart-job2003','Rampart-job2004','Rampart-job2005','Rampart-job2006','Rampart-job2007'], 
    ['Rampart-job2008','Rampart-job2016','Rampart-job2075','Rampart-job2055','Rampart-job2056','Rampart-job2061'], 
    ['Rampart-job2114','Rampart-job2116','Rampart-job2119','Rampart-job2120','Rampart-job2121','Rampart-job2145']
]

MAJOR_AXIS = [
    [1180,1180,1250,1200,1125,1180],
    [1269,1259,1327,1329,1327,1697],
    [1470,1459,1525,1425,1425,1100]
]

MINOR_AXIS = [
    [1180,1125,1200,1200,1100,970],
    [1269,1267,1260,1267,1270,1120],
    [1470,1467,1467,1497,1446,980]
]

ANGLES = [90, 75, 60, 45, 30, 15]
ROW_LABELS = ["Pure Basalt", "Ice on Surface", "Ice in Middle"]

BASE_PATH = "/public2/yuezy/Apollo" 
SLICE_EPSILON = 20.0 
OUTPUT_DIR = "Figures" # 修改输出目录以区分
# 【重要修改】更改缓存目录名，确保不混用 XOZ 的数据
CACHE_DIR = "Cache_Initial_Matrix_YOZ" 
COL_PRESSURE_INDEX = 8 

# ==========================================
# 基础读取函数
# ==========================================
def load_initial_tracers_slice(vtp_directory, step=0, slice_epsilon=1.0):
    pattern = os.path.join(vtp_directory, f"*.proc*.{step}.vtp")
    files = glob.glob(pattern)
    if not files: return None

    pts_list = []
    reader = vtk.vtkXMLPolyDataReader()
    
    for f in files:
        reader.SetFileName(f)
        reader.Update()
        polydata = reader.GetOutput()
        if polydata.GetNumberOfPoints() == 0: continue
        
        pts = vtk_to_numpy(polydata.GetPoints().GetData())
        pd_data = polydata.GetPointData()
        
        if pd_data.HasArray("id") and pd_data.HasArray("matid"):
            ids = vtk_to_numpy(pd_data.GetArray("id"))
            mats = vtk_to_numpy(pd_data.GetArray("matid"))
        else: continue
        
        # 【重要修改】切片逻辑： YOZ 面意味着 X 坐标接近 0
        # pts[:, 0] 是 x, pts[:, 1] 是 y
        mask = np.abs(pts[:, 0]) < slice_epsilon 
        
        if np.any(mask):
            # 筛选数据
            pts_filtered = pts[mask]
            ids_filtered = ids[mask]
            mats_filtered = mats[mask]
            
            # 【重要修改】保存数据：保存 y 和 z (即 pts[:, 1] 和 pts[:, 2])
            # 格式: [id, matid, y, z]
            combined = np.column_stack((ids_filtered, mats_filtered, pts_filtered[:, 1], pts_filtered[:, 2]))
            pts_list.append(combined)

    if not pts_list: return None
    
    # 【重要修改】DataFrame 列名改为 y_init
    df = pd.DataFrame(np.vstack(pts_list), columns=['id', 'matid', 'y_init', 'z_init'])
    df['id'] = df['id'].astype(np.int64) 
    df['matid'] = df['matid'].astype(np.int64) 
    return df

def load_ejecta_ids_and_pressure(ejecta_directory):
    pattern = os.path.join(ejecta_directory, "bm.proc*.ejecta")
    files = glob.glob(pattern)
    if not files: return None
        
    data_list = []
    col_p = COL_PRESSURE_INDEX
    
    for fp in files:
        try:
            if os.path.getsize(fp) == 0: continue
            df_temp = pd.read_csv(fp, sep=r'[,\s]+', header=None, engine='python')
            if df_temp.shape[1] <= col_p: continue
            df_subset = df_temp.iloc[:, [0, col_p]].copy()
            df_subset.columns = ['id', 'pressure']
            data_list.append(df_subset)
        except Exception: continue
            
    if not data_list: return None
    df_ejecta = pd.concat(data_list, ignore_index=True)
    df_ejecta['id'] = df_ejecta['id'].astype(np.int64)
    df_ejecta['pressure'] = df_ejecta['pressure'] / 1.0e9 
    return df_ejecta

# ==========================================
# 缓存逻辑函数
# ==========================================
def get_data_with_cache(folder_name, base_path, epsilon):
    cache_path = os.path.join(CACHE_DIR, f"{folder_name}_slice_yoz.pkl")
    
    # 1. 尝试读取缓存
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception: pass

    # 2. 计算
    print(f"  [Compute] Reading raw VTP/Ejecta files for {folder_name}...")
    vtp_dir = os.path.join(base_path, folder_name, "vtp")
    ejecta_dir = os.path.join(folder_name, "ejecta")
    if not os.path.exists(ejecta_dir):
        ejecta_dir = os.path.join(base_path, folder_name, "ejecta")

    df_initial = load_initial_tracers_slice(vtp_dir, step=0, slice_epsilon=epsilon)
    df_ejecta = load_ejecta_ids_and_pressure(ejecta_dir)
    
    df_merged = pd.DataFrame()
    if df_initial is not None and df_ejecta is not None:
        df_merged = pd.merge(df_initial, df_ejecta, on='id', how='inner')
    
    # 3. 保存数据
    data_to_save = {'background': None, 'ejecta': None}
    
    if df_initial is not None:
        # 【重要修改】保存 y_init
        data_to_save['background'] = df_initial[['matid', 'y_init', 'z_init']].copy()
        
    if not df_merged.empty:
        # 【重要修改】保存 y_init
        data_to_save['ejecta'] = df_merged[['y_init', 'z_init', 'pressure']].copy()

    # 4. 写入
    try:
        if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
        with open(cache_path, "wb") as f:
            pickle.dump(data_to_save, f)
    except Exception: pass

    return data_to_save

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    os.chdir("/public2/yuezy/Apollo")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)

    fig, axes = plt.subplots(3, 6, figsize=(18.4, 10), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.05, right=0.92, top=0.92, bottom=0.08, wspace=0.1, hspace=0.2)

    vmin, vmax = 0.1, 100.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    try: cmap = plt.cm.turbo 
    except: cmap = plt.cm.jet

    print("开始处理 3x6 矩阵数据 (YOZ Plane)...")

    for i in range(3):
        for j in range(6):
            folder_name = FOLDERS[i][j]
            ax = axes[i][j]
            
            print(f"[{i+1},{j+1}] Processing: {folder_name}")
            data = get_data_with_cache(folder_name, BASE_PATH, SLICE_EPSILON)
            
            R = (MAJOR_AXIS[i][j] + MINOR_AXIS[i][j]) * 0.5
            df_bg = data.get('background')
            df_ej = data.get('ejecta')

            # === 1. 绘制背景 (Background) ===
            if df_bg is not None and not df_bg.empty:
                # 【重要修改】读取 y_init
                y_bg = df_bg['y_init'].values / R
                z_bg = df_bg['z_init'].values / R
                mat_bg = df_bg['matid'].values
                
                if i == 0:
                    ax.scatter(y_bg, z_bg, c='gray', s=0.5, alpha=0.3, rasterized=True, zorder=1)
                else:
                    mask_ice = (mat_bg == 2)
                    mask_rock = ~mask_ice
                    
                    if np.any(mask_rock):
                        ax.scatter(y_bg[mask_rock], z_bg[mask_rock], 
                                   c='gray', s=0.5, alpha=0.3, rasterized=True, zorder=1)
                    
                    if np.any(mask_ice):
                        ax.scatter(y_bg[mask_ice], z_bg[mask_ice], 
                                   c='lightgray', 
                                   s=0.5, alpha=0.5, rasterized=True, zorder=1)
            else:
                ax.text(0.5, 0.5, "No VTP", ha='center', transform=ax.transAxes, fontsize=8)

            # === 2. 绘制喷射物 (Ejecta) ===
            if df_ej is not None and not df_ej.empty:
                # 【重要修改】读取 y_init
                y_ej = df_ej['y_init'].values / R
                z_ej = df_ej['z_init'].values / R
                p_ej = df_ej['pressure'].values
                
                ax.scatter(y_ej, z_ej, c=p_ej, norm=norm, cmap=cmap, s=1.0, rasterized=True, zorder=2)
            
            ax.set_aspect('equal')
            ax.set_xlim(-1.2, 1.2) 
            ax.set_ylim(-1.2, 1.2)
            
            if i == 0: ax.set_title(f"Angle={ANGLES[j]}°", fontsize=15)
            if j == 0: 
                ax.set_ylabel(r"$Z_{init}/R_t$", fontsize=15)
                ax.tick_params(axis='y', labelsize=15)
            if i == 2: 
                ax.set_xlabel(r"$Y_{init}/R_t$", fontsize=15)
                ax.tick_params(axis='x', labelsize=15)
                ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cbar.set_label('Peak Shock Pressure (GPa)', fontsize=15)
    cbar.ax.tick_params(labelsize=15)

    print("处理完成，正在保存图像...")
    output_filename = os.path.join(OUTPUT_DIR, "Fig7_Peak_Shock_Pressure_YOZ")

    print(f"\n保存图像中: {OUTPUT_DIR}")
    fig.savefig(f"{output_filename}.png", dpi=600, bbox_inches='tight', pad_inches=0.1)
    print(f"已保存: {output_filename}.png (600 DPI)")
    
    fig.savefig(f"{output_filename}.pdf", dpi=600, bbox_inches='tight', pad_inches=0.1)
    print(f"已保存: {output_filename}.pdf (矢量格式)")

    print("处理完成，显示图像...")
    plt.show()  