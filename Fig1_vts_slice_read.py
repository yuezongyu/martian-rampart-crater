import xml.etree.ElementTree as ET
import base64
import zlib
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import os

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

# ----------------- 核心读取工具函数 -------------------

def read_uint_from_buffer(buf, offset, count, header_type="UInt32"):
    """
    辅助函数：从buffer中读取指定数量的整数
    """
    if header_type == "UInt64":
        fmt = "<" + "Q" * count
        size = 8 * count
    else:
        fmt = "<" + "I" * count
        size = 4 * count
    
    if offset + size > len(buf):
        raise ValueError("Header buffer too small.")
        
    vals = struct.unpack_from(fmt, buf, offset)
    return list(vals), offset + size

def parse_data_array(data_element, header_type, compressor, expected_elements):
    """
    通用解析函数：解析 XML 中的 DataArray 节点
    """
    data_dict = {}
    if data_element is None:
        return data_dict

    is_compressed = compressor is not None and "ZLib" in compressor

    for arr in data_element.findall("DataArray"):
        name = arr.attrib.get("Name", "Coords")
        ncomp = int(arr.attrib.get("NumberOfComponents", "1"))
        dtype = arr.attrib.get("type")
        
        raw_b64 = "".join((arr.text or "").split())
        if not raw_b64:
            continue

        h_size = 8 if header_type == "UInt64" else 4
        
        head_peek = base64.b64decode(raw_b64[:24])
        num_blocks = struct.unpack_from("<Q" if header_type == "UInt64" else "<I", head_peek, 0)[0]
        
        header_bytes_len = (3 + num_blocks) * h_size
        header_b64_len = ((header_bytes_len + 2) // 3) * 4
        
        b64_header = raw_b64[:header_b64_len]
        b64_data = raw_b64[header_b64_len:]
        
        decoded_header = base64.b64decode(b64_header)
        decoded_data = base64.b64decode(b64_data)
        
        header_content = decoded_header[:header_bytes_len]
        data_prefix = decoded_header[header_bytes_len:]
        compressed_payload = data_prefix + decoded_data
        
        offset = 0
        (vals, offset) = read_uint_from_buffer(header_content, offset, 3, header_type)
        (c_sizes, offset) = read_uint_from_buffer(header_content, offset, num_blocks, header_type)
        
        out_buffer = bytearray()
        if is_compressed:
            data_offset = 0
            for size in c_sizes:
                c_block = compressed_payload[data_offset : data_offset + size]
                data_offset += size
                try:
                    decompressed_chunk = zlib.decompress(c_block)
                except zlib.error:
                    decompressed_chunk = zlib.decompress(c_block, zlib.MAX_WBITS + 32)
                out_buffer.extend(decompressed_chunk)
            raw_bytes = bytes(out_buffer)
        else:
            raw_bytes = compressed_payload

        if dtype == "Float32": np_dt = np.float32
        elif dtype == "Float64": np_dt = np.float64
        elif dtype == "Int32": np_dt = np.int32
        elif dtype == "UInt8": np_dt = np.uint8
        else: np_dt = np.float32

        arr_np = np.frombuffer(raw_bytes, dtype=np_dt)
        expected_size = expected_elements * ncomp
        
        if arr_np.size != expected_size: continue
        
        if ncomp > 1: data_dict[name] = arr_np.reshape((expected_elements, ncomp))
        else: data_dict[name] = arr_np
            
    return data_dict

def read_vts(filename):
    """
    读取 VTS 文件的核心函数
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    header_type = root.attrib.get("header_type", "UInt32")
    compressor = root.attrib.get("compressor", None)

    sg = root.find("StructuredGrid")
    if sg is None: raise ValueError("No StructuredGrid")
        
    whole_extent = list(map(int, sg.attrib["WholeExtent"].split()))
    nx = whole_extent[1] - whole_extent[0] + 1
    ny = whole_extent[3] - whole_extent[2] + 1
    nz = whole_extent[5] - whole_extent[4] + 1
    npoints = nx * ny * nz
    
    dims_minus_1 = []
    if nx > 1: dims_minus_1.append(nx - 1)
    if ny > 1: dims_minus_1.append(ny - 1)
    if nz > 1: dims_minus_1.append(nz - 1)
    
    if not dims_minus_1: ncells = 0
    else:
        ncells = 1
        for d in dims_minus_1: ncells *= d

    piece = sg.find("Piece")
    if piece is None: piece = sg

    pd_elem = piece.find("PointData"); point_data = parse_data_array(pd_elem, header_type, compressor, npoints)
    cd_elem = piece.find("CellData"); cell_data = parse_data_array(cd_elem, header_type, compressor, ncells)
    pts_elem = piece.find("Points"); points_dict = parse_data_array(pts_elem, header_type, compressor, npoints)
    
    coords = None
    if points_dict: coords = list(points_dict.values())[0]

    all_data = {**point_data, **cell_data}

    return {
        "nx": nx, "ny": ny, "nz": nz,
        "npoints": npoints, "ncells": ncells,
        "data": all_data,
        "coords": coords
    }

# ----------------- 主程序 (批量处理与绘图) -------------------
if __name__ == "__main__":
    os.chdir("/public2/yuezy/Apollo")
    OUTPUT_DIR = "Figures"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 定义数据源
    folder = [
        ['Rampart-usr-j2002','Rampart-usr-j2003','Rampart-usr-j2004','Rampart-usr-j2005','Rampart-usr-j2006','Rampart-usr-j2007'],
        ['Rampart-usr-j2008','Rampart-usr-j2016','Rampart-usr-j2075','Rampart-usr-j2055','Rampart-usr-j2056','Rampart-usr-j2061'],
        ['Rampart-usr-j2114','Rampart-usr-j2116','Rampart-usr-j2119','Rampart-usr-j2120','Rampart-usr-j2121','Rampart-usr-j2145']
    ]
    
    file = [
        ['Slice.00xoz.37.vts','Slice.00xoz.40.vts','Slice.00xoz.35.vts','Slice.00xoz.31.vts','Slice.00xoz.26.vts','Slice.00xoz.22.vts'],
        ['Slice.00xoz.40.vts','Slice.00xoz.31.vts','Slice.00xoz.16.vts','Slice.00xoz.19.vts','Slice.00xoz.17.vts','Slice.00xoz.17.vts'],
        ['Slice.00xoz.24.vts','Slice.00xoz.23.vts','Slice.00xoz.23.vts','Slice.00xoz.32.vts','Slice.00xoz.32.vts','Slice.00xoz.33.vts']
    ]

    angle_titles = ["Angle=90°", "Angle=75°", "Angle=60°", "Angle=45°", "Angle=30°", "Angle=15°"]

    rows = 3
    cols = 6
    
    # --- 2. 设置画布 (优化布局) ---
    # 您的每个子图范围是 X[-25, 15] (宽40), Y[-10, 10] (高20) -> 比例 2:1
    # 6列 x 3行 -> 整体比例大约是 12:3 = 4:1
    # 设置 figsize 为 (24, 7) 可以让整体形状更扁，从而减少上下的空白
    fontsize = 8
    fontABCD = 9
    
    fig, axes = plt.subplots(rows, cols, figsize=(18.4, 6), sharey=True, sharex=True)
    
    # 手动调整子图间距，使其更紧凑
    # hspace: 上下间距, wspace: 左右间距
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.1, hspace=0.15, wspace=0.1)
    
    # --- 色板定义 ---
    # 第一行: 0=透明, 1=红色, 2=绿色
    cmap_row1 = ListedColormap([
        (0, 0, 0, 0),       # Index 0
        (1, 0, 0, 1),       # Index 1
        (0, 0.8, 0, 1)      # Index 2
    ])
    
    # 第二/三行: 0=透明, 1=红色, 2=蓝色, 3=绿色
    cmap_row23 = ListedColormap([
        (0, 0, 0, 0),       # Index 0
        (1, 0, 0, 1),       # Index 1
        (0, 0, 1, 1),       # Index 2
        (0, 0.8, 0, 1)      # Index 3
    ])

    print(f"开始批量处理 {rows * cols} 个文件...")

    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            
            dir_name = folder[i][j]
            file_name = file[i][j]
            full_path = f"./{dir_name}/{file_name}"
            
            # --- 样式设置 ---
            # 修改 X 轴显示范围为 -25 到 15
            ax.set_xlim(-25, 15)
            # Y 轴保持 -10 到 10
            ax.set_ylim(-10, 10)
            
            ax.set_aspect('equal', adjustable='box')

            # 标题设置
            if i == 0:
                ax.set_title(angle_titles[j], fontsize=15, fontweight='bold')
            else:
                ax.set_title("")

            # X轴标签
            if i == rows - 1:
                ax.set_xlabel(r"$x/a$", fontsize=15)
                ax.tick_params(axis='x', labelsize=15)
            else:
                ax.set_xlabel("")
                # 可选：隐藏内部刻度数字
                # ax.set_xticklabels([])
            
            # Y轴标签
            if j == 0:
                ax.set_ylabel(r"$y/a$", fontsize=15)
                ax.tick_params(axis='y', labelsize=15)
            else:
                ax.set_ylabel("")
                # 可选：隐藏内部刻度数字
                # ax.set_yticklabels([])

            # --- 绘制实心圆 (红色) ---
            # 物理坐标圆心(0, 100) -> 归一化 (0, 1), 半径 1
            circle = Circle((0, 1), 1.0, color='red', zorder=10)
            ax.add_patch(circle)

            try:
                if not os.path.exists(full_path):
                    raise FileNotFoundError
                    
                grid = read_vts(full_path)
                data = grid["data"]
                
                if grid["coords"] is not None:
                    coords_flat = grid["coords"]
                    
                    # --- 坐标处理与归一化 ---
                    ranges = [
                        ('X', coords_flat[:, 0].max() - coords_flat[:, 0].min(), 0),
                        ('Y', coords_flat[:, 1].max() - coords_flat[:, 1].min(), 1),
                        ('Z', coords_flat[:, 2].max() - coords_flat[:, 2].min(), 2)
                    ]
                    ranges.sort(key=lambda x: x[1], reverse=True)
                    idx1, idx2 = ranges[0][2], ranges[1][2]
                    
                    coords_grid = coords_flat.reshape((grid['ny'], grid['nx'], 3))
                    
                    # 归一化 /100
                    X_grid = coords_grid[:, :, idx1] / 100.0
                    Y_grid = coords_grid[:, :, idx2] / 100.0
                    
                    ncx, ncy = grid['nx'] - 1, grid['ny'] - 1
                    
                    # --- 获取 VOF 和 Density 数据 ---
                    def get_data(key_name, default_val=0):
                        if key_name in data:
                            return data[key_name].flatten()
                        else:
                            return np.full(ncx * ncy, default_val)

                    v0 = get_data("VOF-0")
                    v1 = get_data("VOF-1")
                    v2 = get_data("VOF-2")
                    
                    rho_flat = get_data("density")
                    rho_grid = rho_flat.reshape((ncy, ncx))

                    # --- 计算颜色索引 ---
                    if i == 0:
                        # 第一行: Index 0, 1, 2
                        v_stack = np.stack([v0, v1, v2], axis=-1)
                        category_flat = np.argmax(v_stack, axis=-1)
                        category_grid = category_flat.reshape((ncy, ncx))
                        
                        current_cmap = cmap_row1
                        vmax_val = 2
                    else:
                        # 第二/三行: Index 0, 1, 2, 3
                        v3 = get_data("VOF-3")
                        v_stack = np.stack([v0, v1, v2, v3], axis=-1)
                        category_flat = np.argmax(v_stack, axis=-1)
                        category_grid = category_flat.reshape((ncy, ncx))
                        
                        current_cmap = cmap_row23
                        vmax_val = 3

                    # --- 应用 Density < 300 透明化 ---
                    mask_low_density = (rho_grid < 300)
                    category_grid[mask_low_density] = 0

                    # --- 绘图 ---
                    # ax.pcolormesh(X_grid, Y_grid, category_grid, 
                    #               cmap=current_cmap, shading='flat', 
                    #               vmin=0, vmax=vmax_val)
                    ax.pcolormesh(X_grid, Y_grid, category_grid, 
                                    cmap=current_cmap, shading='flat', 
                                    vmin=0, vmax=vmax_val, 
                                    rasterized=True)

                else:
                    ax.text(0, 0, "No Coords", ha='center', fontsize=8)

            except FileNotFoundError:
                ax.text(0, 0, "File Not Found", ha='center', color='red', fontsize=8)
            except Exception as e:
                print(f"Error in {dir_name}: {e}")
                ax.text(0, 0, "Error", ha='center', color='red', fontsize=8)

    
    print("处理完成，正在保存图像...")
    # 定义输出文件名 (不带后缀)
    output_filename = os.path.join(OUTPUT_DIR, "Fig1_VOF_Density_Visualization_Grid")
    
    # --- 保存方式 1: 高质量位图 (PNG) ---
    # 适用于 PPT 展示、网页或快速查看
    # dpi=600: 设置分辨率为 600 DPI (学术出版通常要求 >=300)
    # bbox_inches='tight': 自动裁切掉周围多余的空白区域，非常重要！
    # pad_inches=0.1: 在裁切边缘留一点点空隙，防止切到文字
    fig.savefig(f"{output_filename}.png", dpi=600, bbox_inches='tight', pad_inches=0.1)
    print(f"已保存: {output_filename}.png (600 DPI)")

    # --- 保存方式 2: 矢量图 (PDF 或 SVG) ---
    # 强烈推荐用于学术文章投稿。
    # 矢量图可以无限放大而不失真，且文字在 PDF 中可以被选中和编辑。
    # fig.savefig(f"{output_filename}.pdf", bbox_inches='tight', pad_inches=0.1)
    fig.savefig(f"{output_filename}.pdf", dpi=600, bbox_inches='tight', pad_inches=0.1)
    print(f"已保存: {output_filename}.pdf (矢量格式)")

    print("处理完成，显示图像...")
    plt.show()