import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.colors as colors

def plot_contour_from_file(filename):
    # Đọc dữ liệu thủ công để xử lý số phức
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:  # Bỏ qua dòng trống
                continue
            parts = line.split()
            if len(parts) < 4:  # Kiểm tra nếu không đủ cột
                print(f"⚠️ Dòng lỗi hoặc thiếu dữ liệu: {line}")
                continue
            try:
                kx = float(parts[0])
                ky = float(parts[1])
                p_plus = float(parts[6])  # |pp|
                p_minus = float(parts[6]) # |pm|
                data.append((kx, ky, p_plus, p_minus))
            except ValueError:
                print(f"⚠️ Không thể chuyển đổi dữ liệu: {line}")
                continue

    if not data:
        print("❌ Không có dữ liệu hợp lệ để vẽ.")
        return

    # Chuyển dữ liệu thành numpy array
    data = np.array(data)

    # Tách các cột
    kx = data[:, 0]
    ky = data[:, 1]
    p_plus = data[:, 2]
    p_minus = data[:, 3]

    # Sắp xếp dữ liệu để có lưới đúng
    kx_unique = np.unique(kx)
    ky_unique = np.unique(ky)
    kx_grid, ky_grid = np.meshgrid(kx_unique, ky_unique)

    # Khởi tạo lưới rỗng
    p_plus_grid = np.zeros_like(kx_grid, dtype=float)
    p_minus_grid = np.zeros_like(ky_grid, dtype=float)

    # Điền dữ liệu vào lưới
    for i in range(len(kx)):
        xi = np.where(kx_unique == kx[i])[0][0]
        yi = np.where(ky_unique == ky[i])[0][0]
        p_plus_grid[yi, xi] = p_plus[i]
        p_minus_grid[yi, xi] = p_minus[i]

    # Lấy min/max chung để chuẩn hóa màu
    vmin = min(p_plus_grid.min(), p_minus_grid.min())
    vmax = max(p_plus_grid.max(), p_minus_grid.max())

    cmap = "viridis"

    # Tạo figure với subplot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Tạo colorbar chung ngoài hình
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Vẽ biểu đồ contour
    cp1 = axs[0].contourf(kx_grid, ky_grid, p_plus_grid, levels=10, cmap=cmap, vmin=vmin, vmax=vmax)
    cp2 = axs[1].contourf(kx_grid, ky_grid, p_minus_grid, levels=10, cmap=cmap, vmin=vmin, vmax=vmax)

    # Cài đặt tiêu đề và trục
    axs[0].set_xlabel(r"$k_x \, \left(\frac{2\pi}{a}\right)$")
    axs[0].set_ylabel(r"$k_y \, \left(\frac{2\pi}{a}\right)$")
    axs[0].set_title(r"$|P_x + iP_y|$")

    axs[1].set_xlabel(r"$k_x \, \left(\frac{2\pi}{a}\right)$")
    axs[1].set_ylabel(r"$k_y \, \left(\frac{2\pi}{a}\right)$")
    axs[1].set_title(r"$|P_x - iP_y|$")

    # Dùng ScalarMappable để tạo colorbar đúng
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Không cần dữ liệu thực tế, chỉ cần thang màu

    # Thêm colorbar
    cbar = fig.colorbar(sm, cax=cax)  
    cbar.set_label(r"Amplitude")

    plt.show()

# Gọi hàm với file của bạn
plot_contour_from_file("momentum cho 3NN.txt")
