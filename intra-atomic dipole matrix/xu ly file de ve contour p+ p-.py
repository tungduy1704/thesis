import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.colors as colors

def plot_contour_from_file(filename, xlim=None, ylim=None):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):  
                continue
            parts = line.split()
            if len(parts) < 6:  
                print(f"⚠️ Dòng lỗi hoặc thiếu dữ liệu: {line}")
                continue
            try:
                kx = float(parts[0])
                ky = float(parts[1])
                D_plus = float(parts[6])  
                D_minus = float(parts[6]) 
                data.append((kx, ky, D_plus, D_minus))
            except ValueError:
                print(f"⚠️ Không thể chuyển đổi dữ liệu: {line}")
                continue

    if not data:
        print("❌ Không có dữ liệu hợp lệ để vẽ.")
        return

    data = np.array(data)
    
    kx = data[:, 0]
    ky = data[:, 1]
    D_plus = data[:, 2]
    D_minus = data[:, 3]
    
    mask = np.ones_like(kx, dtype=bool)
    if xlim:
        mask &= (kx >= xlim[0]) & (kx <= xlim[1])
    if ylim:
        mask &= (ky >= ylim[0]) & (ky <= ylim[1])

    kx_filtered = kx[mask]
    ky_filtered = ky[mask]
    D_plus_filtered = D_plus[mask]
    D_minus_filtered = D_minus[mask]

    kx_unique = np.unique(kx_filtered)
    ky_unique = np.unique(ky_filtered)
    kx_grid, ky_grid = np.meshgrid(kx_unique, ky_unique)

    D_plus_grid = np.zeros_like(kx_grid, dtype=float)
    D_minus_grid = np.zeros_like(ky_grid, dtype=float)

    for i in range(len(kx_filtered)):
        xi = np.where(kx_unique == kx_filtered[i])[0][0]
        yi = np.where(ky_unique == ky_filtered[i])[0][0]
        D_plus_grid[yi, xi] = D_plus_filtered[i]
        D_minus_grid[yi, xi] = D_minus_filtered[i]

    vmin = min(D_plus_filtered.min(), D_minus_filtered.min())
    vmax = max(D_plus_filtered.max(), D_minus_filtered.max())

    cmap = "viridis"

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cp1 = axs[0].contourf(kx_grid, ky_grid, D_plus_grid, levels=10, cmap=cmap, vmin=vmin, vmax=vmax)
    cp2 = axs[1].contourf(kx_grid, ky_grid, D_minus_grid, levels=10, cmap=cmap, vmin=vmin, vmax=vmax)

    axs[0].set_xlabel(r"$k_x \, \left(\frac{2\pi}{a}\right)$")
    axs[0].set_ylabel(r"$k_y \, \left(\frac{2\pi}{a}\right)$")
    axs[0].set_title(r"$|P_x + i P_y|$")

    axs[1].set_xlabel(r"$k_x \, \left(\frac{2\pi}{a}\right)$")
    axs[1].set_ylabel(r"$k_y \, \left(\frac{2\pi}{a}\right)$")
    axs[1].set_title(r"$|P_x - i P_y|$")

    if xlim:
        axs[0].set_xlim(xlim)
        axs[1].set_xlim(xlim)

    if ylim:
        axs[0].set_ylim(ylim)
        axs[1].set_ylim(ylim)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  

    cbar = fig.colorbar(sm, cax=cax)  
    cbar.set_label(r"Amplitude")

    annot = axs[1].annotate("", xy=(0, 0), xytext=(15, 15),
                            textcoords="offset points", ha="center", fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    annot.set_visible(False)

    def update_annot(event):
        if event.inaxes == axs[1]:  
            x_mouse, y_mouse = event.xdata, event.ydata  
            
            if x_mouse is not None and y_mouse is not None:
                i = np.abs(kx_unique - x_mouse).argmin()
                j = np.abs(ky_unique - y_mouse).argmin()

                value = D_minus_grid[j, i]  

                annot.set_text(f"Amp: {value:.2f}")
                annot.xy = (x_mouse, y_mouse)
                annot.set_visible(True)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", update_annot)

    plt.show()

# ✅ **Gọi hàm với giới hạn kx và ky mong muốn**
#plot_contour_from_file("momentum cho tb.txt", xlim=(2/3 - 0.1, 2/3 + 0.1), ylim=(-0.1, 0.1))
#plot_contour_from_file("momentum cho tb.txt", xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
#plot_contour_from_file("file test.txt", xlim=(2/3 - 0.1, 2/3 + 0.1), ylim=(-0.1, 0.1))
plot_contour_from_file("file test.txt", xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
#plot_contour_from_file("Dx Dy Global.txt", xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
#plot_contour_from_file("file test.txt", xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
#plot_contour_from_file("momentum cho kp.txt", xlim=(-0.1, 0.1), ylim=(-0.1, 0.1))