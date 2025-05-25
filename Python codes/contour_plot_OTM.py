import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def read_kx_ky_val(filename, column_index):
    kx_list, ky_list, val_list = [], [], []
    with open(filename, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = line.strip().replace("(", "").replace(")", "").replace(",", "").split()
            try:
                kx = float(parts[0])
                ky = float(parts[1])
                value = float(parts[column_index])
                kx_list.append(kx)
                ky_list.append(ky)
                val_list.append(value)
            except ValueError:
                continue
    return np.array(kx_list), np.array(ky_list), np.array(val_list)

files = [
    ("thirdnn_momentum.txt", "3rd NN", 2),         
    ("nntb_momentum.txt", "nearest neighbor tbm", 2),  
    ("p_cv_map294.txt", "Mo with s,p,d", 4)             
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for i, (filename, title, column_index) in enumerate(files):
    kx, ky, val = read_kx_ky_val(filename, column_index=column_index)

    if len(np.unique(ky)) < 3:
        print(f"⚠️ File {filename} không đủ chiều ky để nội suy.")
        continue

    kx_lin = np.linspace(kx.min(), kx.max(), 200)
    ky_lin = np.linspace(ky.min(), ky.max(), 200)
    kx_grid, ky_grid = np.meshgrid(kx_lin, ky_lin)
    val_grid = griddata((kx, ky), val, (kx_grid, ky_grid), method='linear')  # hoặc 'cubic'

    # Vẽ contour
    cp = axes[i].contourf(kx_grid, ky_grid, val_grid, levels=100, cmap='jet')
    axes[i].set_title(title)
    axes[i].set_xlabel("$k_x$")
    axes[i].set_ylabel("$k_y$")
    axes[i].set_aspect('equal')

cbar = fig.colorbar(cp, ax=axes.ravel().tolist(), shrink=0.5, label="|p-|")
plt.show()
