import numpy as np
import matplotlib.pyplot as plt

def read_data_for_contour_kp(filename):
    kx, ky, p = [], [], []
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            if line.strip() == '':
                continue  
            parts = line.strip().split()
            kx.append(float(parts[0]))
            ky.append(float(parts[1]))
            p.append(float(parts[6])) 
    return np.array(kx), np.array(ky), np.array(p)

def read_data_for_contour_tbm(filename):
    """Đọc dữ liệu từ file và trả về kx, ky, |p|"""
    kx, ky, p = [], [], []
    with open(filename, 'r') as f:
        next(f)  
        for line in f:
            if line.strip() == '':
                continue  
            parts = line.strip().split()
            kx.append(float(parts[0]))
            ky.append(float(parts[1]))
            p.append(float(parts[6])) 
    return np.array(kx), np.array(ky), np.array(p)

kx1, ky1, p1 = read_data_for_contour_kp('momentum cho kp.txt')
kx2, ky2, p2 = read_data_for_contour_tbm('file test.txt')  

kx1_grid, ky1_grid = np.meshgrid(np.unique(kx1), np.unique(ky1))
p1_grid = p1.reshape(kx1_grid.shape)

kx2_grid, ky2_grid = np.meshgrid(np.unique(kx2), np.unique(ky2))
p2_grid = p2.reshape(kx2_grid.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

contour1 = axes[0].contourf(kx1_grid, ky1_grid, p1_grid, cmap='viridis')
axes[0].set_title('|p| of k.p')
axes[0].set_xlabel('kx')
axes[0].set_ylabel('ky')
fig.colorbar(contour1, ax=axes[0])

contour2 = axes[1].contourf(kx2_grid, ky2_grid, p2_grid, cmap='viridis')
axes[1].set_title('|p| of corrected tight-binding')
axes[1].set_xlabel('kx')
axes[1].set_ylabel('ky')
#axes[1].set_xlim([2/3 - 0.1, 2/3 + 0.1])
#axes[1].set_ylim([-0.1, 0.1])
fig.colorbar(contour2, ax=axes[1])

plt.tight_layout()
plt.show()