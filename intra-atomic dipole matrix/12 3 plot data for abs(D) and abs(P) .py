import numpy as np
import matplotlib.pyplot as plt

data = []
with open('file test.txt', 'r') as file:
    next(file)  
    for line in file:
        parts = line.strip().split()
        kx = float(parts[0])
        ky = float(parts[1])
        D_abs = float(parts[6])  
        data.append((kx, ky, D_abs))

data = np.array(data)
kx = data[:, 0]
ky = data[:, 1]
D = data[:, 2]

kx_unique = np.unique(kx)
ky_unique = np.unique(ky)
kx_grid, ky_grid = np.meshgrid(kx_unique, ky_unique)

D_grid = np.zeros_like(kx_grid, dtype=float)
for i in range(len(kx_unique)):
    for j in range(len(ky_unique)):
        mask = (kx == kx_unique[i]) & (ky == ky_unique[j])
        if np.any(mask):
            D_grid[j, i] = D[mask][0]

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(kx_grid, ky_grid, D_grid, levels=10, cmap='viridis')

cbar = plt.colorbar(contour)
cbar.set_label(r'$|P|$')

ax.set_xlabel(r'$k_x \left(\frac{2\pi}{a}\right)$')
ax.set_ylabel(r'$k_y \left(\frac{2\pi}{a}\right)$')
ax.set_title(r'$|P| = \sqrt{|P_x|^2 + |P_y|^2}$')

plt.show()
