import matplotlib.pyplot as plt
import numpy as np

def read_selected_columns(filename, col1, col2):
    data = []
    with open(filename, 'r') as file:
        next(file)  
        for line in file:
            if line.strip() == '':
                break
            parts = line.strip().split()
            try:
                kx = float(parts[col1])  
                p = float(parts[col2])
                data.append([kx, p])
            except ValueError:
                continue 
    return np.array(data, dtype=float)

data = []
kx_seen = set()  

with open("Dx_Dy_kx_ky.txt", "r") as file:
    next(file)  
    for line in file:
        line = line.strip()
        if not line or line.startswith("#"): 
            continue
        parts = line.split()
        try:
            kx = float(parts[0])  
            ky = float(parts[1])  
            D_abs = float(parts[6]) 

            if kx not in kx_seen and -1.0 <= kx <= 1.0 and ky == 0.0:
                data.append((kx, D_abs))
                kx_seen.add(kx)  
        except ValueError:
            print(f"⚠️ Không thể chuyển đổi dữ liệu: {line}")
            continue

data = np.array(data)
kx = data[:, 0]
D = data[:, 1]

data1 = read_selected_columns('momentum cho tb.txt', 0, 6)  
data2 = read_selected_columns('momentum cho kp.txt', 0, 6) 
data3 = read_selected_columns('Dx Dy Global.txt', 0, 6) 
data4 = read_selected_columns('file test.txt', 0, 6)  

kx1, p1 = data1[:, 0], data1[:, 1]  
kx2, p2 = data2[:, 0], data2[:, 1]  
kx3, p3 = data3[:, 0], data3[:, 1]  
kx4, p4 = data4[:, 0], data4[:, 1]  

idx_peak = np.argmax(p2)  
kx2_peak = kx2[idx_peak] 
kx_shifted = kx2 + (2/3 - kx2_peak)

plt.xlim(2/3 - 0.1, 2/3 + 0.1)
#plt.xlim(0.625, 0.725)
plt.plot(kx1, p1, label='tight-binding', linestyle='--', linewidth=1.0)
plt.plot(kx_shifted, p2, label='k.p', linestyle='-', linewidth=1.0)
#plt.plot(kx3, p3, label='|D|', linestyle='-', linewidth=1.0)
plt.plot(kx4, p4, label='corrected tight-binding', linestyle='--', linewidth=1.0)

plt.xlabel(r"$k_x \left(\frac{2\pi}{a}\right)$")

plt.ylabel('$|p|$')
plt.legend(title="Loại dữ liệu", loc="best")

plt.show()
