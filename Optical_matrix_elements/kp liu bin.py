import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# Define energy parameters (eV) and lattice 
a = 3.190
Delta = 1.663 
t1 = 1.105

t2 = 1.059 
gamma1_2 = 0.055 
gamma2_2 = 0.077 
gamma3_2 = -0.123

t3 = 1.003
gamma1_3 = 0.196 
gamma2_3 = -0.065 
gamma3_3 = -0.248 
gamma4 = 0.163 
gamma5 = -0.094 
gamma6 = -0.232

# Define Hamiltonian kp with regard to different order perturbation
def h1(kx, ky):
    arr1 =  np.array([[Delta / 2, a * t1 * (kx - 1j * ky)],
                     [a * t1 * (kx + 1j * ky), - Delta / 2]])
    return arr1

def h2(kx, ky):
    arr1 =  np.array([[Delta / 2, a * t2 * (kx - 1j * ky)],
                     [a * t2 * (kx + 1j * ky), - Delta / 2]])
    arr2 =  (a**2) * np.array([[gamma1_2 * (kx**2 + ky**2), gamma3_2 * (kx + 1j * ky)**2],
                     [gamma3_2 * (kx - 1j * ky)**2, gamma2_2 * (kx**2 + ky**2)]])
    return arr1 + arr2

def h3(kx, ky):
    arr1 =  np.array([[Delta / 2 , a * t3 * (kx - 1j * ky)],
                     [a * t3 * (kx + 1j * ky), - Delta / 2]])
    arr2 =  (a**2) * np.array([[gamma1_3 * (kx**2 + ky**2), gamma3_3 * (kx + 1j * ky)**2],
                     [gamma3_3 * (kx - 1j * ky)**2, gamma2_3 * (kx**2 + ky**2)]])
    arr3 = a**3 * np.array([[gamma4 * kx * (kx**2 - 3*ky**2) , gamma6 * (kx**2 + ky**2) * (kx - 1j*ky)],
                            [gamma6 * (kx**2 + ky**2) * (kx + 1j*ky), gamma5 * kx * (kx**2 - 3*ky**2)]])
    return arr1 + arr2 + arr3

# Iteration of three k.p Hamiltonian
N = 500
grid = np.linspace(-0.1 * 2*pi / a, 0.1 * 2*pi / a, N)

def eig():
    conduction1, valence1 = np.zeros(N), np.zeros(N)
    conduction2, valence2 = np.zeros(N), np.zeros(N)
    conduction3, valence3 = np.zeros(N), np.zeros(N)
    ky = 0
    
    for i in range(len(grid)):
        # Tính trị riêng cho mỗi Hamiltonian
        eig1 = np.linalg.eigh(h1(grid[i], ky))[0]
        eig2 = np.linalg.eigh(h2(grid[i], ky))[0]
        eig3 = np.linalg.eigh(h3(grid[i], ky))[0]
        # Chia thành conduction và valence
        conduction1[i] = max(eig1)  # Nghiệm dương (conduction band)
        valence1[i] = min(eig1)     # Nghiệm âm (valence band)
        
        conduction2[i] = max(eig2)
        valence2[i] = min(eig2)
        
        conduction3[i] = max(eig3)
        valence3[i] = min(eig3)

    return conduction1, valence1, conduction2, valence2, conduction3, valence3

conduction1, valence1, conduction2, valence2, conduction3, valence3 = eig()

# Write results to file
with open('conduction_valence_data.txt', 'w') as f:
    f.write('kx,  conduction3, valence3\n')
    for i in range(N):
        f.write(f'{grid[i] / (2*pi/a)},  {conduction3[i]}, {valence3[i]}\n')

# Plotting the eigenvalues with regard to different k.p
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Subplot (a)
ax1.plot(grid, valence1, label="k.p(1)", linestyle='--', color='blue')
ax1.plot(grid, valence2, label="k.p(2)", linestyle='-', color='red')
ax1.plot(grid, valence3, label="k.p(3)", linestyle='-', color='black')

ax1.set_xlabel(r"$kx$")
ax1.set_ylabel("Energy (eV)")
ax1.set_title("(a)")

# Customizing the x-axis labels for (Γ, K, M) points
ax1.set_xticks([-0.1 , 0, 0.1 ])
ax1.set_xticklabels([r"$\Gamma$", r"$K$", r"$M$"])

# Subplot (b)
ax2.plot(grid, conduction1, label="k.p(1)", linestyle='--', color='blue')
ax2.plot(grid, conduction2, label="k.p(2)", linestyle='-', color='red')
ax2.plot(grid, conduction3, label="k.p(3)", linestyle='-', color='black')

ax2.set_xlabel(r"$kx$")
ax2.set_ylabel("Energy (eV)")
ax2.set_title("(b)")

# Customizing the x-axis labels for (Γ, K, M) points
ax2.set_xticks([-0.1 , 0, 0.1])
ax2.set_xticklabels([r"$\Gamma$", r"$K$", r"$M$"])

# Adding legend
ax1.legend(loc="upper right")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()
