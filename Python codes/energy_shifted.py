import numpy as np

def shift_energy(filename, output, energy_start_col):
    data = np.loadtxt(filename, skiprows=1)
    vbm = np.max(data[:, energy_start_col])  
    data[:, energy_start_col:] -= vbm       
    np.savetxt(output, data)
    return vbm

vbm1 = shift_energy('eigenvalues_map294.txt', 'eigenvalues_map294_shifted.txt', energy_start_col=8)
vbm2 = shift_energy('bandstr1 cho tb.txt',    'bandstr1_tb_shifted.txt',        energy_start_col=1)
vbm3 = shift_energy('bandstr1 3NN.txt',       'bandstr1_3nn_shifted.txt',       energy_start_col=1)

print("VBM1 =", vbm1)
print("VBM2 =", vbm2)
print("VBM3 =", vbm3)
