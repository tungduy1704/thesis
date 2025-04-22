import numpy as np
from scipy.linalg import eigh

hb = 0.658229  # eV·fs
m0 = 5.6770736  # eV·fs²/nm²
pi = np.pi

def tb_parameters(material):
    atab = np.array([0.3190, 0.3191, 0.3326, 0.3325, 0.3557, 0.3560])
    e1tab = np.array([1.046, 1.130, 0.919, 0.943, 0.605, 0.606])
    e2tab = np.array([2.104, 2.275, 2.065, 2.179, 1.972, 2.102])
    t0tab = np.array([-0.184, -0.206, -0.188, -0.207, -0.169, -0.175])
    t1tab = np.array([0.401, 0.567, 0.317, 0.457, 0.228, 0.342])
    t2tab = np.array([0.507, 0.536, 0.456, 0.486, 0.390, 0.410])
    t11tab = np.array([0.218, 0.286, 0.211, 0.263, 0.207, 0.233])
    t12tab = np.array([0.338, 0.384, 0.290, 0.329, 0.239, 0.270])
    t22tab = np.array([0.057, -0.061, 0.130, 0.034, 0.252, 0.190])
    lambdatab = np.array([0.073, 0.211, 0.091, 0.228, 0.107, 0.237])
    
    a = atab[material - 1]
    e1 = e1tab[material - 1]
    e2 = e2tab[material - 1]
    t0 = t0tab[material - 1]
    t1 = t1tab[material - 1]
    t2 = t2tab[material - 1]
    t11 = t11tab[material - 1]
    t12 = t12tab[material - 1]
    t22 = t22tab[material - 1]
    lambda_ = lambdatab[material - 1]
    G = 4 * pi / (np.sqrt(3) * a)
    
    return a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, G

def rhombusgrid(nk, G):
    dk = G / (nk - 1)
    akx = np.zeros((nk, nk))
    aky = np.zeros((nk, nk))

    for j in range(nk):
        for i in range(nk):
            ak1 = -G / 2 + (i) * dk
            ak2 = -G / 2 + (j) * dk
            akx[i, j] = np.sqrt(3) / 2 * (ak1 + ak2)
            aky[i, j] = -0.5 * (ak1 - ak2)

    return akx, aky


def kgrid(nk, G, a):
    dk = (4 * pi / a) / (nk - 1)
    akx = np.zeros((nk, nk))
    aky = np.zeros((nk, nk))

    for j in range(nk):
        for i in range(nk):
            akx[i, j] = - 2 * pi / a + i * dk
            aky[i, j] = 0#- 2 * pi / a + j * dk

    return akx, aky

def tb_ham(kx, ky, params):
    a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, G = params
    
    alpha = kx * a / 2
    beta = np.sqrt(3) * ky * a / 2
    
    er0 = np.array([[e1, 0, 0], [0, e2, 0], [0, 0, e2]])
    er1 = np.array([[t0, t1, t2], [-t1, t11, t12], [t2, -t12, t22]])

    er2 = np.array([[t0, t1 / 2 - np.sqrt(3) * t2 / 2, -np.sqrt(3) * t1 / 2 - t2 / 2],
                    [-t1 / 2 - np.sqrt(3) * t2 / 2, t11 / 4 + 3 * t22 / 4, -np.sqrt(3) * t11 / 4 - t12 + np.sqrt(3) * t22 / 4],
                    [np.sqrt(3) * t1 / 2 - t2 / 2, -np.sqrt(3) * t11 / 4 + t12 + np.sqrt(3) * t22 / 4, 3 * t11 / 4 + t22 / 4]])

    er3 = np.array([[t0, -t1 / 2 + np.sqrt(3) * t2 / 2, -np.sqrt(3) * t1 / 2 - t2 / 2],
                    [t1 / 2 + np.sqrt(3) * t2 / 2, t11 / 4 + 3 * t22 / 4, np.sqrt(3) * t11 / 4 + t12 - np.sqrt(3) * t22 / 4],
                    [np.sqrt(3) * t1 / 2 - t2 / 2, np.sqrt(3) * t11 / 4 - t12 - np.sqrt(3) * t22 / 4, 3 * t11 / 4 + t22 / 4]])

    er4 = np.array([[t0, -t1, t2], [t1, t11, -t12], [t2, t12, t22]])

    er5 = np.array([[t0, -t1 / 2 - np.sqrt(3) * t2 / 2, np.sqrt(3) * t1 / 2 - t2 / 2],
                    [t1 / 2 - np.sqrt(3) * t2 / 2, t11 / 4 + 3 * t22 / 4, -np.sqrt(3) * t11 / 4 + t12 + np.sqrt(3) * t22 / 4],
                    [-np.sqrt(3) * t1 / 2 - t2 / 2, -np.sqrt(3) * t11 / 4 - t12 + np.sqrt(3) * t22 / 4, 3 * t11 / 4 + t22 / 4]])

    er6 = np.array([[t0, t1 / 2 + np.sqrt(3) * t2 / 2, np.sqrt(3) * t1 / 2 - t2 / 2],
                    [-t1 / 2 + np.sqrt(3) * t2 / 2, t11 / 4 + 3 * t22 / 4, np.sqrt(3) * t11 / 4 - t12 - np.sqrt(3) * t22 / 4],
                    [-np.sqrt(3) * t1 / 2 - t2 / 2, np.sqrt(3) * t11 / 4 + t12 - np.sqrt(3) * t22 / 4, 3 * t11 / 4 + t22 / 4]])
    
                    
    ham = er0 + np.exp(1j * 2 * alpha) * er1 + np.exp(1j * (alpha - beta)) * er2 + np.exp(1j * (-alpha - beta)) * er3 + np.exp(-1j * 2 * alpha) * er4 + np.exp(1j * (-alpha + beta)) * er5 + np.exp(1j * (alpha + beta)) * er6

    dhkx = (1j * a * np.exp(1j * 2 * alpha) * er1 + 1j * a / 2 * np.exp(1j * (alpha - beta)) * er2
                - 1j * a / 2 * np.exp(1j * (-alpha - beta)) * er3 - 1j * a * np.exp(-1j * 2 * alpha) * er4
                - 1j * a / 2 * np.exp(1j * (-alpha + beta)) * er5 + 1j * a / 2 * np.exp(1j * (alpha + beta)) * er6)

    dhky = (-1j * np.sqrt(3) * a / 2 * np.exp(1j * (alpha - beta)) * er2 - 1j * np.sqrt(3) * a / 2 * np.exp(1j * (-alpha - beta)) * er3
                + 1j * np.sqrt(3) * a / 2 * np.exp(1j * (-alpha + beta)) * er5 + 1j * np.sqrt(3) * a / 2 * np.exp(1j * (alpha + beta)) * er6)
        
    Lz = np.array([[0, 0, 0], [0, 0, 2j], [0, -2j, 0]], dtype=np.complex128)
    
    hamu = ham + lambda_ / 2 * Lz
    hamd = ham - lambda_ / 2 * Lz            

    return ham, hamu, hamd, dhkx, dhky

def initialize_band_data(nk):
    bandu = np.zeros((3, nk, nk))
    bandd = np.zeros((3, nk, nk))
    band = np.zeros((3, nk, nk))
    pmx = np.zeros((3, 3, nk, nk), dtype=complex)
    pmy = np.zeros((3, 3, nk, nk), dtype=complex)
    vecs_save = np.zeros((3, 3, nk, nk), dtype=complex)

    return bandu, bandd, band, pmx, pmy, vecs_save

def initialize_matrices():
    ham = np.zeros((3, 3), dtype=complex)
    hamu = np.zeros((3, 3), dtype=complex)
    hamd = np.zeros((3, 3), dtype=complex)
    dhkx = np.zeros((3, 3), dtype=complex)
    dhky = np.zeros((3, 3), dtype=complex)

    return ham, hamu, hamd, dhkx, dhky

def bandstructure(nk, akx, aky, a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, G):

    bandu, bandd, band, pmx, pmy, vecs_save = initialize_band_data(nk)

    for j in range(nk):
        for i in range(nk):
            kx, ky = akx[i, j], aky[i, j]

            ham, hamu, hamd, dhkx, dhky = initialize_matrices()
            ham, hamu, hamd, dhkx, dhky = tb_ham(kx, ky, (a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, G))

            vals, vecs = eigh(ham)
            band[:, i, j] = vals
            vecs_save[:, :, i, j] = vecs

            for jb in range(3):  # λ'
                for ib in range(3):  #  λ
                    sum1 = 0. + 0j   #  kx
                    sum2 = 0. + 0j   #  ky
                    for jjb in range(3): # j'
                        for iib in range(3): # j
                            sum1 += np.conjugate(vecs[iib, ib]) * dhkx[iib, jjb] * vecs[jjb, jb]
                            sum2 += np.conjugate(vecs[iib, ib]) * dhky[iib, jjb] * vecs[jjb, jb]
                    pmx[ib, jb, i, j] = sum1 * m0 / hb
                    pmy[ib, jb, i, j] = sum2 * m0 / hb

            bandu[:, i, j] = eigh(hamu, eigvals_only=True) 

            bandd[:, i, j] = eigh(hamd, eigvals_only=True) 

    return bandu, bandd, band, pmx, pmy, vecs_save

def save_data(nk, G, akx, aky, bandu, bandd, band, pmx, pmy, a, vecs_save):

    with open("vecs_tb.txt", "w") as f:
        for j in range(nk):
            for i in range(nk):
                kx = akx[i, j] / (2 * np.pi / a)
                ky = aky[i, j] / (2 * np.pi / a)
                f.write(f"{kx:.6e} {ky:.6e} \n")
                for band_idx in range(3):
                    for component in vecs_save[:, band_idx, i, j]:
                        f.write(f"{component.real:.6e} {component.imag:.6e} \n")
                f.write("\n")

    with open("bandstr1 cho tb.txt", "w") as f:
        for i in range(nk):
            kx = -np.sqrt(3) * G / 2 + (i) * np.sqrt(3) * G / (nk - 1)  # 
            f.write(f"{kx / (2 * np.pi / a)} {band[0, i, 0]} {band[1, i, 0]} {band[2, i, 0]}\n")

    with open("bandstr2 cho tb.txt", "w") as f:
        f.write("#kx ky band1 band2 band3 \n")
        for j in range(nk):
            for i in range(nk):
                f.write(f"{akx[i, j] / (2 * np.pi / a)} {aky[i, j] / (2 * np.pi / a)} "
                        f"{band[0, i, j]} {band[1, i, j]} {band[2, i, j]}\n")
            f.write("\n")

    """with open("bandstr_u.txt", "w") as f:
        for j in range(nk):
            for i in range(nk):
                f.write(f"{akx[i, j] / (2 * np.pi / a)} {aky[i, j] / (2 * np.pi / a)} "
                        f"{bandu[0, i, j]} {bandu[1, i, j]} {bandu[2, i, j]}\n")"""

    with open("bandstr_u.txt", "w") as f:
        for i in range(nk):
                f.write(f"{akx[i, 0] / (2 * np.pi / a)} "
                        f"{bandu[0, i, 0]} {bandu[1, i, 0]} {bandu[2, i, 0]}\n")
    
    """with open("bandstr_d.txt", "w") as f:
        for j in range(nk):
            for i in range(nk):
                f.write(f"{akx[i, j] / (2 * np.pi / a)} {aky[i, j] / (2 * np.pi / a)} "
                        f"{bandd[0, i, j]} {bandd[1, i, j]} {bandd[2, i, j]}\n")"""
   
    with open("bandstr_d.txt", "w") as f:
        for i in range(nk):
            f.write(f"{akx[i, 0] / (2 * np.pi / a)}  "
                    f"{bandd[0, i, 0]} {bandd[1, i, 0]} {bandd[2, i, 0]}\n")
    
    with open("momentum cho tb.txt", "w") as f:
        f.write(f"#{'kx':^12} {'ky':^12} {'|pp|':^12} {'|pm|':^12} {'px':^25} {'py':^25} {'|p|':^25} \n")
        for j in range(nk):
            for i in range(nk):
                px, py = pmx[0, 1, i, j], pmy[0, 1, i, j]
                p = np.sqrt((abs(px))**2 + (abs(py))**2)
                pp = px + 1j * py
                pm = px - 1j * py
                kx = akx[i, j] / (2 * np.pi / a)
                ky = aky[i, j] / (2 * np.pi / a)

                f.write(f"{kx:.6e} {ky:.6e} {abs(pp):.6e} {abs(pm):.6e} "
                        f"({px.real:.6e},{px.imag:.6e}) ({py.real:.6e},{py.imag:.6e}) {p:.6e} \n")
                
            f.write("\n") 

    with open("momentum 0_2.txt", "w") as f:
        for j in range(nk):
            for i in range(nk):
                px, py = pmx[0, 2, i, j], pmy[0, 2, i, j]
                p = np.sqrt((abs(px))**2 + (abs(py))**2)
                pp = px + 1j * py
                pm = px - 1j * py
                kx = akx[i, j] / (2 * np.pi / a)
                ky = aky[i, j] / (2 * np.pi / a)

                f.write(f"{kx:.6e} {ky:.6e} {abs(pp):.6e} {abs(pm):.6e} {abs(px):.6e} {abs(py):.6e} {abs(p):.6e} \n")
            
            f.write('\n')

def main():
    nk = 99
    material = 1  # MoS2

    a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, G = tb_parameters(material)
    akx, aky = kgrid(nk, G, a)
    bandu, bandd, band, pmx, pmy, vecs_save = bandstructure(nk, akx, aky, a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, material)

    save_data(nk, G, akx, aky, bandu, bandd, band, pmx, pmy, a, vecs_save)
    print("Hoàn tất tính toán dải năng lượng!")

if __name__ == "__main__":
    main()
