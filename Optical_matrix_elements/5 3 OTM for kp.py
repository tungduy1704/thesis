import numpy as np
from scipy.linalg import eigh, eigvals

hb = 0.658229  
m0 = 5.6770736  
pi = np.pi

Delta = 1.663 
"""#t = 1.105

t = 1.059
gamma1 = 0.055
gamma2 = 0.077
gamma3 = -0.123
"""
t = 1.003
gamma1 = 0.196 
gamma2 = -0.065 
gamma3 = -0.248 
gamma4 = 0.163 
gamma5 = -0.094 
gamma6 = -0.232

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
    dk = (4 * 0.1 *  pi / a) / (nk - 1)
    akx = np.zeros((nk, nk))
    aky = np.zeros((nk, nk))

    for j in range(nk):
        for i in range(nk):
            akx[i, j] = -0.1 * 2 * pi / a + i * dk
            aky[i, j] = 0#-0.1 * 2 * pi / a + j * dk

    return akx, aky

def kp_ham(kx, ky, params):
    a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, G = params
    
    ham1 = np.array([[Delta, a * t * (kx - 1j * ky)],
                     [a * t * (kx + 1j * ky), 0]])

    ham2 = (a**2) * np.array([[gamma1 * (kx**2 + ky**2), gamma3 * (kx + 1j * ky)**2],
                     [gamma3 * (kx - 1j * ky)**2, gamma2 * (kx**2 + ky**2)]])

    ham3 = (a**3) * np.array([[gamma4 * kx * (kx**2 - 3*ky**2) , gamma6 * (kx**2 + ky**2) * (kx - 1j*ky)],
                            [gamma6 * (kx**2 + ky**2) * (kx + 1j*ky), gamma5 * kx * (kx**2 - 3*ky**2)]])
                    
    ham = ham1  + ham2 + ham3

    dham1kx = np.array([[0, a * t ],
                        [a * t , 0]])
    dham2kx = np.array([[2*(a**2) * gamma1 * kx, 2* (a**2) * gamma3 *(kx + 1j*ky) ],
                        [2* (a**2) * gamma3 *(kx - 1j*ky), 2*(a**2) * gamma2 * kx]])
    dham3kx = np.array([[3 * (a**3) * gamma4 * (kx**2 - ky**2) , (a**3) * gamma6 * (3*kx**2 - 2*1j*kx*ky + ky**2) ],
                        [(a**3) * gamma6 * (3*kx**2 + 2*1j*kx*ky + ky**2), 3 * (a**3) * gamma5 * (kx**2 - ky**2)]])

    dham1ky = np.array([[0, -1j*a * t ],
                        [1j*a * t , 0]])
    dham2ky = np.array([[2 * (a**2) * gamma1 * ky, 2*1j* (a**2) * gamma3 * (kx + 1j*ky)  ],
                        [-2*1j* (a**2) * gamma3 * (kx - 1j*ky) , 2 * (a**2) * gamma2 * ky]])
    dham3ky = np.array([[-6 * (a**3) * gamma4 * kx * ky, (a**3) * gamma6 * (-1j*kx**2 + 2*kx*ky - 3*1j*ky**2) ],
                        [(a**3) * gamma6 * (1j*kx**2 + 2*kx*ky + 3*1j*ky**2) , -6 * (a**3) * gamma5 * kx * ky]])

    dhkx = dham1kx + dham2kx + dham3kx
    dhky = dham1ky + dham2ky + dham3ky
                
    return ham, dhkx, dhky

def initialize_band_data(nk):
    band  = np.zeros((2, nk, nk))
    pmx   = np.zeros((2, 2, nk, nk), dtype=complex)
    pmy   = np.zeros((2, 2, nk, nk), dtype=complex)
    return band, pmx, pmy

def initialize_matrices():
    ham  = np.zeros((2, 2), dtype=complex)
    dhkx = np.zeros((2, 2), dtype=complex)
    dhky = np.zeros((2, 2), dtype=complex)
    return ham, dhkx, dhky

def bandstructure(nk, akx, aky, a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, G):
    band, pmx, pmy = initialize_band_data(nk)

    for j in range(nk):
        for i in range(nk):
            kx, ky = akx[i, j], aky[i, j]

            ham, dhkx, dhky = initialize_matrices()
            ham, dhkx, dhky = kp_ham(kx, ky, (a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, G))

            vals, vecs = eigh(ham)
            band[:, i, j] = vals

            for jb in range(2):  # λ'
                for ib in range(2):  # λ
                    sum1 = 0. + 0j   # kx
                    sum2 = 0. + 0j   # ky
                    for jjb in range(2): # j'
                        for iib in range(2): # j
                            sum1 += np.conjugate(vecs[iib, ib]) * dhkx[iib, jjb] * vecs[jjb, jb]
                            sum2 += np.conjugate(vecs[iib, ib]) * dhky[iib, jjb] * vecs[jjb, jb]
                    pmx[ib, jb, i, j] = sum1 * m0 / hb
                    pmy[ib, jb, i, j] = sum2 * m0 / hb

    return band, pmx, pmy

def save_data(nk, G, akx, aky, band, pmx, pmy, a):
  
    with open("bandstr1 cho kp.txt", "w") as f:
        for i in range(nk):
            kx = -np.sqrt(3) * G / 2 + (i) * np.sqrt(3) * G / (nk - 1)  # 
            f.write(f"{kx * 0.1/ (2 * np.pi / a)} {band[0, i, 0]} {band[1, i, 0]} \n")

    with open("bandstr2 cho kp.txt", "w") as f:
        for j in range(nk):
            for i in range(nk):
                f.write(f"{akx[i, j] / (2 * np.pi / a)} {aky[i, j] / (2 * np.pi / a)} "
                        f"{band[0, i, j]} {band[1, i, j]} \n")
        
    """    with open("bandstr_u cho kp.txt", "w") as f:
            for j in range(nk):
                for i in range(nk):
                    f.write(f"{akx[i, j] / (2 * np.pi / a)} {aky[i, j] / (2 * np.pi / a)} "
                            f"{bandu[0, i, j]} {bandu[1, i, j]} \n")

        with open("bandstr_d cho kp.txt", "w") as f:
            for j in range(nk):
                for i in range(nk):
                    f.write(f"{akx[i, j] / (2 * np.pi / a)} {aky[i, j] / (2 * np.pi / a)} "
                            f"{bandd[0, i, j]} {bandd[1, i, j]} \n")"""

    with open("momentum cho kp.txt", "w") as f:
        f.write(f"#{'kx':^12} {'ky':^12} {'|pp|':^12} {'|pm|':^12} {'px':^25} {'py':^25} {'|p|':^25} \n")
        for j in range(nk):
            for i in range(nk):
                px, py = pmx[0, 1, i, j], pmy[0, 1, i, j]
                p = np.sqrt((abs(px))**2 + (abs(py))**2)
                pp = px + 1j * py
                pm = px - 1j * py
                kx = akx[i, j] / ( 2 * np.pi / a)
                ky = aky[i, j] / ( 2 * np.pi / a)

                f.write(f"{kx:.6e} {ky:.6e} {abs(pp):.6e} {abs(pm):.6e} "
                        f"({px.real:.6e},{px.imag:.6e}) ({py.real:.6e},{py.imag:.6e}) {p:.6e} \n")
                """f.write(f"{kx:.6e} {ky:.6e} {abs(pp):.6e} {abs(pm):.6e} "
                        f"{abs(px):.6e} {abs(py):.6e}  {p:.6e} \n")"""
            f.write("\n") 

 
def main():
    nk = 99
    material = 1  # MoS2

    a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, G = tb_parameters(material)

    akx, aky = kgrid(nk, G, a)

    band, pmx, pmy = bandstructure(nk, akx, aky, a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, material)

    save_data(nk, G, akx, aky, band, pmx, pmy, a)

    print("Hoàn tất tính toán dải năng lượng!")

if __name__ == "__main__":
    main()
