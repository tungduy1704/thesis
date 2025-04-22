
from scipy.linalg import eigh
import numpy as np
# Hằng số vật lý
hb = 0.658229  # Planck constant [eV.fs]
m0 = 5.6770736  # Free electron mass [eV.(fs/nm)^2]
pi = np.pi


def tb_parameters(material):
    atab = np.array([0.3190, 0.3191, 0.3326, 0.3325, 0.3557, 0.3560])
    e1tab = np.array([0.683,0.717,0.684,0.728,0.588,0.697])
    e2tab = np.array([1.707,1.916,1.546,1.655,1.303,1.380])    
    t0tab = np.array([-0.146,-0.152,-0.146,-0.146,-0.226,-0.109])        
    t1tab = np.array([-0.114,-0.097,-0.130,-0.124,-0.234,-0.164])    
    t2tab = np.array([0.506,0.590,0.432,0.507,0.036,0.368])
    t11tab = np.array([0.085,0.047,0.144,0.117,0.400,0.204])
    t12tab = np.array([0.162,0.178,0.117,0.127,0.098,0.093])
    t22tab = np.array([0.073,0.016,0.075,0.015,0.017,0.038])
    r0tab = np.array([0.060,0.069,0.039,0.036,0.003,-0.015])
    r1tab = np.array([-0.236,-0.261,-0.209,-0.234,-0.025,-0.209])
    r2tab = np.array([0.067,0.107,0.069,0.107,-0.169,0.107])
    r11tab = np.array([0.016,-0.003,0.052,0.044,0.082,0.115])
    r12tab = np.array([0.087,0.109,0.060,0.075,0.051,0.009])
    u0tab = np.array([-0.038,-0.054,-0.042,-0.061,0.057,-0.066])
    u1tab = np.array([0.046,0.045,0.036,0.032,0.103,0.011])
    u2tab = np.array([0.001,0.002,0.008,0.007,0.187,-0.013])
    u11tab = np.array([0.266,0.325,0.272,0.329,-0.045,0.312])
    u12tab = np.array([-0.176,-0.206,-0.172,-0.202,-0.141,-0.177])
    u22tab = np.array([-0.150,-0.163,-0.150,-0.164,0.087,-0.132])
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
    r0 = r0tab[material - 1]
    r1 = r1tab[material - 1]
    r2 = r2tab[material - 1]
    r11 = r11tab[material - 1]
    r12 = r12tab[material - 1]
    u0 = u0tab[material - 1]
    u1 = u1tab[material - 1]
    u2 = u2tab[material - 1]
    u11 = u11tab[material - 1]
    u12 = u12tab[material - 1]
    u22 = u22tab[material - 1]
    lambda_ = lambdatab[material - 1]
    G = 4 * pi / (np.sqrt(3) * a)
    
    return a, e1, e2, t0, t1, t2, t11, t12, t22, r0, r1, r2, r11, r12, u0, u1, u2, u11, u12, u22, lambda_, G


def kgrid(nk, G, a):
    """ Tạo lưới k-space theo hệ Cartesian. """
    dk = (4 * pi / a) / (nk - 1)
    akx = np.zeros((nk, nk))
    aky = np.zeros((nk, nk))

    for j in range(nk):
        for i in range(nk):
            akx[i, j] = - 2 * pi / a + i * dk
            aky[i, j] = 0#- 2 * pi / a + j * dk

    return akx, aky

def tb_ham_autograd(k, params):
    """ Hàm Hamiltonian với đầu vào là vector k = [kx, ky] để dùng với autograd """
    kx, ky = k
    return tb_ham(kx, ky, params)[0]  # Lấy ma trận Hamiltonian H(k)

def tb_ham(kx, ky, params):
    a, e1, e2, t0, t1, t2, t11, t12, t22, r0, r1, r2, r11, r12, u0, u1, u2, u11, u12, u22, lambda_, G = params
    
    alpha = kx * a / 2
    beta = np.sqrt(3) * ky * a / 2
    c2a = np.cos(2 * alpha)
    s2a = np.sin(2 * alpha)
    ca = np.cos(alpha)
    cb = np.cos(beta)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    c3a = np.cos(3 * alpha)
    s3a = np.sin(3 * alpha)
    c2b = np.cos(2 * beta)
    c4a = np.cos(4 * alpha)
    s4a = np.sin(4 * alpha)
    s2b = np.sin(2 * beta)
    
    SR3 = np.sqrt(3)
    
    V0 = (e1 + 2.0 * t0 * (2.0 * ca * cb + c2a)
          + 2.0 * r0 * (2.0 * c3a * cb + c2b)
          + 2.0 * u0 * (2.0 * c2a * c2b + c4a))
    
    V1 = (-2.0 * SR3 * t2 * sa * sb
          + 2.0 * (r1 + r2) * s3a * sb
          - 2.0 * SR3 * u2 * s2a * s2b
          + 1j * (2.0 * t1 * sa * (2.0 * ca + cb)
                 + 2.0 * (r1 - r2) * s3a * cb
                 + 2.0 * u1 * s2a * (2.0 * c2a + c2b)))
    
    V2 = (2.0 * t2 * (c2a - ca * cb)
          - 2.0 * (r1 + r2) * (c3a * cb - c2b) / SR3
          + 2.0 * u2 * (c4a - c2a * c2b)
          + 1j * (2.0 * SR3 * t1 * ca * sb
                 + 2.0 * (r1 - r2) * sb * (c3a + 2.0 * cb) / SR3
                 + 2.0 * SR3 * u1 * c2a * s2b))
    
    V11 = (e2 + (t11 + 3.0 * t22) * ca * cb
           + 2.0 * t11 * c2a + 4.0 * r11 * c3a * cb
           + 2.0 * (r11 + SR3 * r12) * c2b
           + (u11 + 3.0 * u22) * c2a * c2b + 2.0 * u11 * c4a)
    
    V12 = (SR3 * (t22 - t11) * sa * sb
           + 4.0 * r12 * s3a * sb
           + SR3 * (u22 - u11) * s2a * s2b
           + 1j * (4.0 * t12 * sa * (ca - cb)
                  + 4.0 * u12 * s2a * (c2a - c2b)))
    
    V22 = (e2 + (3.0 * t11 + t22) * ca * cb
           + 2.0 * t22 * c2a + 2.0 * r11 * (2.0 * c3a * cb + c2b)
           + 2.0 * r12 * (4.0 * c3a * cb - c2b) / SR3
           + (3.0 * u11 + u22) * c2a * c2b + 2.0 * u22 * c4a)    
    ham = np.array([[V0, V1, V2],
                    [np.conjugate(V1), V11, V12],
                    [np.conjugate(V2), np.conjugate(V12), V22]
                    ], dtype=complex)
    
    dV0_kx = (2.0 * t0 * (-a * cb * sa  - a * s2a)
         - 6.0 * a * r0 * cb * s3a
         + 2.0 * u0 * (-2.0 * a * c2b * s2a - 2.0 * a * s4a))
    
    dV1_kx = (1j * (3 * a * (r1 - r2) * c3a * cb
                + a * t1 * ca * (2 * ca + cb)
                + 2 * a * u1 * c2a * (2 * c2a + c2b)
                - 2 * a * t1 * sa**2
                - 4 * a * u1 * s2a**2)
          - SR3 * a * t2 * ca * sb
          + 3 * a * (r1 + r2) * c3a * sb
          - SR3 * 2 * a * u2 * c2a * s2b)
    
    dV2_kx = (2.0 * t2 * (0.5 * a * cb * sa - a * s2a)
         + SR3 * a * (r1 + r2) * cb * s3a
         + 2.0 * u2 * (a * c2b * s2a - 2.0 * a * s4a)
         + 1j * (-SR3 * a * t1 * sa * sb
                 - SR3 * a * (r1 - r2) * s3a * sb
                 - 2 * SR3 * a * u1 * s2a * s2b))
    
    dV11_kx = (-0.5 * a * (t11 + 3.0 * t22) * cb * sa
           - 2.0 * a * t11 * s2a
           - a * (u11 + 3.0 * u22) * c2b * s2a
           - 6.0 * a * r11 * cb * s3a
           - 4.0 * a * u11 * s4a)
    
    dV12_kx = (1j * (2.0 * a * t12 * ca * (ca - cb)
                 + 4.0 * a * u12 * c2a * (c2a - c2b)
                 - 2.0 * a * t12 * sa**2
                 - 4.0 * a * u12 * s2a**2)
           + 0.5 * SR3 * a * (-t11 + t22) * ca * sb
           + 6.0 * a * r12 * c3a * sb
           + SR3 * a * (-u11 + u22) * c2a * s2b)
    
    dV22_kx = (-0.5 * a * (3.0 * t11 + t22) * cb * sa
           - 2.0 * a * t22 * s2a
           - a * (3.0 * u11 + u22) * c2b * s2a
           - 6.0 * a * r11 * cb * s3a
           - 4.0 * SR3 * a * r12 * cb * s3a
           - 4.0 * a * u22 * s4a)

    dV0_ky = (-2.0 * SR3 * a * t0 * ca * sb
          - 4.0 * SR3 * a * u0 * c2a * s2b
          + 2.0 * r0 * (-SR3 * a * c3a * sb - SR3 * a * s2b))
    
    dV1_ky = (-3.0 * a * t2 * cb * sa
          - 6.0 * a * u2 * c2b * s2a
          + SR3 * a * (r1 + r2) * cb * s3a
          + 1j * (-SR3 * a * t1 * sa * sb
                 - SR3 * a * (r1 - r2) * s3a * sb
                 - 2.0 * SR3 * a * u1 * s2a * s2b))
    
    dV2_ky = (SR3 * a * t2 * ca * sb
          + 1j * (3.0 * a * t1 * ca * cb
                 + a * (r1 - r2) * cb * (c3a + 2.0 * cb)
                 + 6.0 * a * u1 * c2a * c2b
                 - 2.0 * a * (r1 - r2) * sb**2) 
          + 2.0 * SR3 * a * u2 * c2a * s2b
          - 1.1547 * (r1 + r2) * (-0.5 * SR3 * a * c3a * sb + SR3 * a * s2b))
    
    dV11_ky = (-0.5 * SR3 * a * (t11 + 3.0 * t22) * ca * sb
           - 2.0 * SR3 * a * r11 * c3a * sb
           - 2.0 * SR3 * a * (r11 + SR3 * r12) * s2b
           - SR3 * a * (u11 + 3.0 * u22) * c2a * s2b)
    
    dV12_ky = ((3/2) * a * (-t11 + t22) * cb * sa
           + 3.0 * a * (-u11 + u22) * c2b * s2a
           + 2.0 * SR3 * a * r12 * cb * s3a
           + 1j * (2.0 * SR3 * a * t12 * sa * sb
                  + 4.0 * SR3 * a * u12 * s2a * s2b))
    
    dV22_ky = (-0.5 * SR3 * a * (3.0 * t11 + t22) * ca * sb
           - SR3 * a * (3.0 * u11 + u22) * c2a * s2b
           + 2.0 * r11 * (-SR3 * a * c3a * sb - SR3 * a * s2b)
           + 1.1547 * r12 * (-2.0 * SR3 * a * c3a * sb + SR3 * a * s2b))
    # Gọi đạo hàm tự động bằng autograd
    dhkx = np.array([[dV0_kx, dV1_kx, dV2_kx],
                     [np.conjugate(dV1_kx), dV11_kx, dV12_kx],
                     [np.conjugate(dV2_kx), np.conjugate(dV12_kx), dV22_kx]])
    
    dhky = np.array([[dV0_ky, dV1_ky, dV2_ky],
                     [np.conjugate(dV1_ky), dV11_ky, dV12_ky],
                     [np.conjugate(dV2_ky), np.conjugate(dV12_ky), dV22_ky]])
        
    Lz = np.array([[0, 0, 0], [0, 0, 2j], [0, -2j, 0]], dtype=np.complex128)
    
    hamu = ham + lambda_ / 2 * Lz
    hamd = ham - lambda_ / 2 * Lz            

    return ham, hamu, hamd, dhkx, dhky

# Tạo hàm gradient của Hamiltonian theo k = [kx, ky]



def initialize_band_data(nk):
    """ Khởi tạo các mảng chứa dữ liệu dải năng lượng và phần tử động lượng. """
    bandu = np.zeros((3, nk, nk))
    bandd = np.zeros((3, nk, nk))
    band = np.zeros((3, nk, nk))
    pmx = np.zeros((3, 3, nk, nk), dtype=complex)
    pmy = np.zeros((3, 3, nk, nk), dtype=complex)
    vecs_save = np.zeros((3, 3, nk, nk), dtype=complex)
    return bandu, bandd, band, pmx, pmy, vecs_save

def initialize_matrices():
    """ Khởi tạo ma trận Hamiltonian và mảng làm việc cho diagonalization. """
    ham = np.zeros((3, 3), dtype=complex)
    hamu = np.zeros((3, 3), dtype=complex)
    hamd = np.zeros((3, 3), dtype=complex)
    dhkx = np.zeros((3, 3), dtype=complex)
    dhky = np.zeros((3, 3), dtype=complex)
    return ham, hamu, hamd, dhkx, dhky

def bandstructure(nk, akx, aky, a, e1, e2, t0, t1, t2, t11, t12, t22, r0, r1, r2, r11, r12, u0, u1, u2, u11, u12, u22, lambda_, G):
    bandu, bandd, band, pmx, pmy, vecs_save = initialize_band_data(nk)

    for j in range(nk):
        for i in range(nk):
            kx, ky = akx[i, j], aky[i, j]

            # Khởi tạo ma trận Hamiltonian
            ham, hamu, hamd, dhkx, dhky = initialize_matrices()

            # Lấy Hamiltonian tổng, spin-up, spin-down
            ham, hamu, hamd, dhkx, dhky = tb_ham(kx, ky, (a, e1, e2, t0, t1, t2, t11, t12, t22, r0, r1, r2, r11, r12, u0, u1, u2, u11, u12, u22, lambda_, G))


            # Giải trị riêng Hamiltonian tổng
            vals, vecs = eigh(ham)
            band[:, i, j] = vals
            vecs_save[:, :, i, j] = vecs
            # Tính phần tử động lượng 
            for jb in range(3):  # Chỉ số dải λ'
                for ib in range(3):  # Chỉ số dải λ
                    sum1 = 0. + 0j   # Tổng phần tử động lượng theo kx
                    sum2 = 0. + 0j   # Tổng phần tử động lượng theo ky
                    for jjb in range(3): # Chỉ số trạng thái nguyên tử j'
                        for iib in range(3): # Chỉ số trạng thái nguyên tử j
                            sum1 += np.conjugate(vecs[iib, ib]) * dhkx[iib, jjb] * vecs[jjb, jb]
                            sum2 += np.conjugate(vecs[iib, ib]) * dhky[iib, jjb] * vecs[jjb, jb]
                    pmx[ib, jb, i, j] = sum1 * m0 / hb
                    pmy[ib, jb, i, j] = sum2 * m0 / hb

            # Giải trị riêng cho spin-up & spin-down
            bandu[:, i, j] = eigh(hamu, eigvals_only=True) # up

            bandd[:, i, j] = eigh(hamd, eigvals_only=True) # down

    return bandu, bandd, band, pmx, pmy, vecs_save

def save_data(nk, G, akx, aky, bandu, bandd, band, pmx, pmy, a, vecs_save):
    with open("vecs_tb 3NN.txt", "w") as f:
        for j in range(nk):
            for i in range(nk):
                kx = akx[i, j] / (2 * np.pi / a)
                ky = aky[i, j] / (2 * np.pi / a)
                f.write(f"{kx:.6e} {ky:.6e} \n")
                for band_idx in range(3):
                    for component in vecs_save[:, band_idx, i, j]:
                        f.write(f"{component.real:.6e} {component.imag:.6e} \n")
                f.write("\n")
    with open("bandstr1 3NN.txt", "w") as f:
        for i in range(nk):
            kx = -np.sqrt(3) * G / 2 + (i) * np.sqrt(3) * G / (nk - 1)  # 
            f.write(f"{kx / (2 * np.pi / a)} {band[0, i, 0]} {band[1, i, 0]} {band[2, i, 0]}\n")

    with open("bandstr2 3NN.txt", "w") as f:
        f.write("#kx ky band1 band2 band3 \n")
        for j in range(nk):
            for i in range(nk):
                f.write(f"{akx[i, j] / (2 * np.pi / a)} {aky[i, j] / (2 * np.pi / a)} "
                        f"{band[0, i, j]} {band[1, i, j]} {band[2, i, j]}\n")
            f.write("\n")

    with open("bandstr_u 3NN.txt", "w") as f:
        for i in range(nk):
                f.write(f"{akx[i, 0] / (2 * np.pi / a)} "
                        f"{bandu[0, i, 0]} {bandu[1, i, 0]} {bandu[2, i, 0]}\n")
    
    with open("bandstr_d 3NN.txt", "w") as f:
        for i in range(nk):
            f.write(f"{akx[i, 0] / (2 * np.pi / a)}  "
                    f"{bandd[0, i, 0]} {bandd[1, i, 0]} {bandd[2, i, 0]}\n")
    with open("momentum cho 3NN.txt", "w") as f:
        f.write(f"#{'kx':^12} {'ky':^12} {'|pp|':^12} {'|pm|':^12} {'px':^25} {'py':^25} {'|p|':^25} \n")
        
        for j in range(nk):
            for i in range(nk):
                px, py = pmx[0, 1, i, j], pmy[0, 1, i, j]
                p = np.sqrt((abs(px))**2 + (abs(py))**2)
                pp = px + 1j * py
                pm = px - 1j * py
                kx = akx[i, j] / (2 * np.pi / a)
                ky = aky[i, j] / (2 * np.pi / a)

                # 🔥 Ghi số phức trong dấu ngoặc (real, imag)
                f.write(f"{kx:.6e} {ky:.6e} {abs(pp):.6e} {abs(pm):.6e} "
                        f"({px.real:.6e},{px.imag:.6e}) ({py.real:.6e},{py.imag:.6e}) {p:.6e} \n")
                """f.write(f"{kx:.6e} {ky:.6e} {abs(pp):.6e} {abs(pm):.6e} "
                        f"{abs(px):.6e} {abs(py):.6e}  {p:.6e} \n")"""
            f.write("\n")  # Xuống dòng giữa các block


    with open("momentum 0_2 3NN.txt", "w") as f:
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

    # Lấy tham số vật liệu
    a, e1, e2, t0, t1, t2, t11, t12, t22, r0, r1, r2, r11, r12, u0, u1, u2, u11, u12, u22, lambda_, G = tb_parameters(material)

    # Tạo lưới k-space (dùng Cartesian grid)
    akx, aky = kgrid(nk, G, a)

    # Tính band structure
    bandu, bandd, band, pmx, pmy, vecs_save = bandstructure(nk, akx, aky, a, e1, e2, t0, t1, t2, t11, t12, t22, r0, r1, r2, r11, r12, u0, u1, u2, u11, u12, u22, lambda_, material)

    # Xuất dữ liệu
    save_data(nk, G, akx, aky, bandu, bandd, band, pmx, pmy, a, vecs_save)

    print("Hoàn tất tính toán dải năng lượng!")

if __name__ == "__main__":
    main()
