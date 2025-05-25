import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm

hb = 0.658229  # eV·fs
m0 = 5.6770736 / 100  # eV·fs²/amstrong²
pi = np.pi

def nntb_parameters(material):
    atab = np.array([3.190, 3.191, 3.326, 3.325, 3.557, 3.560])
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

def thirdnn_tb_parameters(material):
    atab = np.array([3.190, 3.191, 3.326, 3.325, 3.557, 3.560])
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

def tb_rhombusgrid(nk, G):
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


def tb_kgrid(nk, G, a):
    dk = (4 * pi / a) / (nk - 1)
    akx = np.zeros((nk, nk))
    aky = np.zeros((nk, nk))

    for j in range(nk):
        for i in range(nk):
            akx[i, j] = - 2 * pi / a + i * dk
            aky[i, j] = - 2 * pi / a + j * dk

    return akx, aky

def kp_kgrid(nk, G, a):
    dk = (4 * 0.1 *  pi / a) / (nk - 1)
    akx = np.zeros((nk, nk))
    aky = np.zeros((nk, nk))

    for j in range(nk):
        for i in range(nk):
            akx[i, j] = -0.1 * 2 * pi / a + i * dk
            aky[i, j] = -0.1 * 2 * pi / a + j * dk

    return akx, aky

def nntb_ham(kx, ky, params):
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

def thirdnn_tb_ham(kx, ky, params):
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

def kp_ham(kx, ky):
    Delta = 1.663 
    a = 3.190

    # 1st-order k.p parameter
    t = 1.105

    # 2nd-order k.p parameters
    #t = 1.059
    gamma1 = 0.055
    gamma2 = 0.077
    gamma3 = -0.123

    # 3rd-order k.p parameters
    #t = 1.003
    #gamma1 = 0.196 
    #gamma2 = -0.065 
    #gamma3 = -0.248 
    gamma4 = 0.163 
    gamma5 = -0.094 
    gamma6 = -0.232
    
    ham1 = np.array([[Delta, a * t * (kx - 1j * ky)],
                     [a * t * (kx + 1j * ky), 0]])

    ham2 = (a**2) * np.array([[gamma1 * (kx**2 + ky**2), gamma3 * (kx + 1j * ky)**2],
                     [gamma3 * (kx - 1j * ky)**2, gamma2 * (kx**2 + ky**2)]])

    ham3 = (a**3) * np.array([[gamma4 * kx * (kx**2 - 3*ky**2) , gamma6 * (kx**2 + ky**2) * (kx - 1j*ky)],
                            [gamma6 * (kx**2 + ky**2) * (kx + 1j*ky), gamma5 * kx * (kx**2 - 3*ky**2)]])
                    
    ham = ham1 + ham2 

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

    dhkx = dham1kx + dham2kx 
    dhky = dham1ky + dham2ky 
                
    return ham, dhkx, dhky

def tb_initialize_band_data(nk):
    bandu = np.zeros((3, nk, nk))
    bandd = np.zeros((3, nk, nk))
    band = np.zeros((3, nk, nk))
    pmx = np.zeros((3, 3, nk, nk), dtype=complex)
    pmy = np.zeros((3, 3, nk, nk), dtype=complex)

    return bandu, bandd, band, pmx, pmy

def tb_initialize_matrices():
    ham = np.zeros((3, 3), dtype=complex)
    hamu = np.zeros((3, 3), dtype=complex)
    hamd = np.zeros((3, 3), dtype=complex)
    dhkx = np.zeros((3, 3), dtype=complex)
    dhky = np.zeros((3, 3), dtype=complex)

    return ham, hamu, hamd, dhkx, dhky

def kp_initialize_band_data(nk):
    band  = np.zeros((2, nk, nk))
    pmx   = np.zeros((2, 2, nk, nk), dtype=complex)
    pmy   = np.zeros((2, 2, nk, nk), dtype=complex)
    return band, pmx, pmy

def kp_initialize_matrices():
    ham  = np.zeros((2, 2), dtype=complex)
    dhkx = np.zeros((2, 2), dtype=complex)
    dhky = np.zeros((2, 2), dtype=complex)
    return ham, dhkx, dhky

def nntb_bandstructure(nk, akx, aky, a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, G):

    bandu, bandd, band, pmx, pmy = tb_initialize_band_data(nk)

    for j in tqdm(range(nk)):
        for i in range(nk):
            kx, ky = akx[i, j], aky[i, j]

            ham, hamu, hamd, dhkx, dhky = tb_initialize_matrices()
            ham, hamu, hamd, dhkx, dhky = nntb_ham(kx, ky, (a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, G))

            vals, vecs = eigh(ham)
            band[:, i, j] = vals

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

    return bandu, bandd, band, pmx, pmy

def thirdnn_bandstructure(nk, akx, aky, a, e1, e2, t0, t1, t2, t11, t12, t22, r0, r1, r2, r11, r12, u0, u1, u2, u11, u12, u22, lambda_, G):
    bandu, bandd, band, pmx, pmy = tb_initialize_band_data(nk)

    for j in tqdm(range(nk)):
        for i in range(nk):
            kx, ky = akx[i, j], aky[i, j]

            ham, hamu, hamd, dhkx, dhky = tb_initialize_matrices()
            ham, hamu, hamd, dhkx, dhky = thirdnn_tb_ham(kx, ky, (a, e1, e2, t0, t1, t2, t11, t12, t22, r0, r1, r2, r11, r12, u0, u1, u2, u11, u12, u22, lambda_, G))

            vals, vecs = eigh(ham)
            band[:, i, j] = vals
           
            for jb in range(3):
                for ib in range(3): 
                    sum1 = 0. + 0j   
                    sum2 = 0. + 0j   
                    for jjb in range(3): 
                        for iib in range(3): 
                            sum1 += np.conjugate(vecs[iib, ib]) * dhkx[iib, jjb] * vecs[jjb, jb]
                            sum2 += np.conjugate(vecs[iib, ib]) * dhky[iib, jjb] * vecs[jjb, jb]
                    pmx[ib, jb, i, j] = sum1 * m0 / hb
                    pmy[ib, jb, i, j] = sum2 * m0 / hb

            bandu[:, i, j] = eigh(hamu, eigvals_only=True) 

            bandd[:, i, j] = eigh(hamd, eigvals_only=True) 

    return bandu, bandd, band, pmx, pmy

def kp_bandstructure(nk, akx, aky):
    band, pmx, pmy = kp_initialize_band_data(nk)

    for j in tqdm(range(nk)):
        for i in range(nk):
            kx, ky = akx[i, j], aky[i, j]

            ham, dhkx, dhky = kp_initialize_matrices()
            ham, dhkx, dhky = kp_ham(kx, ky)

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


def nntb_save_data(nk, G, akx, aky, bandu, bandd, band, pmx, pmy, a):
    with open("nntb_bandstr.txt", "w") as f:
        for i in range(nk):
            kx = -np.sqrt(3) * G / 2 + (i) * np.sqrt(3) * G / (nk - 1)  # 
            f.write(f"{kx / (2 * np.pi / a)} {band[0, i, 0]} {band[1, i, 0]} {band[2, i, 0]}\n")

    with open("nntb_bandstr_up.txt", "w") as f:
        for i in range(nk):
                f.write(f"{akx[i, 0] / (2 * np.pi / a)} "
                        f"{bandu[0, i, 0]} {bandu[1, i, 0]} {bandu[2, i, 0]}\n")
    
    with open("nntb_bandstr_down.txt", "w") as f:
        for i in range(nk):
            f.write(f"{akx[i, 0] / (2 * np.pi / a)}  "
                    f"{bandd[0, i, 0]} {bandd[1, i, 0]} {bandd[2, i, 0]}\n")
    
    with open("nntb_momentum.txt", "w") as f:
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

def thirdnn_save_data(nk, G, akx, aky, bandu, bandd, band, pmx, pmy, a):
    with open("thirdnn_bandstr.txt", "w") as f:
        for i in range(nk):
            kx = -np.sqrt(3) * G / 2 + (i) * np.sqrt(3) * G / (nk - 1)  # 
            f.write(f"{kx / (2 * np.pi / a)} {band[0, i, 0]} {band[1, i, 0]} {band[2, i, 0]}\n")

    with open("thirdnn_bandstr_up.txt", "w") as f:
        for i in range(nk):
                f.write(f"{akx[i, 0] / (2 * np.pi / a)} "
                        f"{bandu[0, i, 0]} {bandu[1, i, 0]} {bandu[2, i, 0]}\n")
    
    with open("thirdnn_bandstr_down.txt", "w") as f:
        for i in range(nk):
            f.write(f"{akx[i, 0] / (2 * np.pi / a)}  "
                    f"{bandd[0, i, 0]} {bandd[1, i, 0]} {bandd[2, i, 0]}\n")
            
    with open("thirdnn_momentum.txt", "w") as f:
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

def kp_save_data(nk, G, akx, aky, band, pmx, pmy, a):
  
    with open("kp_bandstr.txt", "w") as f:
        for i in range(nk):
            kx = -np.sqrt(3) * G / 2 + (i) * np.sqrt(3) * G / (nk - 1)  # 
            f.write(f"{kx*0.1/ (2 * np.pi / a)} {band[0, i, 0]} {band[1, i, 0]} \n")


    """ with open("kp_bandstr_up.txt", "w") as f:
            for j in range(nk):
                for i in range(nk):
                    f.write(f"{akx[i, j] / (2 * np.pi / a)} {aky[i, j] / (2 * np.pi / a)} "
                            f"{bandu[0, i, j]} {bandu[1, i, j]} \n")

        with open("kp_bandstr_down.txt", "w") as f:
            for j in range(nk):
                for i in range(nk):
                    f.write(f"{akx[i, j] / (2 * np.pi / a)} {aky[i, j] / (2 * np.pi / a)} "
                            f"{bandd[0, i, j]} {bandd[1, i, j]} \n")"""

    with open("kp_momentum.txt", "w") as f:
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
                
            f.write("\n") 


def main():
    nk = 199
    material = 1  # MoS2

    a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, G = nntb_parameters(material)
    tnn_a, tnn_e1, tnn_e2, tnn_t0, tnn_t1, tnn_t2, tnn_t11, tnn_t12, tnn_t22, tnn_r0, tnn_r1, tnn_r2, tnn_r11, tnn_r12, tnn_u0, tnn_u1, tnn_u2, tnn_u11, tnn_u12, tnn_u22, tnn_lambda_, tnn_G = thirdnn_tb_parameters(material)
    
    tb_akx, tb_aky = tb_kgrid(nk, G, a)
    kp_akx, kp_aky = kp_kgrid(nk, G, a)

    nn_bandu, nn_bandd, nn_band, nn_pmx, nn_pmy = nntb_bandstructure(nk, tb_akx, tb_aky, a, e1, e2, t0, t1, t2, t11, t12, t22, lambda_, material)
    tnn_bandu, tnn_bandd, tnn_band, tnn_pmx, tnn_pmy = thirdnn_bandstructure(nk, tb_akx, tb_aky, tnn_a, tnn_e1, tnn_e2, tnn_t0, tnn_t1, tnn_t2, tnn_t11, tnn_t12, tnn_t22, tnn_r0, tnn_r1, tnn_r2, tnn_r11, tnn_r12, tnn_u0, tnn_u1, tnn_u2, tnn_u11, tnn_u12, tnn_u22, tnn_lambda_, material)
    kp_band, kp_pmx, kp_pmy = kp_bandstructure(nk, kp_akx, kp_aky)

    nntb_save_data(nk, G, tb_akx, tb_aky, nn_bandu, nn_bandd, nn_band, nn_pmx, nn_pmy, a)
    thirdnn_save_data(nk, G, tb_akx, tb_aky, tnn_bandu, tnn_bandd, tnn_band, tnn_pmx, tnn_pmy, a)
    kp_save_data(nk, G, kp_akx, kp_aky, kp_band, kp_pmx, kp_pmy, a)

    print("SUCCESSFUL CALCULATION! LET'S PLOT THEM!")
    
if __name__ == "__main__":
    main()
