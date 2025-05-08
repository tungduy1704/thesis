import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

hb = 0.658229  # eV·fs
m0 = 5.6770736 / 100
me_over_hbar = m0 / hb

a1 = np.array([3.190315, 0.0, 0.0])
a2 = np.array([-1.595158, 2.762894, 0.0])
a3 = np.array([0.0, 0.0, 17.439502])

def read_hr(filename="wannier90_hr.dat"):
    with open(filename, 'r') as f:
        lines = f.readlines()

    num_wann = int(lines[1])
    num_R = int(lines[2])
    data_lines = lines[3 + num_R:]

    H_dict = {}
    R_set = set()

    for line in data_lines:
        parts = line.strip().split()
        R = tuple(map(int, parts[0:3]))
        m = int(parts[3]) - 1
        n = int(parts[4]) - 1
        val = float(parts[5]) + 1j * float(parts[6])
        H_dict[(R, m, n)] = val
        R_set.add(R)

    return num_wann, sorted(R_set), H_dict

def prepare_H_tensor_and_Rvec(num_wann, R_list, H_dict):
    H_tensor = np.zeros((len(R_list), num_wann, num_wann), dtype=complex)
    Rvec_array = np.zeros((len(R_list), 3))

    for idx, R in enumerate(R_list):
        Rvec = R[0]*a1 + R[1]*a2
        Rvec_array[idx] = Rvec
        for m in range(num_wann):
            for n in range(num_wann):
                key = (R, m, n)
                if key in H_dict:
                    H_tensor[idx, m, n] = H_dict[key]
    return H_tensor, Rvec_array

def momentum_operator_vectorized(kvec, H_tensor, Rvec_array):
    phase = np.exp(1j * Rvec_array @ kvec)
    weighted_phase_x = (1j * Rvec_array[:, 0] * phase)[:, np.newaxis, np.newaxis]
    weighted_phase_y = (1j * Rvec_array[:, 1] * phase)[:, np.newaxis, np.newaxis]

    px = np.sum(H_tensor * weighted_phase_x, axis=0)
    py = np.sum(H_tensor * weighted_phase_y, axis=0)

    return me_over_hbar * px, me_over_hbar * py

def construct_Hk(kvec, num_wann, R_list, H_dict):
    Hk = np.zeros((num_wann, num_wann), dtype=complex)
    for R in R_list:
        Rvec = R[0]*a1 + R[1]*a2
        phase = np.exp(1j * np.dot(kvec, Rvec))
        for m in range(num_wann):
            for n in range(num_wann):
                key = (R, m, n)
                if key in H_dict:
                    Hk[m, n] += H_dict[key] * phase
    return Hk

def extract_pcv(filename="wannier90_hr.dat", nk=100, out_csv="p_cv_map.csv", out_eigen_csv="eigenvalues_map.csv"):
    num_wann, R_list, H_dict = read_hr(filename)
    H_tensor, Rvec_array = prepare_H_tensor_and_Rvec(num_wann, R_list, H_dict)

    a_lat = 3.190
    kmax = 2 * np.pi / a_lat
    kx_vals = np.linspace(-kmax, kmax, nk)
    ky_vals = np.linspace(-kmax, kmax, nk)
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals)
    kx_list = kx_grid.flatten()
    ky_list = ky_grid.flatten()

    data = []
    eigen_data = []

    for kx, ky in tqdm(zip(kx_list, ky_list), total=len(kx_list), desc="Đang quét k-lưới"):
        kvec = np.array([kx, ky, 0.0])
        Hk = construct_Hk(kvec, num_wann, R_list, H_dict)
        evals, U = np.linalg.eigh(Hk)
        px, py = momentum_operator_vectorized(kvec, H_tensor, Rvec_array)
        
        val_idx = 6  # band 6
        cond_idx = 7  # band 7
        #print(evals[val_idx], evals[cond_idx])

        p_bloch_x = U.conj().T @ px @ U
        p_bloch_y = U.conj().T @ py @ U

        p_plus = p_bloch_x[cond_idx, val_idx] + 1j * p_bloch_y[cond_idx, val_idx]
        p_minus = p_bloch_x[cond_idx, val_idx] - 1j * p_bloch_y[cond_idx, val_idx]
        p_abs = np.sqrt(np.abs(p_bloch_x[cond_idx, val_idx])**2 + np.abs(p_bloch_y[cond_idx, val_idx])**2)

        data.append([
            kx / (2*np.pi/a_lat), ky / (2*np.pi/a_lat),
            p_bloch_x[cond_idx, val_idx], p_bloch_y[cond_idx, val_idx],
            abs(p_plus), abs(p_minus), p_abs
        ])

        eigen_row = [kx / (2*np.pi/a_lat), ky / (2*np.pi/a_lat)] + list(np.real(evals))
        eigen_data.append(eigen_row)

    df = pd.DataFrame(data, columns=["kx", "ky", "px", "py", "p_plus", "p_minus", "p_abs"])
    df.to_csv(out_csv, index=False)
    print(f"✅ Đã ghi file: {out_csv}")

    eigen_cols = ["kx", "ky"] + [f"band_{i+1}" for i in range(num_wann)]
    df_eigen = pd.DataFrame(eigen_data, columns=eigen_cols)
    df_eigen.to_csv(out_eigen_csv, index=False)
    print(f"✅ Đã ghi file eigenvalues: {out_eigen_csv}")


extract_pcv(filename="mos2_hr_294.dat", nk=199, out_csv="p_cv_map294.csv", out_eigen_csv="eigenvalues_map294.csv")
df1 = pd.read_csv("eigenvalues_map294.csv")
df1.to_csv("eigenvalues_map294.txt", sep=" ", index=False)
