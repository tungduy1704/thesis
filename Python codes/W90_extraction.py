import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

hb = 0.658229  # eV·fs
m0 = 5.6770736 / 100 # eV·fs²/amstrong²
me_over_hbar = m0 / hb

a1 = np.array([3.190315, 0.0, 0.0])
a2 = np.array([-1.595158, 2.762894, 0.0])
a3 = np.array([0.0, 0.0, 17.439502])

def read_hr(filename="wannier90_hr.dat"):
    with open(filename, 'r') as f:
        lines = f.readlines()
    num_wann = int(lines[1])
    num_R = int(lines[2])
    count = 0
    i = 3  
    while count < num_R:
        count += len(lines[i].strip().split())
        i += 1
    data_lines = lines[i:]
    H_dict = {}
    R_set = set()
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue  
        R = tuple(map(int, parts[0:3]))
        m = int(parts[3]) - 1
        n = int(parts[4]) - 1
        val = float(parts[5]) + 1j * float(parts[6])
        H_dict[(R, m, n)] = val
        R_set.add(R)

    return num_wann, sorted(R_set), H_dict

def read_r_tensor(filename="wannier90_r.dat"):
    with open(filename, "r") as f:
        lines = f.readlines()
    num_wann = int(lines[1].strip())
    num_Rpts = int(lines[2].strip())
    data_lines = lines[3:]
    R_set = set()
    r_tensor_x = {}
    r_tensor_y = {}
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        R = tuple(map(int, parts[0:3]))
        m = int(parts[3]) - 1
        n = int(parts[4]) - 1
        rx = float(parts[5])
        ry = float(parts[7])
        r_tensor_x[(R, m, n)] = rx
        r_tensor_y[(R, m, n)] = ry
        R_set.add(R)
  
    return num_wann, sorted(R_set), r_tensor_x, r_tensor_y

# Calculate the sum <0M|r|RN> e^{ik·R}
def r_sum_k(kvec, num_wann, R_list, r_tensor_alpha):
    r_sum = np.zeros((num_wann, num_wann), dtype=complex)
    for R in R_list:
        Rvec = R[0]*a1 + R[1]*a2 + R[2]*a3
        phase = np.exp(1j * np.dot(kvec, Rvec))
        for m in range(num_wann):
            for n in range(num_wann):
                key = (R, m, n)
                if key in r_tensor_alpha:
                    r_sum[m, n] += r_tensor_alpha[key] * phase
    return r_sum

def prepare_H_tensor_and_Rvec(num_wann, R_list, H_dict):
    H_tensor = np.zeros((len(R_list), num_wann, num_wann), dtype=complex)
    Rvec_array = np.zeros((len(R_list), 3))
    for idx, R in enumerate(R_list):
        Rvec = R[0]*a1 + R[1]*a2 + R[2]*a3
        Rvec_array[idx] = Rvec
        for m in range(num_wann):
            for n in range(num_wann):
                key = (R, m, n)
                if key in H_dict:
                    H_tensor[idx, m, n] = H_dict[key]
    return H_tensor, Rvec_array

# Calculate px, py = ∂H/∂k
def momentum_operator_vectorized(kvec, H_tensor, Rvec_array):
    phase = np.exp(1j * Rvec_array @ kvec)
    weighted_phase_x = (1j * Rvec_array[:, 0] * phase)[:, np.newaxis, np.newaxis]
    weighted_phase_y = (1j * Rvec_array[:, 1] * phase)[:, np.newaxis, np.newaxis]
    px = np.sum(H_tensor * weighted_phase_x, axis=0)
    py = np.sum(H_tensor * weighted_phase_y, axis=0)
    return me_over_hbar * px, me_over_hbar * py

# Main function to get the momentum matrix elements
def extract_pcv_full(hr_file, r_file, nk=100, out_csv="p_cv_full.csv"):
    num_wann, R_list, H_dict = read_hr(hr_file)
    H_tensor, Rvec_array = prepare_H_tensor_and_Rvec(num_wann, R_list, H_dict)
    num_wann_r, R_list_r, r_tensor_x, r_tensor_y = read_r_tensor(r_file)
    assert num_wann == num_wann_r, "Unmatched number of Wannier functions between hr.dat and r.dat"
    a_lat = 3.190
    kmax = 2 * np.pi / a_lat
    kx_vals = np.linspace(-kmax, kmax, nk)
    ky_vals = np.linspace(-kmax, kmax, nk)
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals)
    kx_list = kx_grid.flatten()
    ky_list = ky_grid.flatten()
    val_idx = 6  
    cond_idx = 7 
    data = []
    for kx, ky in tqdm(zip(kx_list, ky_list), total=len(kx_list), desc="Scanning k-grid"):
        kvec = np.array([kx, ky, 0.0])
        Hk = np.zeros((num_wann, num_wann), dtype=complex)
        for R in R_list:
            Rvec = R[0]*a1 + R[1]*a2 + R[2]*a3
            phase = np.exp(1j * np.dot(kvec, Rvec))
            for m in range(num_wann):
                for n in range(num_wann):
                    key = (R, m, n)
                    if key in H_dict:
                        Hk[m, n] += H_dict[key] * phase
        evals, U = np.linalg.eigh(Hk)
        # Tính ∂H/∂k
        px, py = momentum_operator_vectorized(kvec, H_tensor, Rvec_array)
        p_bloch_x = U.conj().T @ px @ U
        p_bloch_y = U.conj().T @ py @ U
        # Term 2: use position matrix
        delta_e = evals[cond_idx] - evals[val_idx]
        Rx_k = r_sum_k(kvec, num_wann, R_list_r, r_tensor_x)
        Ry_k = r_sum_k(kvec, num_wann, R_list_r, r_tensor_y)
        r_bloch_x = U.conj().T @ Rx_k @ U
        r_bloch_y = U.conj().T @ Ry_k @ U
        dp_x = 1j * me_over_hbar * delta_e * r_bloch_x[cond_idx, val_idx]
        dp_y = 1j * me_over_hbar * delta_e * r_bloch_y[cond_idx, val_idx]
        # The total momentum: derivative part + correction part
        total_px = p_bloch_x[cond_idx, val_idx] + dp_x
        total_py = p_bloch_y[cond_idx, val_idx] + dp_y
        p_plus = total_px + 1j * total_py
        p_minus = total_px - 1j * total_py
        p_abs = np.sqrt(np.abs(total_px)**2 + np.abs(total_py)**2)
        data.append([kx / (2*np.pi/a_lat), ky / (2*np.pi/a_lat),
                     total_px, total_py, abs(p_plus), abs(p_minus), p_abs])
    df = pd.DataFrame(data, columns=["kx", "ky", "px", "py", "p_plus", "p_minus", "p_abs"])
    df.to_csv(out_csv, index=False)
    print(f"✅ Already write: {out_csv}")

def extract_abs_pcv(hr_file, r_file, nk=100, out_csv="abs_p_cv.csv"):
    num_wann, R_list, H_dict = read_hr(hr_file)
    H_tensor, Rvec_array = prepare_H_tensor_and_Rvec(num_wann, R_list, H_dict)
    num_wann_r, R_list_r, r_tensor_x, r_tensor_y = read_r_tensor(r_file)
    assert num_wann == num_wann_r, "Unmatched number of Wannier functions between hr.dat and r.dat"
    a_lat = 3.190
    kmax = 2 * np.pi / a_lat
    kx_vals = np.linspace(-kmax, kmax, nk)
    ky_vals = 0
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals)
    kx_list = kx_grid.flatten()
    ky_list = ky_grid.flatten()
    val_idx = 6  # band 6
    cond_idx = 7  # band 7
    data = []
    for kx, ky in tqdm(zip(kx_list, ky_list), total=len(kx_list), desc="Scanning k-grid"):
        kvec = np.array([kx, ky, 0.0])
        Hk = np.zeros((num_wann, num_wann), dtype=complex)
        for R in R_list:
            Rvec = R[0]*a1 + R[1]*a2 + R[2]*a3
            phase = np.exp(1j * np.dot(kvec, Rvec))
            for m in range(num_wann):
                for n in range(num_wann):
                    key = (R, m, n)
                    if key in H_dict:
                        Hk[m, n] += H_dict[key] * phase
        evals, U = np.linalg.eigh(Hk)
        # Tính ∂H/∂k
        px, py = momentum_operator_vectorized(kvec, H_tensor, Rvec_array)
        p_bloch_x = U.conj().T @ px @ U
        p_bloch_y = U.conj().T @ py @ U
        # Số hạng 2: dùng ma trận vị trí
        delta_e = evals[cond_idx] - evals[val_idx]
        Rx_k = r_sum_k(kvec, num_wann, R_list_r, r_tensor_x)
        Ry_k = r_sum_k(kvec, num_wann, R_list_r, r_tensor_y)
        r_bloch_x = U.conj().T @ Rx_k @ U
        r_bloch_y = U.conj().T @ Ry_k @ U
        dp_x =    1j * me_over_hbar * delta_e * r_bloch_x[cond_idx, val_idx]
        dp_y =    1j * me_over_hbar * delta_e * r_bloch_y[cond_idx, val_idx]
        # Tổng momentum: phần đạo hàm + phần hiệu chỉnh
        total_px = p_bloch_x[cond_idx, val_idx] + dp_x
        total_py = p_bloch_y[cond_idx, val_idx] + dp_y
        p_plus = total_px + 1j * total_py
        p_minus = total_px - 1j * total_py
        p_abs = np.sqrt(np.abs(total_px)**2 + np.abs(total_py)**2)
        data.append([kx / (2*np.pi/a_lat), ky / (2*np.pi/a_lat),
                     total_px, total_py, abs(p_plus), abs(p_minus), p_abs])
    df = pd.DataFrame(data, columns=["kx", "ky", "px", "py", "p_plus", "p_minus", "p_abs"])
    df.to_csv(out_csv, index=False)
    print(f"✅ Đã ghi file: {out_csv}")
    
extract_pcv_full("mos2_hr294.dat", "mos2_r294.dat", nk=299, out_csv="p_cv_full_294.csv")
extract_abs_pcv("mos2_hr294.dat", "mos2_r294.dat", nk=299, out_csv="p_cv_abs_294.csv")

df = pd.read_csv("p_cv_abs_294.csv")
df.to_csv("p_cv_abs_294.txt", sep=" ", index=False)

