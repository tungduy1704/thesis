import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt



# === Véc-tơ mạng thực (Angstrom)
a1 = np.array([3.190315, 0.0, 0.0])
a2 = np.array([-1.595158, 2.762894, 0.0])
a3 = np.array([0.0, 0.0, 17.439502])

# === Đọc file wannier90_hr.dat
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

# === Chuẩn bị tensor H_tensor và vector Rvec_array
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

# === H(k)
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

def plot_bandstructure_along_kx(filename="wannier90_hr.dat", nk=200):
    num_wann, R_list, H_dict = read_hr(filename)

    a_lat = 3.190315  # a_lat của bạn
    kmax = 2 * np.pi / a_lat

    kx_vals = np.linspace(-kmax, kmax, nk)
    ky_val = 0.0

    bands = np.zeros((nk, num_wann))

    for i, kx in enumerate(tqdm(kx_vals, desc="Đang tính bandstructure")):
        kvec = np.array([kx, ky_val, 0.0])
        Hk = construct_Hk(kvec, num_wann, R_list, H_dict)
        evals, _ = np.linalg.eigh(Hk)
        bands[i, :] = evals # Lấy phần thực (chéo hóa đôi khi có số ảo rất bé)

    # === Vẽ
    plt.figure(figsize=(8,6))
    for band_idx in range(num_wann):
        plt.plot(kx_vals / (2*np.pi/a_lat), bands[:, band_idx], color='black', lw=1)

    plt.xlabel(r"$k_x$ ($2\pi / a$)")
    plt.ylabel("Energy (eV)")
    tick_positions = [0.0, 2/3, 1.0]
    tick_labels = [r'$\Gamma$', r'$K$', r'$M$']
    plt.xticks(tick_positions, tick_labels)
    plt.title("Band structure along $k_x$ (with $k_y=0$)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# === Gọi hàm
#plot_bandstructure_along_kx(filename="mos2_hr.dat", nk=1000)

def plot_bandstructure_kxky(filename="wannier90_hr.dat", nk=100, out_csv="bands_kxky_full.csv"):
    num_wann, R_list, H_dict = read_hr(filename)

    a_lat = 3.190315  # Độ dài mạng cơ sở
    kmax = 2 * np.pi / a_lat

    kx_vals = np.linspace(-kmax, kmax, nk)
    ky_vals = np.linspace(-kmax, kmax, nk)
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals)

    num_kpoints = nk * nk
    bands = np.zeros((nk, nk, num_wann))  # lưu E(kx, ky) cho tất cả bands

    for ix in tqdm(range(nk), desc="Quét kx"):
        for iy in range(nk):
            kx = kx_grid[iy, ix]
            ky = ky_grid[iy, ix]
            kvec = np.array([kx, ky, 0.0])
            Hk = construct_Hk(kvec, num_wann, R_list, H_dict)
            evals, _ = np.linalg.eigh(Hk)
            bands[iy, ix, :] = np.real(evals)  # (iy, ix, band)

    # === Ghi toàn bộ bands ra file CSV
    kx_flat = (kx_grid / (2*np.pi/a_lat)).flatten()
    ky_flat = (ky_grid / (2*np.pi/a_lat)).flatten()
    bands_flat = bands.reshape((-1, num_wann))  # reshape bands thành (nk*nk, num_wann)

    # Gộp thành bảng: (kx, ky, band1, band2, ..., band_num_wann)
    data = np.column_stack([kx_flat, ky_flat, bands_flat])
    cols = ["kx", "ky"] + [f"band_{i+1}" for i in range(num_wann)]

    df = pd.DataFrame(data, columns=cols)
    df.to_csv(out_csv, index=False)
    print(f"✅ Đã ghi file {out_csv} với {num_wann} bands.")

    # === Vẽ nhanh 1 vài band để kiểm tra
    bands_to_plot = [6, 7]  # ví dụ
    fig, axs = plt.subplots(1, len(bands_to_plot), figsize=(7*len(bands_to_plot),6), constrained_layout=True)

    if len(bands_to_plot) == 1:
        axs = [axs]  # Nếu chỉ có 1 band thì vẫn thành list để dễ lặp

    for idx, band_idx in enumerate(bands_to_plot):
        ax = axs[idx]
        cf = ax.contourf(kx_grid / (2*np.pi/a_lat), ky_grid / (2*np.pi/a_lat), bands[:, :, band_idx], levels=50, cmap="jet")
        fig.colorbar(cf, ax=ax, label="Energy (eV)")
        ax.set_xlabel(r"$k_x$ ($2\pi/a$)")
        ax.set_ylabel(r"$k_y$ ($2\pi/a$)")
        ax.set_title(f"Energy surface for band {band_idx}")

    plt.show()

# === Ví dụ gọi
plot_bandstructure_kxky(filename="mos2_hr.dat", nk=49, out_csv="bands_kxky_full.csv")
