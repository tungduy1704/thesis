import numpy as np
import matplotlib.pyplot as plt

def plot_combined_band_structure(kp_filename, tb_filename):
    # Đọc dữ liệu từ mô hình k·p
    data_kp = np.loadtxt(kp_filename)
    kx_kp = data_kp[:, 0] + 2/3  # Dịch chuyển kx của k.p lên 2/3
    band1_kp = data_kp[:, 1]
    band2_kp = data_kp[:, 2]

    # Đọc dữ liệu từ mô hình tight-binding (TB)
    data_tb = np.loadtxt(tb_filename)
    kx_tb = data_tb[:, 0]
    band1_tb = data_tb[:, 1]
    band2_tb = data_tb[:, 2]
    band3_tb = data_tb[:, 3]

    # Tạo figure với tỷ lệ hẹp cho k.p
    fig, ax = plt.subplots(figsize=(8, 6))

    # Vẽ band structure của mô hình tight-binding (TB)
    ax.plot(kx_tb, band1_tb, 'b', label='Band 1 (TB)')
    ax.plot(kx_tb, band2_tb, 'r', label='Band 2 (TB)')
    ax.plot(kx_tb, band3_tb, 'g', label='Band 3 (TB)')

    # Vẽ band structure của mô hình k·p (đã dịch chuyển)
    ax.plot(kx_kp, band1_kp, 'b--', label='Valence Band (KP)')
    ax.plot(kx_kp, band2_kp, 'r--', label='Conduction Band (KP)')

    # Định dạng trục
    ax.set_xlabel(r"$k_x$", fontsize=14)
    ax.set_ylabel('Energy (eV)', fontsize=14)
    ax.set_title('Band Structure Comparison: k·p vs TB', fontsize=16)

    # Chỉnh tỷ lệ trục để làm hẹp k.p
    ax.set_aspect(0.3)  # Làm hình hẹp như k.p trước đó

    # Đưa chú thích ra ngoài khung hình để gọn hơn
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)

    # Giới hạn trục kx từ Γ đến M
    #ax.set_xlim(0, 1)

    # Chỉ hiển thị các điểm Γ, K, M trên trục kx
    kx_ticks = [0, 2/3, 1]  # Γ, K, M
    kx_labels = [r'$\Gamma$', r'$K$', r'$M$']

    ax.set_xticks(kx_ticks)
    ax.set_xticklabels(kx_labels, fontsize=14)

    # Hiển thị biểu đồ
    plt.tight_layout()  # Đảm bảo layout không bị cắt
    plt.show()

# Gọi hàm để vẽ với giới hạn kx từ Γ đến M
plot_combined_band_structure('bandstr1 cho kp.txt', 'bandstr1 3NN.txt')
