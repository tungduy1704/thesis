import numpy as np
import pandas as pd

m = 5.6770736
hbar = 0.658229

def read_selected_columns(filename, columns, complex_columns=[]):
    data = []
    with open(filename, 'r') as file:
        next(file)  
        for line_num, line in enumerate(file, start=2):  
            if line.strip() == '':
                continue  
            
            parts = line.strip().split()
            if len(parts) < max(columns) + 1:
                print(f"⚠️ Cảnh báo: Dòng {line_num} thiếu dữ liệu, bỏ qua!")
                continue  

            try:
                selected_data = []
                for col in columns:
                    value = parts[col]
                    if col in complex_columns:
                        if "(" in value and ")" in value:
                            value = value.strip("()")  
                            real, imag = map(float, value.split(','))
                            value = complex(real, imag)
                        else:
                            raise ValueError(f"Lỗi số phức ở dòng {line_num}: {value}")
                    else:
                        value = float(value)  

                    selected_data.append(value)
                data.append(selected_data)
            except ValueError as e:
                print(f"⚠️ Cảnh báo: {e}")
                continue  
    return np.array(data, dtype=object)  

bandstr_tb = read_selected_columns('bandstr2 cho tb.txt', [0, 1, 2, 3, 4])

if  len(bandstr_tb) == 0:
    raise ValueError("⚠️ Lỗi: Một hoặc nhiều file không có dữ liệu hợp lệ!")

epsilon_0 = bandstr_tb[:, 2].astype(float)
epsilon_1 = bandstr_tb[:, 3].astype(float)

def read_C_matrix_filtered(filename):
    """ Đọc dữ liệu ma trận C, chỉ lấy 3 giá trị đầu cho C^0 và 3 giá trị tiếp theo cho C^1, bỏ 3 giá trị cuối. """
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue  
        
        kx, ky = map(float, line.split())
        i += 1
        
        C_values = []
        while i < len(lines) and lines[i].strip():
            real, imag = map(float, lines[i].split())  
            C_values.append(complex(real, imag))
            i += 1  
        
        # Kiểm tra nếu có đủ số phần tử
        if len(C_values) >= 6:
            C_0 = np.array(C_values[:3], dtype=complex)  
            C_1 = np.array(C_values[3:6], dtype=complex)  
        else:
            print(f"⚠️ Cảnh báo: Dữ liệu ở (kx={kx}, ky={ky}) không đủ phần tử!")
            continue  

        data.append((kx, ky, C_0, C_1))
        i += 1  
    
    return data

# ======= ĐỌC DỮ LIỆU C^0, C^1 =======
C_data = read_C_matrix_filtered('vecs_tb.txt')

# Chuyển đổi thành mảng numpy để dễ xử lý
kx_values = np.array([entry[0] for entry in C_data])
ky_values = np.array([entry[1] for entry in C_data])
C_0_matrices = np.array([entry[2] for entry in C_data], dtype=object) 
C_1_matrices = np.array([entry[3] for entry in C_data], dtype=object)

def read_global_D(filename):
    """ Đọc dữ liệu từ file data_for_Dx_global.txt hoặc data_for_Dy_global.txt """
    with open(filename, "r") as f:
        lines = f.readlines()
    
    data_line = lines[1].strip().split(",") 
    data = [complex(float(data_line[i]), float(data_line[i+1])) for i in range(0, len(data_line), 2)]
    
    return tuple(data)  

D12_x, D13_x, D23_x, D12_xconj, D13_xconj, D23_xconj = read_global_D("data_for_Dx_global.txt")
D12_y, D13_y, D23_y, D12_yconj, D13_yconj, D23_yconj = read_global_D("data_for_Dy_global.txt")

output_file = "Dx Dy Global.txt"

with open(output_file, "w") as f:
    f.write("# kx  ky  Dx  Dy |Dplus| |Dminus| |D|\n")  
    for i in range(len(kx_values)):  
        kx, ky = kx_values[i], ky_values[i]
        
        epsilon_0_i, epsilon_1_i = epsilon_0[i], epsilon_1[i]  
        C_0, C_1 = C_0_matrices[i], C_1_matrices[i]  
        C_0 = np.array(C_0, dtype=complex).flatten()
        C_1 = np.array(C_1, dtype=complex).flatten()
        
        if C_0.shape[0] < 3 or C_1.shape[0] < 3:
            print(f"⚠️ Cảnh báo: C_0 hoặc C_1 không có đủ 3 phần tử! Bỏ qua (kx={kx}, ky={ky})")
            continue

        epsilon_diff = epsilon_0_i - epsilon_1_i

        """term_12_x =  (C_0[0].conj() * C_1[1] - C_0[1].conj() * C_1[0]) * D12_x
        term_13_x =  (C_0[0].conj() * C_1[2] + C_0[2].conj() * C_1[0]) * D13_x
        term_23_x =  (C_0[1].conj() * C_1[2] - C_0[2].conj() * C_1[1]) * D23_x"""

        term_12_x =  (C_0[0].conj() * C_1[1] + C_0[1].conj() * C_1[0]) * (0.22492294158354334 * 2.79085)
        Dx_01 = 1j * (m / hbar) * epsilon_diff * (term_12_x)

        """term_12_y =  (C_0[0].conj() * C_1[1] - C_0[1].conj() * C_1[0]) * D12_y
        term_13_y =  (C_0[0].conj() * C_1[2] + C_0[2].conj() * C_1[0]) * D13_y
        term_23_y =  (C_0[1].conj() * C_1[2] - C_0[2].conj() * C_1[1]) * D23_y"""

        term_13_y =  (C_0[0].conj() * C_1[2] + C_0[2].conj() * C_1[0]) * (0.23432913958090337 / 2.79085)
        Dy_01 = 1j * (m / hbar) * epsilon_diff * (term_13_y)
    
        D = np.sqrt(abs(Dx_01)**2 + abs(Dy_01)**2)
        Dp = Dx_01 + 1j * Dy_01
        Dm = Dx_01 - 1j * Dy_01
        
        f.write(f"{kx:.6e} {ky:.6e} "
                f"({Dx_01.real:.6e},{Dx_01.imag:.6e}) ({Dy_01.real:.6e},{Dy_01.imag:.6e}) "
                f"{abs(Dp):.6e} {abs(Dm):.6e} {D:.6e}\n")
    
print(f"✅ Kết quả đã được lưu vào: {output_file}")



