import numpy as np
import pandas as pd

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

momentum_tb = read_selected_columns('momentum cho tb.txt', [0, 1, 4, 5], complex_columns=[4, 5])
D_tb = read_selected_columns('Dx Dy Global.txt', [0, 1, 2, 3], complex_columns=[2, 3])

kx = momentum_tb[:, 0].astype(float)
ky = momentum_tb[:, 1].astype(float)
S1 = momentum_tb[:, 2].astype(complex)
S2 = momentum_tb[:, 3].astype(complex)

Dx = D_tb[:, 2].astype(complex)
Dy = D_tb[:, 3].astype(complex)

px_new = S1 + Dx
py_new = S2 + Dy

with open('file test.txt', 'w') as f:
    f.write("# kx ky px py |p+| |p-| p\n")
    for i in range(len(kx)):
        
        p = np.sqrt(abs(px_new[i])**2 + abs(py_new[i])**2)
        pp = px_new[i] + 1j*py_new[i]
        pm = px_new[i] - 1j*py_new[i]

        f.write(f"{kx[i]:.6e} {ky[i]:.6e} "
                f"({px_new[i].real:.6e},{px_new[i].imag:.6e}) ({py_new[i].real:.6e},{py_new[i].imag:.6e}) {abs(pp):.6e} {abs(pm):.6e} {p:.6e}\n")

print("✅ Đã lưu file file test.txt thành công!")