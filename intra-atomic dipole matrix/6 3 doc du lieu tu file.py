import numpy as np

hb = 0.658229  
m0 = 5.6770736

px_tbm = np.array([3.022738 + 0j, 3.237383 + 0j, 3.443929 + 0j, 3.613826 + 0j, 3.714914 + 0j, -3.726330 + 0j, 3.652438 + 0j, -3.521786 + 0j, -3.371845 + 0j, 3.234088 + 0j, -3.128080 + 0j])
py_tbm = np.array([0 + 1j * 3.603273,0 + 1j * 3.645818,0 + 1j * 3.681547,0 + 1j * 3.709178,0 + 1j * 3.727150,0 + 1j * -3.733483,0 + 1j * 3.725604,0 + 1j * -3.700143,0 + 1j * -3.652722,0 + 1j * -3.577814,0 + 1j * -3.468869])

px_kp = np.array([2.827223 + 0j, 2.816869 + 0j, 2.804925 + 0j, 2.791390 + 0j, 2.776265 + 0j, 2.759555 + 0j, 2.741227 + 0j, 2.721248 + 0j, 2.699592 + 0j, 2.676232 + 0j, 2.652714 + 0j])
py_kp = np.array([0 + 1j * 2.669439, 0 + 1j * 2.687882, 0 + 1j * 2.706115, 0 + 1j * 2.724138, 0 + 1j * 2.741952, 0 + 1j * 2.759555, 0 + 1j * 2.776949, 0 + 1j * 2.794134,0 + 1j * 2.811113, 0 + 1j * 2.827871, 0 + 1j * 2.844425])

E0 = np.array([-0.232548, -0.176984, -0.128063, -0.090630, -0.069019, -0.065849, -0.081103, -0.112180, -0.154887, -0.204661, -0.257404])
E1 = np.array([1.866709, 1.771867, 1.693351, 1.636073, 1.604118, 1.599561, 1.621250, 1.665394, 1.726001, 1.796387, 1.870041])

delta_px = px_kp - px_tbm
delta_py = py_kp - py_tbm
delta_E = E0 - E1

delta_all = np.column_stack((delta_px, delta_py, delta_E))
np.savetxt("delta_result.txt", delta_all, fmt="%.6f", header="Delta_px   Delta_py   Delta_E")

data = np.loadtxt("data C.txt")

C0_1 = data[:, 0] + 1j * data[:, 1]
C0_2 = data[:, 2] + 1j * data[:, 3]
C0_3 = data[:, 4] + 1j * data[:, 5]

C1_1 = data[:, 6] + 1j * data[:, 7]
C1_2 = data[:, 8] + 1j * data[:, 9]
C1_3 = data[:, 10] + 1j * data[:,11]

Delta_Cx = np.conj(C0_1) * C1_2 + np.conj(C0_2) * C1_1
Delta_Cy = np.conj(C0_1) * C1_3 + np.conj(C0_3) * C1_1

with open("delta_C_result.txt", "w") as f:
    f.write("#Re_Delta_Cx Im_Delta_Cx Re_Delta_Cy Im_Delta_Cy\n")
    for cx, cy in zip(Delta_Cx, Delta_Cy):
        f.write(f"{cx.real:.6e} {cx.imag:.6e} {cy.real:.6e} {cy.imag:.6e}\n")

print("Đã lưu dữ liệu thành công!!")

# LINEAR REGRESSION
Y1 = delta_px 
Y2 = delta_py

X1 = 1j * (m0 / hb) * delta_E * Delta_Cx
X2 = 1j * (m0 / hb) * delta_E * Delta_Cy

with open("Y_X.txt","w") as f:
    f.write("#Y1 X1 Y2 X2 \n")
    for i in range(len(Y1)):
        f.write(f"({Y1[i]:.6e})  ({X1[i]:.6e})  ({Y2[i]:.6e})   ({X2[i]:.6e}) \n")


# Hồi quy phức: W = (X† Y) / (X† X)
W1 = np.vdot(X1, Y1) / np.vdot(X1, X1) 
W2 = np.vdot(X2, Y2) / np.vdot(X2, X2)

print(f"W1 = {W1}  (Re: {W1.real:.6e}, Im: {W1.imag:.6e})")
print(f"W2 = {W2}  (Re: {W2.real:.6e}, Im: {W2.imag:.6e})")

Y1_pred = W1 * X1
Y2_pred = W2 * X2
residuals1 = Y1 - Y1_pred

residuals2 = Y2 - Y2_pred
mse1 = np.mean(np.abs(residuals1)**2)
mse2 = np.mean(np.abs(residuals2)**2)

print(f"MSE Y1 = {mse1:.6e}")
print(f"MSE Y2 = {mse2:.6e}")
print("Mean |Y1|^2 =", np.mean(np.abs(Y1)**2))
print("Mean |Y2|^2 =", np.mean(np.abs(Y2)**2))
