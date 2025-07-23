import matplotlib.pyplot as plt
import numpy as np

setting_A_comp = np.array([0.35699, 0.38, 0.321, 0.1084, 0.0873, 0.088, 0.0806, 0.079, 0.0811])
setting_A_cent = np.array([0.0935, 0.1158, 0.01855, 0.00635, 0.00609, 0.0011, 0.0011, 0.0011])
setting_A_cwce = np.array([0.097, 0.144, 0.118, 0.019, 0.00656, 0.0063, 0.00116, 0.00112, 0.00114])*0.3
setting_A_zscore = np.array([0.34, 0.77, 0.59, 0.074, 0.025, 0.024, 0.0043, 0.0042, 0.0043])*0.2
setting_A_nmse = np.array([0.52, 0.38, 0.33, 0.18, 0.16, 0.16, 0.16, 0.16, 0.16])*0.5
setting_B_comp = np.array([0.0833, 0.288, 0.402, 0.194, 0.0393, 0.041, 0.0413, 0.0371, 0.0374])
setting_B_cent = np.array([0.016, 0.29, 0.164, 0.003, 0.0075, 0.0057, 0.00169, 0.00068])
setting_B_cwce = np.array([0.019, 0.25, 0.31, 0.18, 0.0034, 0.0085, 0.0065, 0.0019, 0.00077])*0.3
setting_B_zscore = np.array([0.032, 0.58, 0.883, 0.42, 0.006, 0.015, 0.011, 0.0034, 0.0013])*0.2
setting_B_nmse = np.array([0.14, 0.19, 0.27, 0.12, 0.074, 0.071, 0.074, 0.072, 0.073])*0.5
setting_C_comp = np.array([0.054, 0.4, 0.34, 0.12, 0.015, 0.014, 0.02, 0.019, 0.02])
setting_C_cwce = np.array([0.017, 0.24, 0.22, 0.089, 0.002, 0.0005, 0.0037, 0.0049, 0.0044])*0.3
setting_C_zscore = np.array([0.032, 0.89, 0.76, 0.25, 0.0047, 0.0013, 0.0088, 0.0116, 0.011])*0.2
setting_C_nmse = np.array([0.14, 0.29, 0.25, 0.085, 0.027, 0.027, 0.034, 0.031, 0.033])*0.5

x = [0, 0.2, 0.25, 0.5, 1, 1.57, 2, 2.5, 3]

plt.figure()
plt.plot(x, setting_A_comp, label='Composite Error', marker='o')
plt.plot(x, setting_A_cwce, label='CWCE', marker='o')
plt.plot(x, setting_A_zscore, label='Z-score', marker='o')
plt.plot(x, setting_A_nmse, label='NMSE', marker='o')
plt.xlabel('Stimulus center')
plt.ylabel('Error')
plt.ylim([-0.02, 0.42])
plt.title('sigma exc = 0.15')
plt.legend()
# plt.show()

plt.figure()
plt.plot(x, setting_C_comp, label='Composite Error', marker='o')
plt.plot(x, setting_C_cwce, label='CWCE', marker='o')
plt.plot(x, setting_C_zscore, label='Z-score', marker='o')
plt.plot(x, setting_C_nmse, label='NMSE', marker='o')
plt.xlabel('Stimulus center')
plt.ylabel('Error')
plt.title('sigma exc = 0.175')
plt.legend()
# plt.show()

plt.figure()
plt.plot(x, setting_B_comp, label='Composite Error', marker='o')
plt.plot(x, setting_B_cwce, label='CWCE', marker='o')
plt.plot(x, setting_B_zscore, label='Z-score', marker='o')
plt.plot(x, setting_B_nmse, label='NMSE', marker='o')
plt.xlabel('Stimulus center')
plt.ylabel('Error')
plt.title('sigma exc = 0.2')
plt.legend()
plt.show()