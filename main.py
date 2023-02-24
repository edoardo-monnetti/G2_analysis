import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path
import os
import g2

E9 = 0.000000001


# Format of the G2 txt files
# 3rd cross bottom left
#
# APD1 20k
# APD2 21k
#
# Time 1000s
# Power 695 uW
#
# BG APD1 3k
# BG APD2 6k
#
# #TimeHarp 200  Histogram Data			19.02.2023 15:51:51
# #channels per curve
# 4096
# #display curve no.
# 0
# #memory block no.
# 0
# #ns/channel
# 0.8875
# #counts
#

def g2_function(t, alpha, lambda_1, lambda_2, N: int, t0):
    y = 1 - (1 / N) * ((1 + alpha) * np.exp(-lambda_1 * abs(t - t0)) - alpha * np.exp(-lambda_2 * (abs(t - t0))))
    # lambda_1 -> antibunching
    # lambda_2 -> bunching
    return y


plt.close('all')
base_path = os.getcwd()
input_folder = os.path.join(base_path, 'input')
output_folder = os.path.join(base_path, 'output')

channels = 4096
w = 0.8875  # * e9  # Expressed in ns

for file in os.listdir(input_folder):
    if '.txt' not in file:
        continue

    g2_corr, x = g2.correction(file, w, channels)  # Correction for the counts C_N(t)
    # g2.plot_data(file, g2_corr, x)  # Can use **kwargs = left: float, right:float to define boundaries of the plot

    # ####################################### Fitting & plotting with curve_fit
    # best_vals = np.zeros(5)
    # best_vals, covar = curve_fit(g2_function, x, g2_corr, p0=[1, 0.1, 0.025, 1, 3530])
    # plt.plot(x, g2_function(x, *best_vals))
    # plt.show()
    # print(f't0 = {best_vals[4]} \nN = {best_vals[3]} \nlambda_1 = {best_vals[1]} \nalpha = {best_vals[0]}')

    ####################################### Fitting & plotting & saving the model
    plt.style.use('seaborn-v0_8-whitegrid')
    x_data = x[(x > 3000) & (x < 3600)]
    y_data = g2_corr[(x > 3000) & (x < 3600)]
    eval_result = g2.fit_model(g2_function, x_data, y_data)
    g2.save_model(eval_result, file)
    eval_result.plot(datafmt='-', show_init=False)  # yerr=np.sqrt(y_data)

    ####################################### Plot 0.5 (2 NVs) and 0.7 (3 NVs) lines
    left = 3000
    right = 3600
    y_1 = 0.5 * np.ones(len(x_data))
    y_2 = 0.666666 * np.ones(len(x_data))
    plt.plot(x_data, y_1, '--')
    plt.plot(x_data, y_2, '--')
    plt.xlim(right=right, left=left)
    plt.ylim(bottom=0)
    plt.ylabel('$g^{(2)}(t)$')
    plt.xlabel('Time (ns)')
    N_color_centers = round(eval_result.best_values.get('N'), 2)
    plt.legend(title = f"N = {N_color_centers}")
    plt.title(file)
    plt.savefig(os.path.join(output_folder, '{0}_plot_FIT'.format(Path(file).stem)))
    plt.show()