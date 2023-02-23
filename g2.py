import numpy as np
import os
import re
from lmfit import Model
from pathlib import Path
import matplotlib.pyplot as plt

E9 = 0.000000001
left = 3000
right = 3600

base_path = os.getcwd()
input_folder = os.path.join(base_path, 'input')
output_folder = os.path.join(base_path, 'output')


def plot_data(file, g2_corr, x, **kwargs):
    limits = np.array([0, max(x)])
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(x, g2_corr)
    plt.ylabel('$g^{(2)}(t)$')
    plt.xlabel('Time')
    plt.title(file)
    for index, values in enumerate(kwargs.values()):
        limits[index] = values
    plt.xlim(left=limits[0], right=limits[1])
    plt.ylim(bottom=0)

    y_1 = 0.5 * np.ones(len(x))
    y_2 = 0.666666 * np.ones(len(x))
    plt.plot(x, y_1, '--')
    plt.plot(x, y_2, '--')
    plt.savefig(os.path.join(output_folder, '{0}_plot'.format(Path(file).stem)))
    plt.show()
    return


def correction(file, w, channels):
    with open(os.path.join(input_folder, file)) as fp:
        line = fp.readline()
        cnt = 1
        while cnt < 21:
            line = fp.readline()
            cnt += 1
            apd_1 = re.search('^APD1 (\d+|\d+.\d)k', line)
            apd_2 = re.search('^APD2 (\d+|\d+.\d)k', line)
            time = re.search('Time (\d+)s', line)
            power = re.search('Power (\d+).*W', line)  # controlla lo spazio tra numero e uW
            bg_1 = re.search('BG APD1 (\d+|\d.\d)k', line)
            bg_2 = re.search('BG APD2 (\d+|\d.\d)k', line)

            if apd_1:
                N_1 = float(apd_1.group(1)) * 1000
                # print(N_1)
            if apd_2:
                N_2 = float(apd_2.group(1)) * 1000
                # print(N_2)
            if power:
                P_laser = float(power.group(1))
                # print(P_laser)
            if time:
                time_measure = float(time.group(1))  # Expressed in s
            if bg_1:
                BG_1 = float(bg_1.group(1)) * 1000
                # print(BG_1)
            if bg_2:
                BG_2 = float(bg_2.group(1)) * 1000
                # print(BG_2)

    conteggi = np.genfromtxt(os.path.join(input_folder, file), skip_header=21, dtype=float, autostrip=True)
    C_N = conteggi / (N_1 * N_2 * w * time_measure * E9)
    x = np.linspace(0, channels * w, channels)

    B = BG_1 + BG_2
    S = (N_1 + N_2 - B)

    rho = S / (S + B)

    g2_corr = (C_N - (1 - rho ** 2)) / (rho ** 2)
    return g2_corr, x


def fit_model(g2_function, x, y):
    g2_fit_func = Model(g2_function, name='$g^{(2)}(t)$')
    g2_fit_func.set_param_hint('alpha', value=20, min=1)
    g2_fit_func.set_param_hint('lambda_1', value=0.1, min=0, max=1)
    g2_fit_func.set_param_hint('lambda_2', value=0.025, min=0, max=1)
    g2_fit_func.set_param_hint('t0', value=3530, min=3500, max=3600)
    g2_fit_func.set_param_hint('N', value=2, min=1, max=6)
    params = g2_fit_func.make_params()
    eval_result = g2_fit_func.fit(y, params, t=x)
    print(eval_result.fit_report())
    return eval_result


def save_model(model_result, file_name):
    if file_name not in os.listdir(output_folder):
        with open(os.path.join(output_folder, '{0}_FIT.txt'.format(Path(file_name).stem)), 'w') as f:
            f.write(model_result.fit_report())
    return
