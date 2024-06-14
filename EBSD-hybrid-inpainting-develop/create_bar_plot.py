import argparse
import os

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

METHODS = ['ML', 'ML_Criminisi_w0.00',
           'ML_Criminisi_w0.05', 'ML_Criminisi_w0.10',
           'ML_Criminisi_w0.15', 'ML_Criminisi_w0.20',
           'ML_Criminisi_w0.25', 'ML_Criminisi_w0.30',
           'ML_Criminisi_w0.35', 'ML_Criminisi_w0.40',
           'ML_Criminisi_w0.45', 'ML_Criminisi_w0.50',
           'ML_Criminisi_w0.55', 'ML_Criminisi_w0.60',
           'ML_Criminisi_w0.65', 'ML_Criminisi_w0.70',
           'ML_Criminisi_w0.75', 'ML_Criminisi_w0.80',
           'ML_Criminisi_w0.85', 'ML_Criminisi_w0.90',
           'ML_Criminisi_w0.95', 'ML_Criminisi_w1.00',
           'Criminisi_SSDelta_Euclidean_1', 'Criminisi_SSDelta_Euclidean_1_WxH',
           'Criminisi_SSDelta', 'Criminisi_standard']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('errors_dir')
    parser.add_argument('output_image')
    args = parser.parse_args()

    error_file_names: list[str] = os.listdir(args.errors_dir)
    errors = np.empty((len(error_file_names) // len(METHODS), len(METHODS)),
                      np.float64)
    for error_file_name in tqdm(error_file_names):
        indexes = (int(error_file_name.partition('_')[0]),
                   METHODS.index(error_file_name.partition('_')[2][:-4]))
        error_file_path = os.path.join(args.errors_dir, error_file_name)
        errors[indexes] = np.load(error_file_path).mean()
    average_errors = np.mean(errors, axis=0)
    errors = np.array(errors).astype(np.float64)
    average_errors = np.mean(errors, axis=0)
    print('Average Error\tMethod')
    for method, error in zip(METHODS, average_errors):
        print(f'{error:.10f}\t\t{method}')
    colors = []
    widths = []
    x_poss = []
    cmap = matplotlib.colormaps['viridis']
    for i in range(len(average_errors)):
        if i == 0:
            color = cmap(0.0)
            width = 0.8
            x_pos = 0
        elif 1 <= i <= 21:
            color = cmap(0.2)
            width = 0.4
            x_pos = 0.2 + 0.6 * i
        elif i == 22 or i == 23:
            color = cmap(0.4)
            width = 0.8
            x_pos = 0.4 + 0.6 * i
        elif i == 24:
            color = cmap(0.6)
            width = 0.8
            x_pos = 0.6 + 0.6 * i
        else:
            color = cmap(0.8)
            width = 0.8
            x_pos = 0.8 + 0.6 * i
        colors.append(color)
        widths.append(width)
        x_poss.append(x_pos)
    plt.bar(range(len(average_errors)), average_errors, color=colors)
    # plt.bar(x_poss, average_errors, color=colors, width=widths)
    # plt.xlabel('Inpainting Method')
    plt.ylabel('Mean Average Disorientation')
    plt.xticks(ticks=(1, 11, 21), labels=('ω=0', 'ω=0.5', 'ω=1'))
    plt.tick_params(axis='x', which='both', bottom=False)
    patches = []
    _, idxs = np.unique(colors, axis=0, return_index=True)
    unique_colors = np.array(colors)[np.sort(idxs)]
    for color, label in zip(unique_colors,
                            ('ML', 'Hybrid',
                             'Criminisi (SSDelta, Euclidean)',
                             'Criminisi (SSDelta)',
                             'Criminisi (Standard)')):
        patches.append(mpatches.Patch(label=label, color=color))
    plt.legend(handles=patches)
    plt.savefig(args.output_image, dpi=300)
