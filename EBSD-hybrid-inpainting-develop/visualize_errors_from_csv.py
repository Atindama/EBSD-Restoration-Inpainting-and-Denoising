import argparse
import csv
import itertools

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('errors_file')
    parser.add_argument('output_image')
    args = parser.parse_args()

    with open(args.errors_file) as f:
        csv_reader = csv.reader(f)
        methods = next(csv_reader)  # Ignore column headers
        methods.remove('Criminisi (MSGD)')
        methods.append('Criminisi (MSGD)')
        errors = []
        for error_row in csv_reader:
            errors.append(error_row)
    errors = np.array(errors).astype(np.float64)
    criminisi = errors[:, 1:2]
    errors = np.delete(errors, 1, 1)
    errors = np.append(errors, criminisi, 1)
    average_errors = np.mean(errors, axis=0)
    print('Average Error\tMethod')
    for method, error in zip(methods, average_errors):
        print(f'{error:.3f}\t\t{method}')
    colors = []
    widths = []
    x_poss = []
    for i in range(len(average_errors)):
        if i == 0:
            color = (0.25, 0.25, 0.25)
            width = 0.8
            x_pos = 0
        elif 1 <= i <= 21:
            color = (0.5, 0.5, 0.5)
            width = 0.4
            x_pos = 0.2 + 0.6 * i
        else:
            color = (0.75, 0.75, 0.75)
            width = 0.8
            x_pos = 0.4 + 0.6 * i
        colors.append(color)
        widths.append(width)
        x_poss.append(x_pos)
    plt.bar(range(len(average_errors)), average_errors, color=colors)
    # plt.bar(x_poss, average_errors, color=colors, width=widths)
    plt.xlabel('Inpainting Method')
    plt.ylabel('Mean Squared Disorientation')
    plt.savefig(args.output_image)
