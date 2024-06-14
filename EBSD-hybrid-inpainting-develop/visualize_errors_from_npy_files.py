import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

METHODS = ['ML', 'Criminisi', 'ML_Criminisi_w0.00',
           'ML_Criminisi_w0.05', 'ML_Criminisi_w0.10',
           'ML_Criminisi_w0.15', 'ML_Criminisi_w0.20',
           'ML_Criminisi_w0.25', 'ML_Criminisi_w0.30',
           'ML_Criminisi_w0.35', 'ML_Criminisi_w0.40',
           'ML_Criminisi_w0.45', 'ML_Criminisi_w0.50',
           'ML_Criminisi_w0.55', 'ML_Criminisi_w0.60',
           'ML_Criminisi_w0.65', 'ML_Criminisi_w0.70',
           'ML_Criminisi_w0.75', 'ML_Criminisi_w0.80',
           'ML_Criminisi_w0.85', 'ML_Criminisi_w0.90',
           'ML_Criminisi_w0.95', 'ML_Criminisi_w1.00']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('errors_dir')
    parser.add_argument('output_image')
    args = parser.parse_args()

    error_file_names = os.listdir(args.errors_dir)
    errors = np.empty((len(error_file_names) // len(METHODS), len(METHODS)),
                      np.float64)
    for error_file_name in tqdm(error_file_names):
        indexes = (int(error_file_name.partition('_')[0]),
                   METHODS.index(error_file_name.partition('_')[2][:-4]))
        error_file_path = os.path.join(args.errors_dir, error_file_name)
        errors[indexes] = np.load(error_file_path).mean()
    average_errors = np.mean(errors, axis=0)
    print('Average Error\tMethod')
    for method, error in zip(METHODS, average_errors):
        print(f'{error:.3f}\t\t{method}')
    plt.bar(range(len(average_errors)), average_errors)
    plt.savefig(args.output_image)
