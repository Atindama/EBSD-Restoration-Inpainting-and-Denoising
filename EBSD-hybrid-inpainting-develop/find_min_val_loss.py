import argparse

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    args = parser.parse_args()

    # Read lines from log file.
    with open(args.log_file) as f:
        lines = f.readlines()

    # Find minimum validation loss.
    val_loss_lines = [line for line in lines if 'Validation loss' in line]
    val_losses = [float(line.split(' ')[-1][:-1]) for line in val_loss_lines]

    print('Minimum validation loss of', min(val_losses),
          'at epoch', np.argmin(val_losses))
