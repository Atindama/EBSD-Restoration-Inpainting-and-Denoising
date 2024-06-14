import argparse
import os

import numpy as np

KNOWN_WEIGHTS = list(np.arange(0, 1.05, 0.05))
METHOD_STRS = (['ML', 'Criminisi']
               + [f'Hybrid_w{w:.2f}' for w in KNOWN_WEIGHTS])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('errors_dir')
    args = parser.parse_args()

    for method_str in METHOD_STRS:
        errors = []
        i = 0
        while True:
            try:
                error_file_name = f'{i}_{method_str}.npy'
                error_file_path = os.path.join(args.errors_dir,
                                               error_file_name)
                errors.append(np.load(error_file_path))
                i += 1
            except:
                break
        print(f'{method_str}: {np.mean(errors)}')
