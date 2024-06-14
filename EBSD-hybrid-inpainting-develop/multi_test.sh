#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=32000
#SBATCH --time=7-00:00:00
source /mnt/home/atindaea/miniconda3/etc/profile.d/conda.sh
conda activate hybrid-inpainting
python multi_test_paper.py "$@"
