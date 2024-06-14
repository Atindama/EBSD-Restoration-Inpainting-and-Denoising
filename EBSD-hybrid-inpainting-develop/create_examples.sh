#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=32000
#SBATCH --time=3-00:00:00
source /mnt/home/atindaea/miniconda3/etc/profile.d/conda.sh
conda activate hybrid-inpainting
python create_examples_paper.py "$@"