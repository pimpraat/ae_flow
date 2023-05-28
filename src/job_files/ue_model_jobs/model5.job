#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=AE_NF_UE_Model1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=14:00:00
#SBATCH --mem=32000M
#SBATCH --output=job_files/train.out

module purge
module load 2021
module load Anaconda3/2021.05

# Your job starts in the directory where you call sbatch
# Activate your environment
source activate dl2022

# The 5 random seeds we will use: 85 59 91 68 1

# Run experiments on the X-ray dataset
# These lines below indicate how to run an experiment
srun python -u train.py --epochs 100 --dataset chest_xray --subnet_architecture resnet_like --model ae_flow --n_validation_folds 5 --num_workers 3 --seed 1 --ue_model True
