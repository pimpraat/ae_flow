#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=RunAE_Normalized_Flow_Development
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=00:20:00
#SBATCH --mem=32000M
#SBATCH --output=job_files/train.out

module purge
module load 2021
module load Anaconda3/2021.05

# Your job starts in the directory where you call sbatch
# Activate your environment
source activate dl2022

# Run experiments on the X-ray dataset
# These lines below indicate how to run an experiment
srun python -u train.py --dataset chest_xray --subnet_architecture resnet_like --model ae_flow --final_experiments False -fully_deterministic True --epochs 100 --seed 42 
srun python -u train.py --dataset chest_xray --loss_beta 0.0 --model ae_flow --final_experiments False -fully_deterministic True --epochs 100 --seed 42 
