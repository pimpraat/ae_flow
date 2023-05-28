current_experiment_number = 0


for dataset in ['OCT2017', 'chest_xray', 'ISIC', 'BRATS', 'MIIC']:
    for model in ['autoencoder', 'fastflow', 'ae_flow']:

        with open(f'job_files/experiment_run_files/Exp_{current_experiment_number}.txt', 'w') as f:

            lines = []
            
            lines.append("#!/bin/bash")

            lines.append("#SBATCH --partition=gpu_titanrtx_shared_course")
            lines.append("#SBATCH --gres=gpu:1")
            lines.append("#SBATCH --job-name=AE_Experiments")
            lines.append("#SBATCH --ntasks=1")
            lines.append("#SBATCH --cpus-per-task=3")
            lines.append("#SBATCH --time=05:00:00")
            lines.append("#SBATCH --mem=32000M")
            lines.append("#SBATCH --output=job_files/train.out")

            lines.append("module purge")
            lines.append("module load 2021")
            lines.append("module load Anaconda3/2021.05")

            lines.append("# Your job starts in the directory where you call sbatch")
            lines.append("# Activate your environment")
            lines.append("source activate dl2022")

            sbn = 'resnet_like' if dataset == 'chest_xray' else 'conv_like'

            lines.append(str(f"srun python -u train.py --dataset {dataset} --subnet_architecture {sbn} --model {model} --final_experiments True -fully_deterministic True --n_validation_folds 5 --epochs 100 --seed 42"))

            f.writelines(lines)
