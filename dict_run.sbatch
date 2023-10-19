#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=14:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=MSK_dict_run_2
#SBATCH --output=MSK_dict_run_2.out
module purge ## purge modules that we are not using
module load python/intel/3.8.6 ## load python module
echo "Job start at: $(date)" # print the date and time the job finished
bash dict_run.sh train RotatE MSK  0 'dict' 1024 256 1000 24.0 1.0 0.0001 75000 16 -de
echo "Job finished at: $(date)" # print the date and time the job finished

