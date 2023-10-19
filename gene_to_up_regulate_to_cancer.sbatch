#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=14:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=gene_to_up_regulate_to_cancer
#SBATCH --output=gene_to_up_regulate_to_cancer.out
module purge ## purge modules that we are not using
module load python/intel/3.8.6 ## load python module
echo "Job start at: $(date)" # print the date and time the job finished
bash uniform_run.sh train RotatE gene_to_up_regulate_to_cancer  0 'gene_to_up_regulate_to_cancer' 1024 256 1000 24.0 1.0 0.0001 50000 16 -de
echo "Job finished at: $(date)" # print the date and time the job finished

