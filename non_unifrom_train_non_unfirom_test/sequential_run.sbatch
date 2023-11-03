#!/bin/bash

#SBATCH --job-name=./sequential_run/non_uniform_train_non_uniform_test
#SBATCH --output=./sequential_run/non_uniform_train_non_uniform_test.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1


module purge ## purge modules that we are not using
module load python/intel/3.8.6 ## load python module
echo "Job start at: $(date)" # print the date and time the job finished

## set vars 
MODE=test ## this is only really for testing models using non-unfirom negative sampleing.
MODEL=RotatE
overall_dataset=MSK
GPU_DEVICE=0
SAVE_ID=dict
negative_sample_type_train=dict
negative_sample_type_test=dict
BATCH_SIZE=1024
NEGATIVE_SAMPLE_SIZE=256
HIDDEN_DIM=1000
GAMMA=24.0
ALPHA=1.0
LEARNING_RATE=0.0001
MAX_STEPS=75000
TEST_BATCH_SIZE=1



triplet_types=(cancer_to_drug cancer_to_gene cancer_to_treatment gene_to_up_regulate_to_cancer)
# triplet_types=(all)
for triplet_type in "${triplet_types[@]}"
do
    echo "----------------------"
    echo  "triplet type :  ${triplet_type}" ## prints the task id and the corresponding triplet type
    echo "----------------------"
    bash ./non_unifrom_train_non_unfirom_test/sequential_run.sh $MODE $MODEL $overall_dataset $triplet_type $GPU_DEVICE $SAVE_ID \
    $BATCH_SIZE $NEGATIVE_SAMPLE_SIZE $HIDDEN_DIM $GAMMA $ALPHA $LEARNING_RATE $MAX_STEPS $TEST_BATCH_SIZE $triplet_type \
    $negative_sample_type_train $negative_sample_type_test \
    -de ## run task on the corresponding dataset
    echo "----------------------"
done



