#!/bin/bash
#SBATCH --job-name=case_1
#SBATCH --output=./runs/case_1/mapping_run.out
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
SAVE_ID=uniform
negative_sample_type_train=uniform
negative_sample_type_test=unifrom
BATCH_SIZE=1024
NEGATIVE_SAMPLE_SIZE=256
HIDDEN_DIM=1000
GAMMA=24.0
ALPHA=1.0
LEARNING_RATE=0.0001
MAX_STEPS=75000
TEST_BATCH_SIZE=16
file_name=case_1
triplet_type=all
AUC=True
Device=gpu


echo "----------------------"
echo  "triplet type :  ${triplet_type}" ## prints the task id and the corresponding triplet type
echo "----------------------"
bash ./runs/run_files/${Device}_run.sh $MODE $MODEL $overall_dataset $triplet_type $GPU_DEVICE $SAVE_ID \
$BATCH_SIZE $NEGATIVE_SAMPLE_SIZE $HIDDEN_DIM $GAMMA $ALPHA $LEARNING_RATE $MAX_STEPS $TEST_BATCH_SIZE $triplet_type \
$negative_sample_type_train $negative_sample_type_test $AUC \
-de ## run task on the corresponding dataset
echo "----------------------"
