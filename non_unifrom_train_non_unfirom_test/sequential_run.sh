#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models

#The first four parameters must be provided
MODE=$1
MODEL=$2
save_DATASET=$3
test_dataset=$4
GPU_DEVICE=$5
SAVE_ID=$6
## if the test_dataset is all, then we use the save_DATASET as the test_dataset
if [ $test_dataset == "all" ]
then
    test_dataset=$save_DATASET
fi


TEST_DATA_PATH=$DATA_PATH/$test_dataset
FULL_DATA_PATH=$DATA_PATH/$save_DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$save_DATASET"_"$SAVE_ID"

#Only used in training
BATCH_SIZE=$7
NEGATIVE_SAMPLE_SIZE=$8
HIDDEN_DIM=$9
GAMMA=${10}
ALPHA=${11}
LEARNING_RATE=${12}
MAX_STEPS=${13}
TEST_BATCH_SIZE=${14}
TRIPLET_TYPE=${15}
negative_sample_type_train=${16}
negative_sample_type_test=${17}

if [ $MODE == "train" ]
then

echo "Start Training......"


CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path $TEST_DATA_PATH \
    --overall_data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    --triplet_type $TRIPLET_TYPE \
    --negative_sample_type_train $negative_sample_type_train \
    ${18} ${19} ${20} ${21} ${22} ${23} ${24}


elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE --triplet_type $TRIPLET_TYPE --overall_data_path $DATA_PATH --negative_sample_type_test $negative_sample_type_test
    
elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE --data_path $FULL_DATA_PATH  --triplet_type $TRIPLET_TYPE --overall_data_path $DATA_PATH --negative_sample_type_test $negative_sample_type_test

else
   echo "Unknown MODE" $MODE
fi