#!/bin/bash 
## bash script to run, data geneaioon and preprocessing files
# Generate data for training and testing
module purge
module load python/intel/3.8.6
DATA_PATH='./data'

CODE_PATH="./codes/paitent_graph_generation"
echo "Generating triplets ..."

python $CODE_PATH'/generate_triplets.py' 
echo "done"

echo "Pre-processing data ..."
python $CODE_PATH'/generate_graph.py'
echo "done "

echo "moving data"
rm -rf $DATA_PATH'/paitent_focoused_kg'
if [ $? -ne 0 ]; then
  exit 1
  fi
cp -r $CODE_PATH'/generated_data' $DATA_PATH'/paitent_focoused_kg'
if [ $? -ne 0 ]; then
  exit 1
  fi
cd $DATA_PATH'/paitent_focoused_kg/all_triplets'
if [ $? -ne 0 ]; then
  exit 1
  fi
mv ./* ..
cd ..
rm -r ./all_triplets
if [ $? -ne 0 ]; then
  exit 1
  fi
cd ..
cd ..
pwd

# data/paitent_focoused_kg/patient_cancer_type_triplets
# cp -r $DATA_PATH/paitent_focoused_kg/patient_cancer_type_triplets $DATA_PATH
# if [ $? -ne 0 ]; then
#   exit 1
#   fi
cp -r $DATA_PATH/paitent_focoused_kg/pid_age_triplets/ $DATA_PATH
if [ $? -ne 0 ]; then
  exit 1
  fi
cp -r $DATA_PATH/paitent_focoused_kg/pid_drugs_triplets/ $DATA_PATH
if [ $? -ne 0 ]; then
  exit 1
  fi
cp -r $DATA_PATH/paitent_focoused_kg/pid_mutation_missense_variant_triplets/ $DATA_PATH
if [ $? -ne 0 ]; then
  exit 1
  fi
cp -r $DATA_PATH/paitent_focoused_kg/pid_mutation_non_missense_variant_triplets/ $DATA_PATH
if [ $? -ne 0 ]; then
  exit 1
  fi
cp -r $DATA_PATH/paitent_focoused_kg/pid_race_triplets/ $DATA_PATH
if [ $? -ne 0 ]; then
  exit 1
  fi
cp -r $DATA_PATH/paitent_focoused_kg/pid_sex_triplets/ $DATA_PATH
if [ $? -ne 0 ]; then
  exit 1
  fi
cp -r $DATA_PATH/paitent_focoused_kg/pid_treatment_triplets/ $DATA_PATH
if [ $? -ne 0 ]; then
  exit 1
  fi
echo "done"
