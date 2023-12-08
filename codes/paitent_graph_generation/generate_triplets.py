import pandas as pd
import numpy as np 
import warnings
import os
import re


# TOOD: wait for the data generation stuff to finish running, then validate results for binary auc on the original data as well as data split by pid 
warnings.filterwarnings("ignore")
def format_clinical(path, save_path):
    race_pattern = r'A\s+(.*?)\s(Male|Female)'
    age_pattern = r'age\s+(\w+)\syears'
    race_pattern = re.compile(race_pattern, re.IGNORECASE | re.DOTALL)
    age_pattern = re.compile(age_pattern, re.IGNORECASE | re.DOTALL)
    pid_race_dict = {}
    pid_sex_dict = {}
    pid_age_dict = {}
    with open (path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pid, demographic_info = line.split('\t')
            matches = race_pattern.findall(demographic_info)
            age_match = age_pattern.findall(demographic_info)
            try:
                race = matches[0][0].strip()
            except:
                race = " NA "
            try:
                sex = matches[0][1].strip()
            except:
                sex = " NA "
            try:
                age = age_match[0].strip()
            except:
                age = " NA "
            if age == "":
                age = " NA "
            if race == "":
                race = " NA "
            if sex == "":
                sex = " NA "
            pid_race_dict[pid] = race
            pid_age_dict[pid] = age
            pid_sex_dict[pid] = sex
    f.close()
    for atr, d in zip(["race", "sex", "age"], [pid_race_dict, pid_sex_dict, pid_age_dict]):
        file_save_path = save_path+"pid_"+atr+"_triplets.txt"
        with open (file_save_path, "w") as f: 
            for pid in d.keys():
                f.write(pid+"\t"+atr+"\t"+d[pid]+'\n')
        f.close()
def get_patient_to_cancer_type(path = 'codes/triplet_generation/original_data/patient_cancer_status_triplet.txt', save_path='codes/paitent_graph_generation/generated_data/patient_cancer_type_triplets.txt'):
    pid_to_cancer_type = {}
    with open (path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pid, status, cancer_type = line.split('\t')
            pid_to_cancer_type[pid] = [status.strip(),cancer_type.strip()]
    f.close()
    with open (save_path, "w") as f:
        for pid in pid_to_cancer_type.keys():
            f.write(pid+"\t"+pid_to_cancer_type[pid][0]+"\t"+pid_to_cancer_type[pid][1]+'\n')
    f.close()
def get_treatment_and_drug(path="codes/triplet_generation/original_data/patient_treatment.txt", save_path="codes/paitent_graph_generation/generated_data/"):
    df = pd.read_csv(path , sep="\t", header=0, names=["patient_id", "treatment"])
    df['treatment'] = df['treatment'].str.replace('The patient was treated with', '')
    delimiter = 'using agent'
    df['drugs'] = df['treatment'].str.split(delimiter, expand=True)[1]
    df['drugs'] = df['drugs'].str.replace('with response measure.*', '', regex=True) ## drug used
    df['treatment'] = df['treatment'].str.replace('using agent.*', '', regex=True) ## treated with 
    for atr, relation in zip(["treatment", "drugs"], ["drug used", "treated with"]):
        file_path = save_path+"pid_"+atr+"_triplets.txt"
        with open (file_path, "w") as f:
            for index, row in df.iterrows():
                if row[atr].strip() == "":
                    row[atr] = " NA " ##TCGA-A1-A0SO
                f.write(row['patient_id']+"\t"+relation+"\t"+row[atr]+'\n')
        f.close()
def patient_mutation_gene_triplets(path = "codes/triplet_generation/original_data/patient_mutationgene_triplet.txt", save_path="codes/paitent_graph_generation/generated_data/"):
    df = pd.read_csv(path, sep="\t", header=0, names=["patient_id", "mutation", "gene"])
    df.fillna(" NA ", inplace=True)
    missense_dfs = df[df['mutation'].str.contains("missense_variant")]
    non_missense_dfs = df[~df['mutation'].str.contains("missense_variant")]
    for df, mutation_type in zip([missense_dfs, non_missense_dfs], ["missense_variant", "non_missense_variant"]):
        file_path = save_path+"pid_"+mutation_type+"_gene_triplets.txt"
        print("write mutation type: ", mutation_type)
        with open (file_path, "w") as f:
            for index, row in df.iterrows():
                if row['mutation'].strip() == "":
                    row['mutation'] = " NA "
                if row['gene'].strip() == "":
                    row['gene'] = " NA "
                f.write(row['patient_id']+"\t"+mutation_type+'\t'+row["gene"]+'\n')
        f.close()

def get_all_triplets(path, save_path):
    all_df = None
    files = os.listdir(path)
    triplets_to_consider = [file[:-4] for file in files if (file.endswith(".txt") and "triplets" in file and "all" not in file)]
    for triplet_type in triplets_to_consider:
        df = pd.read_csv(path + triplet_type + ".txt", sep='\t', header=None, names=["head", "relation", "tail"])
        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat([all_df, df])
    all_df.fillna(" NA ", inplace=True)
    all_df.to_csv(save_path+"all_triplets.txt", sep='\t', index=False, header=False)



if __name__ == "__main__":
    save_path = "./codes/paitent_graph_generation/generated_data/"
    os.makedirs(save_path, exist_ok=True)
    clinical_path = 'codes/triplet_generation/original_data/patient_clinical.txt'
    format_clinical(clinical_path, save_path)
    # get_patient_to_cancer_type()
    get_treatment_and_drug()
    patient_mutation_gene_triplets()
    get_all_triplets(save_path, save_path)
    all_path = save_path+"all_triplets.txt"
    df = pd.read_csv(all_path, sep='\t', header=None, names=["head", "relation", "tail"])