import pandas as pd 
import os 
import numpy as np
GENERATED_DATA_PATH = './codes/triplet_generation/generated_triplets/'
def get_entities_and_relations(path:str, triplet_type:str)->None:
    """takes path to a triplet file, saves all the entities and relations to a text file

    Args:
        path (string): path to a triplet file
        triplet_type (string): name of the triplet file, will be used to name the directory where the entities and relations are saved
    Returns:
        None
    """
    ## makes a directoty if it doesn't exist
    triplet_type_path = GENERATED_DATA_PATH + triplet_type + "/"
    try: 
        os.mkdir(triplet_type_path)
    except:
        pass

    with open(path, 'r') as f:
        heads = []
        tails = []
        relations = []
        for line in f:
            line = line.strip()
            temp = line.split('\t')
            one, two, three, = temp 
            heads.append(one)
            relations.append(two)
            tails.append(three)
    ## get unique values for each of the heads, tails, and relations
    heads = pd.Series(heads).drop_duplicates()
    tails = pd.Series(tails).drop_duplicates()
    entities = pd.concat([heads, tails]).drop_duplicates().reset_index(drop=True)
    relations = pd.Series(relations).drop_duplicates().reset_index(drop=True)
    ## want to mak
    entities.fillna(' NA', inplace=True)
    relations.fillna(' NA', inplace=True)
    entities.to_csv(triplet_type_path +"entities.dict", sep='\t', index=True, header=False)
    relations.to_csv(triplet_type_path + "relations.dict", sep='\t', index=True, header=False)

def make_train_test_val(path:str,triplet_type : str, train_ratio= .75, val_ratio=.10, seed = 15100873)->None:
    """takes path to a triplet file reads in that triplet file and splits it into train test and val sets.

    Args:
        path (str) : path to a triplet file
        triplet_type (str) : name of the triplet file, will be used to name the directory where the entities and relations are saved
        train_ratio (float, optional): ratio of the data to be used for training. Defaults to .75.
        val_ratio (float, optional): ratio of the data to be used for validation. Defaults to .10.
        seed (int, optional) : seed used for random number generator. Defaults to 15100873.
    Returns:
        None
    """
    assert np.isclose(train_ratio + val_ratio + (1-train_ratio - val_ratio), 1)
    triplet_type_path = GENERATED_DATA_PATH + triplet_type + "/"
    ## lets read this in without pandas 
    with open(path, 'r') as f:
        rng = np.random.default_rng(seed = seed)
        lines = f.readlines()
        rng.shuffle(lines)
        train_lines = lines[:int(train_ratio*len(lines))]
        val_lines = lines[int(train_ratio*len(lines)):int((train_ratio+val_ratio)*len(lines))]
        test_lines = lines[int((train_ratio+val_ratio)*len(lines)):]
        with open(triplet_type_path + "train.txt", 'w') as f:
            for line in train_lines:
                f.write(line)
        with open(triplet_type_path + "valid.txt", 'w') as f:
            for line in val_lines:
                f.write(line)
        with open(triplet_type_path + "test.txt", 'w') as f:
            for line in test_lines:
                f.write(line)
def entity_to_triplet_type(triplets_to_consider: list)->None:
    """Generates a text file mapping each entity to its triplet type then writes it to a text file 

    Args: 
        triplets_to_consider (list): list of triplet types to consider
    Returns: 
        None
    """
    overall_dict = {}
    for triplet_type in triplets_to_consider:
        with open(GENERATED_DATA_PATH + triplet_type + "/entities.dict", 'r') as f:
            i = 0
            for line in f:
                temp = line.strip().split('\t')
                overall_dict[temp[1]] = triplet_type
    df = pd.DataFrame([overall_dict.values(), overall_dict.keys()]).T
    df.fillna(' NA', inplace=True)
    df.to_csv(GENERATED_DATA_PATH + "/entity_to_triplet_type.txt", sep='\t', index=False, header=False)
def relations_to_triplet_type(triplets_to_consider: list)->None:
    """Generates a text file mapping each relation to its triplet type then writes it to a text file 

    Args: 
        triplets_to_consider (list): list of triplet types to consider
    Returns: 
        None
    """
    overall_dict = {}
    for triplet_type in triplets_to_consider:
        with open(GENERATED_DATA_PATH + triplet_type + "/relations.dict", 'r') as f:
            i = 0
            for line in f:
                temp = line.strip().split('\t')
                overall_dict[temp[1]] = triplet_type
    df = pd.DataFrame([overall_dict.values(), overall_dict.keys()]).T
    df.fillna(' NA', inplace=True)
    df.to_csv(GENERATED_DATA_PATH + "/relation_to_triplet_type.txt", sep='\t', index=False, header=False)

if __name__ == "__main__":
    # ## read in the triplets 
    cancer_to_drug =GENERATED_DATA_PATH + "cancer_to_drug_triplets.txt"
    get_entities_and_relations(cancer_to_drug, 'cancer_to_drug')
    cancer_to_gene = GENERATED_DATA_PATH +  "cancer_to_gene_triplets.txt"
    get_entities_and_relations(cancer_to_gene, 'cancer_to_gene')
    cancer_to_treatment = GENERATED_DATA_PATH + "cancer_to_treatment_triplets.txt"
    get_entities_and_relations(cancer_to_treatment, 'cancer_to_treatment')
    gene_to_up_regulate_to_cancer = GENERATED_DATA_PATH +"gene_to_up_regulate_to_cancer_triplets.txt"
    get_entities_and_relations(gene_to_up_regulate_to_cancer, 'gene_to_up_regulate_to_cancer')
    all = GENERATED_DATA_PATH +"all_triplets.txt"
    get_entities_and_relations(all, 'all_triplets')
    make_train_test_val(all, "all_triplets")

    triplets_to_consider = ["cancer_to_drug", "cancer_to_gene", "cancer_to_treatment", "gene_to_up_regulate_to_cancer"]
    entity_to_triplet_type(triplets_to_consider)
