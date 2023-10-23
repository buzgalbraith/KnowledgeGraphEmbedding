import pandas as pd 
import os
import numpy as np
GENERATED_DATA_PATH = './codes/triplet_generation/generated_triplets/'  

def construct_triplet_head_tail_files(triplet_maps:dict,):
    """creates the head tail and relation files for each triplet type and saves them to the appropriate directory.
    
    Args: 
        triplet_maps (dict): dictionary of the form {"triplet_type": {"head":"head_type", "relation":"relation_type", "tail":"tail_type"}, ..}
        
    """
    entity_types = {}
    for triplet_type, triplet_map in triplet_maps.items():
        path = GENERATED_DATA_PATH + triplet_type + "_triplets.txt"
        if triplet_map["head"] not in entity_types.keys():
            entity_types[triplet_map["head"]] = []
        if triplet_map["tail"] not in entity_types.keys():
            entity_types[triplet_map["tail"]] = []
        if triplet_map["relation"] not in entity_types.keys():
            entity_types[triplet_map["relation"]] = []   
        if "relations" not in entity_types.keys():
            entity_types["relations"] = []
        if "entities" not in entity_types.keys():
            entity_types["entities"] = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                temp = line.split('\t')
                one, two, three, = temp 
                entity_types[triplet_map["head"]].append(one)
                entity_types[triplet_map["relation"]].append(two)
                entity_types[triplet_map["tail"]].append(three)
                entity_types["relations"].append(two)
                entity_types["entities"].append(one)
                entity_types["entities"].append(three)

    for key in entity_types.keys():
        entity_types[key] = pd.Series(entity_types[key]).drop_duplicates().reset_index(drop=True)
        save_path = GENERATED_DATA_PATH + key + ".dict"
        entity_types[key].to_csv(save_path,sep="\t", index=True, header=False)

def make_train_test_val(triplet_types:list, train_ratio= .75, val_ratio=.10, seed = 15100873)->None:
    """takes path to a triplet file reads in that triplet file and splits it into train test and val sets.

    Args:
        triplet_types (list) : name of the triplet files to split, will be used to name the directory where the entities and relations are saved
        train_ratio (float, optional): ratio of the data to be used for training. Defaults to .75.
        val_ratio (float, optional): ratio of the data to be used for validation. Defaults to .10.
        seed (int, optional) : seed used for random number generator. Defaults to 15100873.
    Returns:
        None
    """
    assert np.isclose(train_ratio + val_ratio + (1-train_ratio - val_ratio), 1)
    for triplet_type in triplet_types:
        triplet_type_path = GENERATED_DATA_PATH + triplet_type + "/"
        try: 
            os.mkdir(triplet_type_path)
        except:
            pass
        ## lets read this in without pandas 
        path = GENERATED_DATA_PATH + triplet_type + "_triplets.txt"
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


if __name__ == '__main__':
    triplet_maps = {
        "cancer_to_drug": {"head":"cancer_type", "relation":"drug_used", "tail":"drug_type"},
        "cancer_to_gene": {"head":"cancer_type", 'relation':'mutation' ,"tail":"gene_mutated"},
        "cancer_to_treatment": {'head':"cancer_type", 'relation':'treated_with', 'tail':'treated_with'},
        "gene_to_up_regulate_to_cancer": {'head':'gene', 'relation':'up_down_regulates', 'tail':'cancer_type'},
    }
    
    triplet_types = ["all",'cancer_to_drug', 'cancer_to_gene', 'cancer_to_treatment', 'gene_to_up_regulate_to_cancer']
    construct_triplet_head_tail_files(triplet_maps)
    make_train_test_val(triplet_types)