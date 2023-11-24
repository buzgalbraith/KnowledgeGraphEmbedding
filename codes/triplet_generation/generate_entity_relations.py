import pandas as pd 
import os 
import numpy as np
GENERATED_DATA_PATH = './codes/triplet_generation/generated_triplets/'
PATIENT_ID_TRIPLETS_PATH = './codes/triplet_generation/paitint_id_triplets/'

def get_all_patient_ids()->np.ndarray:
    """gets all the patient ids from the patient_id_triplets files
    Args:
        None
    Returns:
        all_ids (np.ndarray): array of all the patient ids
        
    """
    all_ids = []
    for file in os.listdir(PATIENT_ID_TRIPLETS_PATH):
        df = pd.read_csv(PATIENT_ID_TRIPLETS_PATH + file, sep='\t', header=0, names=['patient_id', 'head', 'relation', 'tail'])
        all_ids.append(df['patient_id'].unique().tolist())
    all_ids = np.unique(np.concatenate(all_ids))
    return all_ids
def split_on_patient_id(train_ratio=0.75, val_ratio=.10, seed=100873, verbose = False)->None:
    assert np.isclose(train_ratio + val_ratio + (1-train_ratio - val_ratio), 1)
    patient_ids = get_all_patient_ids()
    rng = np.random.default_rng(seed = seed)
    rng.shuffle(patient_ids)
    train_patient_ids = patient_ids[:int(train_ratio*len(patient_ids))]
    val_patient_ids = patient_ids[int(train_ratio*len(patient_ids)):int((train_ratio+val_ratio)*len(patient_ids))]
    test_patient_ids = patient_ids[int((train_ratio+val_ratio)*len(patient_ids)):]
    train_patient_ids = pd.DataFrame(train_patient_ids, columns=['patient_id'])
    val_patient_ids = pd.DataFrame(val_patient_ids, columns=['patient_id'])
    test_patient_ids = pd.DataFrame(test_patient_ids, columns=['patient_id'])

    if verbose:
        print("number of train patient ids: {0}".format(len(train_patient_ids)))
        print("number of val patient ids: {0}".format(len(val_patient_ids)))
        print("number of test patient ids: {0}".format(len(test_patient_ids)))
        print("number of total patient ids: {0}".format(len(patient_ids)))
        train_val_overlap = np.intersect1d(train_patient_ids.values, val_patient_ids.values)
        train_test_overlap = np.intersect1d(train_patient_ids.values, test_patient_ids.values)
        val_test_overlap = np.intersect1d(val_patient_ids.values, test_patient_ids.values)
        print("overlap between train and val patient ids: {0}".format(len(train_val_overlap)))
        print("overlap between train and test patient ids: {0}".format(len(train_test_overlap)))
        print("overlap between val and test patient ids: {0}".format(len(val_test_overlap)))
    return train_patient_ids, val_patient_ids, test_patient_ids

def get_entities_and_relations(path:str, triplet_type:str)->None:
    """takes path to a triplet file, saves all the entities relations, heads and tails to text files

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
    ## want to make
    entities.fillna(' NA', inplace=True)
    relations.fillna(' NA', inplace=True)
    ## here we are adding these head and tail files 
    heads = heads.reset_index(drop=True)
    tails = tails.reset_index(drop=True)
    heads.fillna(' NA', inplace=True)
    tails.fillna(' NA', inplace=True)
    

    entities.to_csv(triplet_type_path +"entities.dict", sep='\t', index=True, header=False)
    relations.to_csv(triplet_type_path + "relations.dict", sep='\t', index=True, header=False)
    heads.to_csv(triplet_type_path + "heads.dict", sep='\t', index=True, header=False)
    tails.to_csv(triplet_type_path + "tails.dict", sep='\t', index=True, header=False)
def make_train_test_val_with_patient_id(path:str,triplet_type : str, train_ratio= .75, val_ratio=.10, seed = 15100873)->None:
    assert np.isclose(train_ratio + val_ratio + (1-train_ratio - val_ratio), 1)
    triplet_type_path = GENERATED_DATA_PATH + triplet_type + "/"
    if triplet_type == "all_triplets":
        ## a more complex case. 
        train_patient_ids, val_patient_ids, test_patient_ids = split_on_patient_id(train_ratio, val_ratio, seed)
        train_triplets = []
        val_triplets = []
        test_triplets = []
        for file in os.listdir(PATIENT_ID_TRIPLETS_PATH):
            with open(PATIENT_ID_TRIPLETS_PATH+file, 'r') as f:
                    
                lines = f.readlines()
                i = 0 
                for line in lines:
                    i += 1
                    if i%10000 == 0:
                        print(i)
                    temp = line.strip().split('\t')
                    if temp[0] in train_patient_ids.values:
                        train_triplets.append(line[13:])
                    elif temp[0] in val_patient_ids.values:
                        val_triplets.append(line[13:])
                    elif temp[0] in test_patient_ids.values:
                        test_triplets.append(line[13:])
                    else:
                        print("{0} is a weird patient id".format(temp[0]))
                f.close()
        with open(GENERATED_DATA_PATH + 'gene_to_up_regulate_to_cancer_triplets.txt', 'r') as f:
            rng = np.random.default_rng(seed = seed)
            lines = f.readlines()
            rng.shuffle(lines)
            train_triplets += lines[:int(train_ratio*len(lines))]
            val_triplets += lines[int(train_ratio*len(lines)):int((train_ratio+val_ratio)*len(lines))]
            test_triplets += lines[int((train_ratio+val_ratio)*len(lines)):]
            f.close()
        with open(triplet_type_path + "train.txt", 'w') as f:
            print("writing train file")
            for line in train_triplets:
                f.write(line)
        with open(triplet_type_path + "valid.txt", 'w') as f:
            print("writing val file")
            for line in val_triplets:
                f.write(line)
        with open(triplet_type_path + "test.txt", 'w') as f:
            print("writing test file")
            for line in test_triplets:
                f.write(line)
    elif triplet_type != "gene_to_up_regulate_to_cancer":
        path = PATIENT_ID_TRIPLETS_PATH + triplet_type + "_triplets.txt"
        train_patient_ids, val_patient_ids, test_patient_ids = split_on_patient_id(train_ratio, val_ratio, seed)
        train_triplets = []
        val_triplets = []
        test_triplets = []
        print(path)
        i = 0 
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                i += 1
                if i%10000 == 0:
                    print(i)
                temp = line.strip().split('\t')
                if temp[0] in train_patient_ids.values:
                    train_triplets.append(line[13:])
                elif temp[0] in val_patient_ids.values:
                    val_triplets.append(line[13:])
                elif temp[0] in test_patient_ids.values:
                    test_triplets.append(line[13:])
                else:
                    print("{0} is a weird pid".format(temp[0]))
            f.close()
        with open(triplet_type_path + "train.txt", 'w') as f:
            print("writing train file")
            for line in train_triplets:
                f.write(line)
        with open(triplet_type_path + "valid.txt", 'w') as f:
            print("writing val file")
            for line in val_triplets:
                f.write(line)
        with open(triplet_type_path + "test.txt", 'w') as f:
            print("writing test file")
            for line in test_triplets:
                f.write(line)
    else:
        ## otherwise there is no patient id any way so just use the other method. 
        make_train_test_val_without_patient_id(path, triplet_type, train_ratio, val_ratio, seed)
        




def make_train_test_val_without_patient_id(path:str,triplet_type : str, train_ratio= .75, val_ratio=.10, seed = 15100873)->None:
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
def make_train_test_val(path:str,triplet_type : str, train_ratio= .75, val_ratio=.10, seed = 15100873, use_pid = False)->None:
    if use_pid:
        make_train_test_val_with_patient_id(path, triplet_type, train_ratio, val_ratio, seed)
    else:
        make_train_test_val_without_patient_id(path, triplet_type, train_ratio, val_ratio, seed)

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
    ## read in the triplets 
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
    print("starting all triplets ")
    make_train_test_val(all, "all_triplets", use_pid=True)

    triplets_to_consider = ["cancer_to_drug", "cancer_to_gene", "cancer_to_treatment", "gene_to_up_regulate_to_cancer"]
    for triplet_type in triplets_to_consider:
        print("splitting on patient id for {0}".format(triplet_type))
        make_train_test_val(GENERATED_DATA_PATH + triplet_type + "_triplets.txt", triplet_type, use_pid=True)
    entity_to_triplet_type(triplets_to_consider)
    relations_to_triplet_type(triplets_to_consider)
    