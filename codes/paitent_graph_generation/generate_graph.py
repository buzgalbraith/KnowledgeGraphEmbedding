import pandas as pd 
import os 
import numpy as np
GENERATED_DATA_PATH = 'codes/paitent_graph_generation/generated_data/'


def get_all_patient_ids(triplets_to_consider)->list:
    all_pids = []
    for triplet_type in triplets_to_consider:
        df = pd.read_csv(GENERATED_DATA_PATH + triplet_type + ".txt", sep='\t', header=None)
        type_pid = np.unique(df[0]).tolist()
        all_pids += type_pid
    all_pids = np.unique(all_pids).tolist()
    print("all pids: ", len(all_pids))
    with open(GENERATED_DATA_PATH + "all_patient_ids.txt", 'w') as f:
        for pid in all_pids:
            f.write(pid + '\n')
        
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
            try:
                one, two, three, = temp 
            except:
                one ,two = temp
                three = " NA"
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

def train_test_split(triplets_to_consider,train_ratio=0.75, val_ratio=.10, seed = 15100873):
    assert np.isclose(train_ratio + val_ratio + (1-train_ratio - val_ratio), 1)
    all_pids = pd.read_csv(GENERATED_DATA_PATH + "all_patient_ids.txt", sep='\t', header=None).values
    all_pids = all_pids.reshape(-1)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_pids)
    train_pids = all_pids[:int(train_ratio*len(all_pids))]
    val_pids = all_pids[int(train_ratio*len(all_pids)):int((train_ratio+val_ratio)*len(all_pids))]
    test_pids = all_pids[int((train_ratio+val_ratio)*len(all_pids)):]
    for triplet in triplets_to_consider:
        df = pd.read_csv(GENERATED_DATA_PATH + triplet + ".txt", sep='\t', header=None, names=['head', 'relation', 'tail'])
        train_df = df[df['head'].isin(train_pids.tolist())]
        val_df = df[df['head'].isin(val_pids.tolist())]
        test_df = df[df['head'].isin(test_pids.tolist())]
        train_df.to_csv(GENERATED_DATA_PATH + triplet +'/'+ "train.txt", sep='\t', index=False, header=False)
        val_df.to_csv(GENERATED_DATA_PATH + triplet +'/' + "valid.txt", sep='\t', index=False, header=False)
        test_df.to_csv(GENERATED_DATA_PATH + triplet +'/' + "test.txt", sep='\t', index=False, header=False)



if __name__ == "__main__":
    ## read in the triplets 
    ## list all files in directory
    files = os.listdir(GENERATED_DATA_PATH)
    triplets_to_consider = [file[:-4] for file in files if (file.endswith(".txt") and "triplets" in file)]
    triplets_to_consider = triplets_to_consider
    for triplet in triplets_to_consider:
        print("working on triplet: ", triplet)
        get_entities_and_relations(path = GENERATED_DATA_PATH + triplet + ".txt", triplet_type = triplet)
    entity_to_triplet_type(triplets_to_consider)
    relations_to_triplet_type(triplets_to_consider)
    get_all_patient_ids(triplets_to_consider)
    train_test_split(triplets_to_consider)