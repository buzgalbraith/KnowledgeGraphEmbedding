import pandas as pd 
import os 
import numpy as np
def get_entities_and_relations(path:str, triplet_type:str)->None:
    """takes path to a triplet file, saves all the entities and relations to a text file

    Args:
        path (string): path to a triplet file
        triplet_type (string): name of the triplet file, will be used to name the directory where the entities and relations are saved
    Returns:
        None
    """
    ## makes a directoty if it doesn't exist
    try: 
        os.mkdir("./generated_triplets/"+triplet_type)
    except:
        pass
    df = pd.read_csv(path, sep="\t", header=None, names=['heads', 'relations', 'tails'])
    ## now want to just take the unique of the heads and tails cols
    entities = pd.concat([df['heads'], df['tails']]).unique()
    relations = df['relations'].unique()
    entities[entities == ''] = np.nan
    relations[relations == ''] = np.nan
    entities[entities == ' '] = np.nan
    relations[relations == ' '] = np.nan
    entities[entities == '  '] = np.nan
    relations[relations == ' '] = np.nan
    entities = pd.DataFrame(entities)
    relations = pd.DataFrame(relations)
    ## want to map anything that is blank or missing to NA

    ## take out spaces 
    # entities.fillna("NA", inplace=True)
    # relations.fillna("NA", inplace=True)
    entities.dropna(inplace=True)
    relations.dropna(inplace=True)
    entities.to_csv("./generated_triplets/"+triplet_type+"/entities.dict", sep='\t', index=True, header=False)
    relations.to_csv("./generated_triplets/"+triplet_type+"/relations.dict", sep='\t', index=True, header=False)

def make_train_test_val(path:str,triplet_type : str, train_ratio= .75, val_ratio=.10)->None:
    """takes path to a triplet file reads in that triplet file and splits it into train test and val sets.

    Args:
        path (str) : path to a triplet file
        triplet_type (str) : name of the triplet file, will be used to name the directory where the entities and relations are saved
        train_ratio (float, optional): ratio of the data to be used for training. Defaults to .75.
        val_ratio (float, optional): ratio of the data to be used for validation. Defaults to .10.
    Returns:
        None
    """
    assert np.isclose(train_ratio + val_ratio + (1-train_ratio - val_ratio), 1)
    df = pd.read_csv(path, sep="\t", header=None, names=['heads', 'relations', 'tails'])
    df = df.sample(frac=1).reset_index(drop=True)
    train = df.iloc[:int(train_ratio*len(df))]
    val = df.iloc[int(train_ratio*len(df)):int((train_ratio+val_ratio)*len(df))]
    test = df.iloc[int((train_ratio+val_ratio)*len(df)):]
    train.to_csv("./generated_triplets/"+triplet_type+"/train.txt", sep='\t', index=False, header=False)
    val.to_csv("./generated_triplets/"+triplet_type+"/valid.txt", sep='\t', index=False, header=False)
    test.to_csv("./generated_triplets/"+triplet_type+"/test.txt", sep='\t', index=False, header=False)
def generate_entity_type(triplets_to_consider):
    overall_dict = {}
    for triplet_type in triplets_to_consider:
        entities = pd.read_csv("./generated_triplets/"+triplet_type+"/entities.dict", sep="\t", header=None, names=['entities'])
        entities['entity_type'] = triplet_type
        ## make a dictionary
        d = dict(zip(entities['entities'], entities['entity_type']))
        overall_dict.update(d)
    ## write to a file 
    #import ipdb; ipdb.set_trace()
       
    with open("./generated_triplets/entity_type.txt", 'w') as f:
        for key in overall_dict.keys():
            f.write(str(key)+"\t"+overall_dict[key]+"\n")
    

    


if __name__ == "__main__":
    # ## read in the triplets 
    cancer_to_drug = "./generated_triplets/cancer_to_drug_triplets.txt"
    get_entities_and_relations(cancer_to_drug, 'cancer_to_drug')
    cancer_to_gene = "./generated_triplets/cancer_to_gene_triplets.txt"
    get_entities_and_relations(cancer_to_gene, 'cancer_to_gene')
    cancer_to_treatment = "./generated_triplets/cancer_to_treatment_triplets.txt"
    get_entities_and_relations(cancer_to_treatment, 'cancer_to_treatment')
    gene_to_up_regulate_to_cancer = "./generated_triplets/gene_to_up_regulate_to_cancer_triplets.txt"
    get_entities_and_relations(gene_to_up_regulate_to_cancer, 'gene_to_up_regulate_to_cancer')
    all = "./generated_triplets/all_triplets.txt"
    get_entities_and_relations(all, 'all_triplets')
    make_train_test_val(all, "all_triplets")

    triplets_to_consider = ["cancer_to_drug", "cancer_to_gene", "cancer_to_treatment", "gene_to_up_regulate_to_cancer"]
    generate_entity_type(triplets_to_consider)
