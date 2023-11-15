import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import roc_auc_score

def auc_roc_single(argsort:np.array, positive_arg:np.array, model, multiclass:bool, **args) -> float:
    """This will calculate the auc_roc score for the model.
    Here we are only taking the top prediction of the model. 
    Args:
        argsort (np.array): array of the predictions of the model
        positive_arg (np.array): array of the true labels
        model (KGEModel): the model we are using
        multiclass (bool): whether or not the model is multiclass
        **args: any other arguments we want to pass to the function
    Returns:
        auc (float): the auc_roc score for the model
    """
    y_hat = argsort[0] ## we are just taking the top prediction 
    y = positive_arg ## declare as a new variable for readability
    if multiclass: ## if this is the case we need to make this one hot 
        y_hat = np.eye(model.nentity)[y_hat] 
        y = np.eye(model.nentity)[y] 
        auc = roc_auc_score(y, y_hat, **args) # call the function 
        return auc 
    else:
        raise ValueError("use batch auc for binary classification")
def auc_roc_batch(argsort:np.array, positive_arg:np.array, model, multiclass:bool, **args) -> float:
    """This will calculate the auc_roc score for the model.
    Here we are only taking the top prediction of the model. 
    Args:
        argsort (np.array): array of the predictions of the model
        positive_arg (np.array): array of the true labels
        model (KGEModel): the model we are using
        multiclass (bool): whether or not the model is multiclass
        **args: any other arguments we want to pass to the function
    Returns:
        auc (float): the auc_roc score for the model
    """
    y_hat = argsort[:,0]
    y = positive_arg
    if multiclass:## do the multiclass case
        ## we need to stack the one hot arrays for each row
        y_hat = np.stack([np.eye(model.nentity)[y_hat[i]] for i in range(len(y_hat))])
        y = np.stack([np.eye(model.nentity)[y[i]] for i in range(len(y))])
        auc = roc_auc_score(y, y_hat, **args)
        return auc
    else: ## want to make y all ones, and y_hat one if it equals y 
        y_hat = np.array([1 if y_hat[i] == y[i] else 0 for i in range(len(y))])
        y = np.ones(len(y))
        auc = roc_auc_score(y, y_hat, **args)
        return auc
    

def get_entity2id(all_datapath: str= "./data/MSK") -> dict:
    """returns a dict mapping all entities to their ids (which is used for training the model on all data) 
    
    Args:
        all_datapath (str): path to the directory with all the training data defaults to ./data/MSK
    Returns:
        entity2id (dict): dict mapping all entities to their ids
    """
    path = all_datapath + "/entities.dict"
    with open(path, 'r') as f:
        entity2id = dict()
        for line in f:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
    return entity2id
def get_relation2id(all_datapath: str)->dict:
    """returns a dict mapping all relations to their ids (which is used for training the model on all data)
    Args:
        all_datapath (str): path to the directory with all the training data defaults to ./data/MSK
    Returns:
        relation2id (dict): dict mapping all relations to their ids
    """
    path = all_datapath + "/relations.dict"
    with open(path, 'r') as f:
        relation2id = {}
        for line in f:
            eid, entity = line.strip().split('\t')
            relation2id[entity] = int(eid)
    return relation2id
def get_possible_entities(test_datapath: str, entity2id: dict)->np.ndarray:
    """finds the possible entities for a certain triplet type and returns them as a numpy array

    Args:
        test_datapath (str): path to the directory with the test data (ie the triplet type we are looking at )
        entity2id (dict): dict mapping all entities to their ids
    Returns:
        possible_entities (np.array): numpy array of all the possible entities for a certain triplet type
    """
    path = test_datapath + "/entities.dict"
    with open (path, 'r') as f:
        possible_entities = []
        for line in f:
            line = line.strip()
            temp = line.split('\t')
            eid, entity = line.strip().split('\t')
            possible_entities.append(entity2id[entity])
    return np.array(possible_entities)
def get_possible_relations(test_datapath:str, relation2id: dict)->np.ndarray:
    """finds the possible relations for a certain triplet type and returns them as a numpy array
    Args: 
        test_datapath (str): path to the directory with the test data (ie the triplet type we are looking at )
        relation2id (dict): dict mapping all relations to their ids
    Returns:
        possible_relations (np.array): numpy array of all the possible relations for a certain triplet type
    """
    path = test_datapath + "/relations.dict"
    with open (path, 'r') as f:
        possible_relations = []
        for line in f:
            line = line.strip()
            temp = line.split('\t')
            eid, entity = line.strip().split('\t')
            possible_relations.append(relation2id[entity])
    return np.array(possible_relations)
def reset_index(entity2id, possible_entities):
    """resets the index of the entities to be from 0 to len(possible_entities) - 1
    
    Args:
        entity2id (dict): dict mapping all entities to their ids
        possible_entities (np.array): numpy array of all the possible entities for a certain triplet type
    Returns:
        new_entity2id (dict): dict mapping all entities to their new ids
    """
    new_entity2id = {}
    for i in range(len(possible_entities)):
        new_entity2id[possible_entities[i]] = i
    return new_entity2id
def reset_triplets(triplets: List[Tuple], new_entity2id: dict, new_relation2id:dict)->list:
    """resets the triplets ids in terms of the new entities and relations.
    Args: 
        triplets  ([tuples]): list of tuples of the form (entity, relation, entity) triplets
        new_entity2id (dict): dict mapping all entities to their new ids
        new_relation2id (dict): dict mapping all relations to their new ids
    Returns:
        new_triplets ([tuples]): list of tuples of the form (entity, relation, entity) triplets with the new ids
    """
    new_triplets= [(new_entity2id[triplets[i][0]], new_relation2id[triplets[i][1]], new_entity2id[triplets[i][2]]) for i in range(len(triplets))]
    return new_triplets
def stratify_model(kge_model, new_entity2id, new_relation2id, possible_entities, possible_relations, args):
            """
            Stratify the model to the test set (type of triplet)
            Args:
                kge_model (KGEModel): the model to stratify
                new_entity2id (dict): dict mapping all entities to their new ids
                new_relation2id (dict): dict mapping all relations to their new ids
                possible_entities (np.array): numpy array of all the possible entities for a certain triplet type
                possible_relations (np.array): numpy array of all the possible relations for a certain triplet type
                args (Namespace): arguments for the model
            Returns:
                kge_model (KGEModel): the stratified model

            """
            kge_model.negative_sample_type = args.negative_sample_type_test
            kge_model.entity2id = new_entity2id
            kge_model.relation2id = new_relation2id
            kge_model.entity_embedding.data = kge_model.entity_embedding.data[possible_entities]
            kge_model.relation_embedding.data = kge_model.relation_embedding.data[possible_relations]
            kge_model.nentity = len(possible_entities)
            kge_model.nrelation = len(possible_relations)
            return kge_model
def txt_to_dict(path_to_txt: str) -> None:
    """takes a text file and creates a dictionary file with the same name.

    Args:
        path_to_txt (str) : path to text file to convert.
    """
    path_to_dict = path_to_txt.replace(".txt", ".dict")
    with open(path_to_txt, "r") as f:
        key_value = 0
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split("\n") for line in lines]
        with open(path_to_dict, "w") as f:
            for key_value, value in enumerate(lines):
                f.write(f"{key_value}\t{value[0]}\n")
        f.close()


def tsv_to_txt(path_to_tsv: str) -> None:
    """converts .tsv file to .txt.

    Args:
        path_to_tsv (str) : path to tsv file to convert.
    """
    txt_path = path_to_tsv.replace(".tsv", ".txt")
    with open(path_to_tsv, "r") as f:
        lines = f.readlines()
        with open(txt_path, "w") as f:
            for line in lines:
                f.write(f"{line}")
        f.close()


def train_to_train_and_val(path: str, validation_rate: float = 0.8) -> None:
    """takes a training file and splits it into a training and validation file.

    Args:
        path(str) : path to train.txt
        validation_rate(float, optional) : ratio of train to validation that is len(train)/len(train \cup validation)

    """
    assert validation_rate <= 1.0 and validation_rate >= 0.0
    with open(path, "r") as f:
        lines = f.readlines()
        train = lines[: int(len(lines) * validation_rate)]
        val = lines[int(len(lines) * validation_rate) :]
        path_to_train = path.replace("train.tsv", "train.txt")
        path_to_val = path.replace("train.tsv", "valid.txt")
        with open(path_to_train, "w") as f:
            for line in train:
                f.write(f"{line}")
        f.close()
        with open(path_to_val, "w") as f:
            for line in val:
                f.write(f"{line}")
        f.close()


