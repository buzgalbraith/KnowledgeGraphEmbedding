import numpy as np
from typing import List, Tuple, Dict

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
