import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataloader import TestDataset
import torch
from torch.nn.functional import softmax

## TODO: now we want to add functionality that associates each head with a list of potential tails and vice versa, so that we can do negative sampling 


def prob_auc(score:torch.Tensor, true_labels:torch.Tensor, run_args,**args):
    """computes ROC_AUC score for a given score and true labels
    Args:
        score (torch.Tensor): the score of the model for the given triplets (n_samples, n_entities)
        true_labels (torch.Tensor): the true labels for the triplets (n_samples, )
        args: arguments for roc_auc_score
    REturns:
        au_roc (float): the ROC_AUC score for the given score and true labels
    """
    prob_vector = softmax(score, dim=1) ## soft max over the entities  (to make them probability vectors)
    if run_args.cuda == True: ## have to be on same device to convert to numpy 
        true_labels = true_labels.cpu().numpy()
        prob_vector = prob_vector.cpu().numpy()
    au_roc = roc_auc_score(true_labels, prob_vector, **args)  
    return au_roc

def total_dataloader(all_true_triples:list, args)->list:
    """constructs the dataloader for AUC calculation.
    Args:
        all_true_triples (list): list of all the true triplets
        args (Namespace): arguments for the model
    Returns:
        test_dataset_list (list): list of the dataloaders for the test set
    """
    test_dataloader_head = DataLoader(
                TestDataset(
                    all_true_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

    test_dataloader_tail = DataLoader(
                TestDataset(
                    all_true_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
    test_dataset_list = [test_dataloader_head, test_dataloader_tail]
    return test_dataset_list


def auc_total(all_true_triples:list, model, logging,run_args, **args)->None:
        """computes ROC_AUC score over all data (ie training + validation + test). 
        it is like this since, the roc_auc_score requires all the possible entities to be present in the true label vector which is hard to obtain in this case since some entities are quite rare. 
        Args: 
            all_true_triples (list): list of all the true triplets
            model (KGEModel): the model to evaluate
            logging (logging): the logger
            run_args (Namespace): arguments for the model
            args: arguments for roc_auc_score
        Returns:
            None : just logs the score
        """
        model.eval()
        ## construct the test dataloader
        test_dataset_list = total_dataloader(all_true_triples, run_args)
        all_scores = None
        all_true_labels = None
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])
        with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if run_args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias ## this is our continuous score of shape (batch_size, n_entities)
                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        
                        if all_scores is None:
                            all_scores = score
                        else:
                            all_scores = torch.cat((all_scores, score), dim=0)
                        if all_true_labels is None:
                            all_true_labels = positive_arg
                        else:
                            all_true_labels = torch.cat((all_true_labels, positive_arg), dim=0)
                        step += 1
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                        
        auc = prob_auc(all_scores, all_true_labels,run_args, **args)
        logging.info('-'*100) 
        logging.info('AUC: %f' % auc)
        logging.info('-'*100) 


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


