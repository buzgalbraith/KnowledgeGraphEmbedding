import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dataloader import TestDataset
import torch
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
from matplotlib import cm

## TODO: validate the splitting on patient id stuff and then check binary auc stuff as well

def get_cancer_labels(k=10, path_to_cancer_labels:str = "data/patient_cancer_type_triplets/") -> Dict[str, int]:
    """Returns a dictionary mapping patient id to color representation of cancer type.

    Args:
        k (int, optional): number of cancer types to keep. Defaults to 10.
        path_to_cancer_labels (str, optional): path to the cancer labels file
    Returns:
        pid_to_cancer_map (dict): dict mapping patient id to int representation of cancer type
    """
    files = ["train.txt",'valid.txt', 'test.txt']
    ## read in all patient id and cancer type pairs
    pid_to_cancer_map = {}
    for file in files:
        file_path = path_to_cancer_labels + file
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                pid, _, cancer_type = line.strip().split("\t")
                pid_to_cancer_map[pid] = cancer_type
    cancer_type_map = {cancer_type:i for i, cancer_type in enumerate(np.unique(list(pid_to_cancer_map.values())))}
    ## find k most common cancer types
    cancer_type_counts = {cancer_type:0 for cancer_type in cancer_type_map.keys()}
    for cancer_type in pid_to_cancer_map.values():
        cancer_type_counts[cancer_type] += 1
    ## sort the cancer types by count
    cancer_type_counts = {cancer_type:count for cancer_type, count in sorted(cancer_type_counts.items(), key=lambda item: item[1], reverse=True)}
    ## get the k most common cancer types and associate them with an int for color mapping
    cancer_type_map = {cancer_type:i for i, cancer_type in enumerate(list(cancer_type_counts.keys())[:k])}
    ## get rid of all paints with cancers not in top k 
    pid_to_cancer_map = {pid:cancer_type for pid, cancer_type in pid_to_cancer_map.items() if cancer_type in cancer_type_map.keys()}
    ## get list of those patients
    pids = list(pid_to_cancer_map.keys()) 
    ## get colors for each cancer type and call it c 
    cmap = plt.get_cmap('tab10')
    norm = plt.Normalize(vmin=0, vmax=len(cancer_type_map)-1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    colors = {cancer_type:sm.to_rgba(i) for cancer_type, i in cancer_type_map.items()}
    c = [colors[cancer_type] for cancer_type in pid_to_cancer_map.values()]
    return c, pids, colors


def binary_auc(model, test_triples, entity2entitytype, entitytype2taisls, args):
    """Calculates the ROC_AUC score for a given model and test set. Note that
    Args:
        model (KGEModel): the model to evaluate
        test_triples (list): list of all the true triplets
        entity2entitytype (dict): dict mapping all entities to their entity type
        entitytype2taisls (dict): dict mapping all entity types to their possible tails
        args (Namespace): arguments for the model
    Returns:
        roc_auc (float): the ROC_AUC score for the given model and test set
    """
    if args.triplet_type == 'all':
        raise ValueError("binary auc only works for a single triplet type, in other words for this at one triplet type at a time.")

    sample = list()
    y_true  = list()
    for head, relation, tail in test_triples:
        possible_tails = entitytype2taisls[args.triplet_type] 

        if len(possible_tails) > args.negative_sample_size: ## if there are more possible tails than the negative sample size then we just take that number of them.
            possible_tails = np.random.choice(size=args.negative_sample_size, a = possible_tails)
        for candidate_tail in possible_tails:
            y_true.append(1 if candidate_tail == tail else 0) ## adding them to the y true
            sample.append((head, relation, candidate_tail)) ## adding them to the sample
        ## making sure that the true triplet is in the sample
        sample.append((head, relation, tail)) ## adding the true triplet to the sample
        y_true.append(1)
    sample = torch.LongTensor(sample)
    if args.cuda:
        sample = sample.cuda()
    with torch.no_grad():
        y_score = model(sample).squeeze(1).cpu().numpy()
    y_true = np.array(y_true)
    roc_auc = roc_auc_score(y_true, y_score)
    return roc_auc

def get_entity_type_2_id(args):
        """Returns a dictionary mapping entity id to entity type.
        Args:
            new_entity2id (dict) : dict mapping all entities to their new ids (only matters for testing)
            entity2id (dict): dict mapping all entities to their original ids 
            args (Namespace): arguments for the model
        Returns:
            entity_type_hash (dict) : dictionary mapping entity id to entity type
        
        """
        entity_type_hash = {}
        path_extension = 'entity_to_triplet_type.txt'
        path = args.all_datapath + "/" + path_extension
        print(path)
        with open(path) as fin:
            for line in fin:
                entity_type , entity = line.strip().split('\t')
                if args.triplet_type == 'all':
                    try:
                        entity_id = args.entity2id[entity.strip()]
                    except:
                        entity_id = args.entity2id[str(float(entity))]
                    entity_type_hash[entity_id] = entity_type
                else:
                    if entity_type == args.triplet_type:
                        entity_id = args.new_entity2id[args.entity2id[entity]]
                        entity_type_hash[entity_id] = entity_type
        return entity_type_hash


def get_possible_tails(args):
    """returns a dictionary mapping triplet type to a list of possible tails for a given head.
    
    Args:
        new_entity2id (dict) : dict mapping all entities to their new ids (only matters for testing)
        entity2id (dict): dict mapping all entities to their original ids 
        args (Namespace): arguments for the model
    Returns:
        triplet_type_to_tails (dict) : dictionary mapping triplet type to a list of possible tails for a given head
    """
    base_path = args.all_datapath
    if "MSK" in base_path:
        triplet_types = [args.triplet_type] if args.triplet_type != 'all' else ["cancer_to_drug", "cancer_to_gene", "cancer_to_treatment", "gene_to_up_regulate_to_cancer"]
    else:
        triplet_types = [args.triplet_type] if args.triplet_type != 'all' else ['patient_cancer_type_triplets', 'pid_age_triplets', 'pid_drugs_triplets','pid_mutation_missense_variant_triplets', 'pid_mutation_non_missense_variant_triplets', 'pid_race_triplets','pid_sex_triplets','pid_treatment_triplets' ]

    triplet_type_to_tails = {}
    for triplet_type in triplet_types:
        possible_tails = []
        path = base_path +"/"+ triplet_type + "/tails.dict"
        with open(path) as fin:
            for line in fin:
                _, tail = line.strip().split('\t')
                if args.triplet_type == 'all':
                    try:
                        tail_id = args.entity2id[tail.strip()]
                    except:
                        tail_id = args.entity2id[str(float(tail))]
                    possible_tails.append(tail_id)
                else:
                    if triplet_type == args.triplet_type:
                        tail_id = args.new_entity2id[args.entity2id[tail.strip()]]
                        possible_tails.append(tail_id)
        triplet_type_to_tails[triplet_type] = possible_tails
    return triplet_type_to_tails


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
            entity2id[entity.strip()] = int(eid)
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
            possible_entities.append(entity2id[entity.strip()])
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

