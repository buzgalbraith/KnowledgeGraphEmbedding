#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode,data_path, negative_sample_type = "uniform", entity2id = None):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.data_path = data_path
        self.negative_sample_type = negative_sample_type ## add optional arg for negative sampling type
        if self.negative_sample_type == "dict":
            self.triplet_maps = {
        "cancer_to_drug": {"head":"cancer_type", "relation":"drug_used", "tail":"drug_type"},
        "cancer_to_gene": {"head":"cancer_type", 'relation':'mutation' ,"tail":"gene_mutated"},
        "cancer_to_treatment": {'head':"cancer_type", 'relation':'treated_with', 'tail':'treatment_type'},
        "gene_to_up_regulate_to_cancer": {'head':'gene', 'relation':'up_down_regulates', 'tail':'cancer_type'},
            }
            self.entity2id = entity2id ## mapping (entity, entity_id) 
            self.possible_entity_hash = self.possible_entity_hash() ## mapping (entity_type, possible_entities)
            self.entity_type_hash = self.entity_type_2_id() 


    def entity_type_2_id(self, path_extension:str = 'entity_to_triplet_type.txt'):
            """Returns a dictionary mapping entity id to entity type.
            
            Args:
                path_extension (str, optional ) : path from self.data_path to the entity type file. Defaults to 'entity_type.txt'. 
            Returns:
                entity_type_hash (dict) : dictionary mapping entity id to entity type
            
            """
            entity_type_hash = {}
            path = self.data_path + "/" + path_extension
            with open(path) as fin:
                for line in fin:
                    entity_type , entity = line.strip().split('\t')
                    entity_id = self.entity2id[entity.strip()]
                    entity_type_hash[int(entity_id)] = entity_type
            return entity_type_hash
    
    def possible_entity_hash(self)->dict:
        """Returns a dictionary mapping entity type to a numpy array of possible entities to use as negative samples.
            Returns:
                possible_entity_hash (dict): dictionary mapping entity type to a numpy array of possible entities (ids) to use as negative samples.
        """
        possible_entity_hash = {}
        for triplet_type in self.triplet_maps:
            for direction in ["head", "tail"]:
                entity_type = self.triplet_maps[triplet_type][direction]
                path = self.data_path + "/" + entity_type + ".dict" 
                ## make a dictionary
                vals = []
                with open(path, "r") as f: 
                    for line in f:
                        vals.append(int(self.entity2id[line.strip().split('\t')[1].strip()]))
                possible_entity_hash[entity_type] = np.array(vals)
        return possible_entity_hash

    def __len__(self):
        return self.len
    def get_negative_sample(self, entity_name) -> np.ndarray:
        """Returns numpy array of negative samples, using given method. 
        Args:
            entity_name (str) : entity_name of the triplet being considered (can either be head or tail of the triplet)
        Returns:
            negative_sample (np.ndarray) : of negative samples
        """ 
        if self.negative_sample_type == "uniform":
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
        else:
            current_triplet_type = self.entity_type_hash[entity_name] ## get the entity type of the current entity type. 
            current_entity_type = self.triplet_maps[current_triplet_type]["head"] if self.mode == "head-batch" else self.triplet_maps[current_triplet_type]["tail"]
            possible_entities = self.possible_entity_hash[current_entity_type] ## get the possible entities for that entity type
            negative_sample = np.random.choice(size=self.negative_sample_size*2, a = possible_entities) ## sample from the possible entities
        return negative_sample



    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample
        
        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
         
            negative_sample = self.get_negative_sample(head)
            
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                ) ## 
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode,data_path ,negative_sample_type = "uniform", entity2id = None):
        self.len = len(triples)
        self.triple_set = set(all_true_triples) ## this is for them the concatenation of all entity types  from the train test and validation sets
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.data_path = data_path
        self.negative_sample_type = negative_sample_type ## add optional arg for negative sampling type
        if self.negative_sample_type == "dict":
            self.triplet_maps = {
        "cancer_to_drug": {"head":"cancer_type", "relation":"drug_used", "tail":"drug_type"},
        "cancer_to_gene": {"head":"cancer_type", 'relation':'mutation' ,"tail":"gene_mutated"},
        "cancer_to_treatment": {'head':"cancer_type", 'relation':'treated_with', 'tail':'treatment_type'},
        "gene_to_up_regulate_to_cancer": {'head':'gene', 'relation':'up_down_regulates', 'tail':'cancer_type'},
            }
            self.entity2id = entity2id ## mapping (entity, entity_id) 
            self.possible_entity_hash = self.possible_entity_hash() ## mapping (entity_type, possible_entities)
            self.entity_type_hash = self.entity_type_2_id() 


    def entity_type_2_id(self, path_extension:str = 'entity_to_triplet_type.txt'):
            """Returns a dictionary mapping entity id to entity type.
            
            Args:
                path_extension (str, optional ) : path from self.data_path to the entity type file. Defaults to 'entity_type.txt'. 
            Returns:
                entity_type_hash (dict) : dictionary mapping entity id to entity type
            
            """
            entity_type_hash = {}
            path = self.data_path + "/" + path_extension
            with open(path) as fin:
                for line in fin:
                    entity_type , entity = line.strip().split('\t')
                    entity_id = self.entity2id[entity.strip()]
                    entity_type_hash[int(entity_id)] = entity_type
            return entity_type_hash
    
    def possible_entity_hash(self)->dict:
        """Returns a dictionary mapping entity type to a numpy array of possible entities to use as negative samples.
            Returns:
                possible_entity_hash (dict): dictionary mapping entity type to a numpy array of possible entities (ids) to use as negative samples.
        """
        possible_entity_hash = {}
        for triplet_type in self.triplet_maps:
            for direction in ["head", "tail"]:
                entity_type = self.triplet_maps[triplet_type][direction]
                path = self.data_path + "/" + entity_type + ".dict" 
                ## make a dictionary
                vals = []
                with open(path, "r") as f: 
                    for line in f:
                        vals.append(int(self.entity2id[line.strip().split('\t')[1].strip()]))
                possible_entity_hash[entity_type] = np.array(vals)
        return possible_entity_hash


    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.negative_sample_type == "uniform":
            if self.mode == 'head-batch':
                tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                    else (-1, head) for rand_head in range(self.nentity)] 
                tmp[head] = (0, head)
            elif self.mode == 'tail-batch':
                tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                    else (-1, tail) for rand_tail in range(self.nentity)]
                tmp[tail] = (0, tail)
            else:
                raise ValueError('negative batch mode %s not supported' % self.mode)
        else:
            if self.mode == 'head-batch':
                
                current_triplet_type = self.entity_type_hash[head] ## get the entity type of the current entity type. 
                current_entity_type = self.triplet_maps[current_triplet_type]["head"] 
                possible_entities = self.possible_entity_hash[current_entity_type] ## get the possible entities for that entity type
                tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                    else (-1, head) for rand_head in possible_entities] 
                tmp[head] = (0, head)
                return 0 
            elif self.mode == 'tail-batch':
                current_triplet_type = self.entity_type_hash[head] ## get the entity type of the current entity type. 
                current_entity_type = self.triplet_maps[current_triplet_type]["tail"] 
                possible_entities = self.possible_entity_hash[current_entity_type] ## get the possible entities for that entity type
                tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                    else (-1, tail) for rand_tail in possible_entities]
                tmp[tail] = (0, tail)
            else:
                raise ValueError('negative batch mode %s not supported' % self.mode)  
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
