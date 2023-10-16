import pandas as pd 
import os 
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
    ## save the entities and relations to a file
    entities = pd.DataFrame(entities)
    relations = pd.DataFrame(relations)
    entities.to_csv("./generated_triplets/"+triplet_type+"/entities.txt", sep='\t', index=False, header=False)
    relations.to_csv("./generated_triplets/"+triplet_type+"/relations.txt", sep='\t', index=False, header=False)




if __name__ == "__main__":
    ## read in the triplets 
    cancer_to_drug = "./generated_triplets/cancer_to_drug_triplets.txt"
    get_entities_and_relations(cancer_to_drug, 'cancer_to_drug')
    cancer_to_gene = "./generated_triplets/cancer_to_gene_triplets.txt"
    get_entities_and_relations(cancer_to_gene, 'cancer_to_gene')
    cancer_to_treatment = "./generated_triplets/cancer_to_treatment_triplets.txt"
    get_entities_and_relations(cancer_to_treatment, 'cancer_to_treatment')
    gene_to_up_regulate_to_cancer = "./generated_triplets/gene_to_up_regulate_to_cancer_triplets.txt"
    get_entities_and_relations(gene_to_up_regulate_to_cancer, 'gene_to_up_regulate_to_cancer')
