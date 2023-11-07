import pandas as pd 
import matplotlib.pyplot as plt

def save_pie(data:pd.Series, triplet_type:str, save_path:str):
    """Saves a pie chart of relative frequencies for a given triplet type
    Args:
        data (pd.Series): series of cancer types
        triplet_type (str): the triplet type we are looking at
        save_path (str): path to the directory where the graphs will be saved
    Returns:
        None
    """
    frequencies = data.value_counts() / len(data)
    plt.figure(figsize=(20,10))
    plt.pie(frequencies.values, labels=frequencies.index, autopct='%1.1f%%')
    plt.title("Cancer Type Frequencies in {0} triplets".format(triplet_type))
    plt.savefig(SAVED_FIGS + triplet_type + "_triplets_pie.png")
    plt.close()
def save_bar(data:pd.Series, triplet_type:str, save_path:str):
    """
    Saves a bar chart of absolute frequencies for a given triplet type
    Args:
        data (pd.Series): series of cancer types
        triplet_type (str): the triplet type we are looking at
        save_path (str): path to the directory where the graphs will be saved
    Returns:
        None
    """
    frequencies = data.value_counts()
    plt.figure(figsize=(20,10))
    plt.bar(frequencies.index, frequencies.values)
    plt.xticks(rotation=90)
    plt.title("Cancer Type Frequencies in {0} triplets".format(triplet_type))
    plt.savefig(save_path + triplet_type + "_triplets.png")
    plt.close()

def get_frequency_graphs(triplet_types:dict, data_path:str, save_path:str):
    """plot frequency of cancer type across triplet types
    Args: 
        triplet_types (dict): dictionary mapping triplet types to their cancer index
        data_path (str): path to the directory with the data
        save_path (str): path to the directory where the graphs will be saved
    Returns:
        None
    """
    all_cancer_triplets = None
    for triplet_type, cancer_index in triplet_types.items():
        data = pd.read_csv(data_path + triplet_type + "_triplets.txt", sep="\t", header=None)
        data.rename(columns={cancer_index: "cancer_type"}, inplace=True)
        data = data["cancer_type"]
        save_bar(data, triplet_type, save_path)
        save_pie(data, triplet_type, save_path)
        if all_cancer_triplets is None:
            all_cancer_triplets = data
        else:
            all_cancer_triplets = pd.concat([all_cancer_triplets, data])
    save_bar(all_cancer_triplets, "all", save_path)
    save_pie(all_cancer_triplets, "all", save_path)


if __name__ == "__main__":
    DATA_PATH = 'data/MSK/'
    TRIPLET_TYPES = ["cancer_to_drug", 'cancer_to_gene', 'cancer_to_treatment_triplets', 'gene_to_up_regulate_to_cancer', 'all']
    TRIPLET_TYPES = {"cancer_to_drug": 0, 'cancer_to_gene': 0, 'cancer_to_treatment': 0, 'gene_to_up_regulate_to_cancer': 2}
    SAVED_FIGS = 'saved_figs/'
    get_frequency_graphs(TRIPLET_TYPES, DATA_PATH, SAVED_FIGS)
