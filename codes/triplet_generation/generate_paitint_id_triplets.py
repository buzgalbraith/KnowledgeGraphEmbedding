
import pandas as pd
import numpy as np 
import warnings

# Read data from the first file into a DataFrame
warnings.filterwarnings("ignore")
try: 
    os.mkdir('./codes/triplet_generation/paitint_id_triplets')
except:
    pass


original_data_path = "./codes/triplet_generation/original_data"
generated_data_path = './codes/triplet_generation/paitint_id_triplets'
df1 = pd.read_csv(original_data_path + "/patient_mutationgene_triplet.txt", sep="\t", header=0, names=["patient_id", "mutation", "gene"])


df1


# Read data from the second file into another DataFrame
df2 = pd.read_csv(original_data_path + "/patient_cancer_status_triplet.txt", sep="\t", header=0, names=["patient_id", "has_cancer", "cancer type"])


df2


cancer_gene = pd.merge(df2, df1, on="patient_id", how="left")


cancer_gene

## dont drop paitent_id for now
cancer_gene = cancer_gene.drop(columns=['has_cancer'])


cancer_gene


df3 = pd.read_csv(original_data_path + "/patient_treatment.txt", sep="\t", header=0, names=["patient_id", "treatment"])


df3


df3['treatment'] = df3['treatment'].str.replace('The patient was treated with', '')


df3


delimiter = 'using agent'


df3['drugs'] = df3['treatment'].str.split(delimiter, expand=True)[1]


df3


delimiter2='with response measure'


df3['drugs'] = df3['drugs'].str.replace('with response measure.*', '', regex=True)


df3['treatment'] = df3['treatment'].str.replace('using agent.*', '', regex=True)


df3


treatment = df3['treatment'].unique()



drugs = df3['drugs'].unique()



cancer_treatment = pd.merge(df2, df3, on="patient_id", how="left")


## we are going to keep patient_id for now
cancer_treatment = cancer_treatment.drop(columns=['has_cancer'])


cancer_treat = cancer_treatment.drop_duplicates()


cancer_treat


cancer_treat['treated with'] = 'treated with'


cancer_treat


column_to_move = cancer_treat.pop('treated with')

# Insert the column at the desired location (0-indexed)
cancer_treat.insert(loc=1, column='treated with', value=column_to_move)


cancer_treat = cancer_treat.drop(columns=['drugs']).dropna(subset=['treatment']).drop_duplicates()


cancer_treat


cancer_drug = cancer_treatment.drop_duplicates()


cancer_drug['drugs used'] = 'drugs used'


column_to_move = cancer_drug.pop('drugs used')

# Insert the column at the desired location (0-indexed)
cancer_drug.insert(loc=1, column='drugs used', value=column_to_move)


cancer_drug


cancer_drug = cancer_drug.drop(columns=['treatment']).dropna(subset=['drugs']).drop_duplicates()


cancer_drug


df4 = pd.read_csv(original_data_path + "/tcga_gene_cancer_type.txt", sep="\t", header=0, names=['gene', 'up/downregulate', 'cancer'])


df4['up/downregulate'] = df4['up/downregulate'].str.replace('_ZSCORES*', '', regex=True)
df4['up/downregulate'] = df4['up/downregulate'].str.replace(r'UPREGULATES.*', 'UPREGULATES', regex=True)
df4['up/downregulate'] = df4['up/downregulate'].str.replace(r'DOWNREGULATES.*', 'DOWNREGULATES', regex=True)
df4


cancer_gene.replace(r'^\s*$',  np.nan,regex=True, inplace=True) ## fill white space with NA
cancer_treat.replace(r'^\s*$',  np.nan,regex=True, inplace=True)
cancer_drug.replace(r'^\s*$',  np.nan,regex=True, inplace=True)
df4.replace(r'^\s*$',  np.nan,regex=True, inplace=True)
cancer_gene.fillna(' NA', inplace=True)  
cancer_treat.fillna(' NA', inplace=True) ## should be 3555 in train Kidney renal clear cell carcinoma
cancer_drug.fillna(' NA', inplace=True)
df4.fillna(' NA', inplace=True)
triplets=[cancer_gene, cancer_treat, cancer_drug]
cancer_gene = cancer_gene[['patient_id','cancer type', 'mutation', 'gene' ]]
cancer_treat = cancer_treat[['patient_id',  'cancer type', 'treated with','treatment']]
cancer_drug = cancer_drug[['patient_id',  'cancer type', 'drugs used','drugs']]

for triplet in triplets:
    print(triplet.columns)
    print("-"*100)
    triplet.columns = ['patient_id', 'heads', 'relations', 'tails']





cancer_gene.to_csv(generated_data_path+'/cancer_to_gene_triplets.txt', sep='\t', index=False, header=False)
cancer_treat.to_csv(generated_data_path + '/cancer_to_treatment_triplets.txt', sep='\t', index=False, header=False)
# df4.to_csv(generated_data_path + '/gene_to_up_regulate_to_cancer_triplets_patient_id.txt', sep='\t', index=False, header=False)
cancer_drug.to_csv(generated_data_path +'/cancer_to_drug_triplets.txt', sep='\t', index=False, header=False)


