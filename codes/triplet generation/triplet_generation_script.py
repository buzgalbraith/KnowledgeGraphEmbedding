# %%
import pandas as pd

# %%
# Read data from the first file into a DataFrame
df1 = pd.read_csv("./original_data/patient_mutationgene_triplet.txt", sep="\t", header=0, names=["patient_id", "mutation", "gene"])

# %%
df1


# %%
# Read data from the second file into another DataFrame
df2 = pd.read_csv("./original_data/patient_cancer_status_triplet.txt", sep="\t", header=0, names=["patient_id", "has_cancer", "cancer type"])

# %%
df2
## df2 is patient id to cancer type

# %%
cancer_gene = pd.merge(df2, df1, on="patient_id", how="left")

# %%
cancer_gene

# %%
cancer_gene.drop(columns=['patient_id','has_cancer'])

# %%
cancer_gene
## has both the info from df1 and df2 

# %%
df3 = pd.read_csv("./original_data/patient_treatment.txt", sep="\t", header=0, names=["patient_id", "treatment"])

# %%
df3

# %%
df3['treatment'] = df3['treatment'].str.replace('The patient was treated with', '')

# %%
df3

# %%
delimiter = 'using agent'

# %%
df3['drugs'] = df3['treatment'].str.split(delimiter, expand=True)[1]

# %%
df3
# df3 is patient id to drugs

# %%
delimiter2='with response measure'

# %%
df3['drugs'] = df3['drugs'].str.replace('with response measure.*', '', regex=True)

# %%
df3['treatment'] = df3['treatment'].str.replace('using agent.*', '', regex=True)

# %%
df3
## df3 is patient id tratment to  drugs

# %%
treatment = df3['treatment'].unique()
print(treatment)

# %%
drugs = df3['drugs'].unique()
print(drugs)

# %%
cancer_treatment = pd.merge(df2, df3, on="patient_id", how="left")

# %%
cancer_treatment = cancer_treatment.drop(columns=['patient_id','has_cancer'])

# %%
cancer_treat = cancer_treatment.drop_duplicates()

# %%
cancer_treat
## cancer treatment is cancer type to treatment to drug

# %%
cancer_treat['treated with'] = 'treated with'

# %%
cancer_treat

# %%
column_to_move = cancer_treat.pop('treated with')

# Insert the column at the desired location (0-indexed)
cancer_treat.insert(loc=1, column='treated with', value=column_to_move)

# %%
cancer_treat = cancer_treat.drop(columns=['drugs']).dropna(subset=['treatment']).drop_duplicates()

# %%
cancer_treat

# %%
cancer_drug = cancer_treatment.drop_duplicates()

# %%
cancer_drug['drugs used'] = 'drugs used'

# %%
column_to_move = cancer_drug.pop('drugs used')

# Insert the column at the desired location (0-indexed)
cancer_drug.insert(loc=1, column='drugs used', value=column_to_move)

# %%
cancer_drug
## cancer drug is 

# %%
cancer_drug = cancer_drug.drop(columns=['treatment']).dropna(subset=['drugs']).drop_duplicates()

# %%
cancer_drug

# %%
df4 = pd.read_csv("./original_data/tcga_gene_cancer_type.txt", sep="\t", header=0, names=['gene', 'up/downregulate', 'cancer'])

# %%
df4
## df4 is gene to up/downregulate to cancer type

# %%
df4['up/downregulate'] = df4['up/downregulate'].str.replace('_ZSCORES*', '', regex=True)

# %%
df4

# %%
triplets=[df1,df2, cancer_treat, cancer_drug, df4]
# so then the triplet types are 
# df1 (patient id to mutation to gene)
# df2 (patient id to cancer type)
# cancer_treat (cancer type to treatment)
# cancer_drug (cancer type to drugs)
# df4 (gene to up/downregulate to cancer type)

# %%
for triplet in triplets:
    triplet.columns = ['heads', 'relations', 'tails']

# %%
final_triplets=pd.concat(triplets, ignore_index=True)

# %%
final_triplets

# %%
final_triplets.to_csv('./generated_triplets/all_triplets.txt', sep='\t', index=False, header=False)

# %%
df1.to_csv('./generated_triplets/patient_gene_triplet.txt', sep='\t', index=False, header=False)

# %%
df2.to_csv('./generated_triplets/patient_to_cancer.txt', sep='\t', index=False, header=False)

# %%
cancer_treat.to_csv('./generated_triplets/cancer_to_treatment.txt', sep='\t', index=False, header=False)

# %%
cancer_drug
cancer_drug.to_csv('./generated_triplets/cancer_to_drug.txt', sep='\t', index=False, header=False)

# %%
df4.to_csv('./generated_triplets/gene_to_up_regulate_to_cancer.txt', sep='\t', index=False, header=False)


