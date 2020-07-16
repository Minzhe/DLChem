import pandas as pd
import numpy as np

drug_path = '/work/bioinformatics/s418336/projects/DLMed/data/curated/Lung/merge_final_version/ccle_utsw.lung_drug.split.csv'
fp_path = '/work/bioinformatics/s418336/projects/DLMed/data/curated/Lung/merge_final_version/drug_fingerprint.csv'

resp = pd.read_csv(drug_path)
fp = pd.read_csv(fp_path, index_col=0)
drugs = pd.Series(list(set(resp['Drug']) & set(fp.index))).sample(frac=1).reset_index(drop=True)
n = len(drugs)
drug_train = drugs[:int(0.7*n)]
drug_val = drugs[int(0.7*n):int(0.9*n)]
drug_test = drugs[int(0.9*n):]

resp['Train_split_drug'] = np.nan
resp.loc[resp['Drug'].isin(drug_train),'Train_split_drug'] = 'Train'
resp.loc[resp['Drug'].isin(drug_val),'Train_split_drug'] = 'Val'
resp.loc[resp['Drug'].isin(drug_test),'Train_split_drug'] = 'Inference'

train_counts = resp.loc[resp['Drug'].isin(drug_train),'Drug'].value_counts()
val_counts = resp.loc[resp['Drug'].isin(drug_val),'Drug'].value_counts()
test_counts = resp.loc[resp['Drug'].isin(drug_test),'Drug'].value_counts()
print('Train {} drug, total {} samples'.format(len(train_counts), sum(train_counts)), train_counts, sep='\n')
print('Val {} drug, total {} samples'.format(len(val_counts), sum(val_counts)), val_counts, sep='\n')
print('Test {} drug, total {} samples'.format(len(test_counts), sum(test_counts)), test_counts, sep='\n')

resp.to_csv(drug_path, index=False)