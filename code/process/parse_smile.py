from rdkit import Chem
import pandas as pd

def parse_smile(s):
    s = s.replace('.[(Z)]', '').replace('.[(E)]', '')
    if 'Pt' in s:
        s = ''
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(s),True)
    except:
        print(s)
        exit()

smile_path = '/work/bioinformatics/s418336/projects/DLMed/data/curated/Lung/drug_target.csv'
cansmile_path = '/work/bioinformatics/s418336/projects/DLMed/data/curated/Lung/drug_smile.csv'
zinc_path = '/work/bioinformatics/s418336/projects/DLChem/data/zinc.smi'
lung_path = '/work/bioinformatics/s418336/projects/DLChem/data/lung.drug.smi'
fp_path = '/work/bioinformatics/s418336/projects/DLChem/data/lung.drug.0.5_256.fp'
lung_fp_path = '/work/bioinformatics/s418336/projects/DLMed/data/curated/Lung/drug_fingerprint.csv'

# data = pd.read_csv(smile_path)
# data['Smile'] = data['Smile'].fillna('').apply(lambda x: x)
# data['Can_Smile'] = data['Smile'].apply(parse_smile)

# zinc = list(pd.read_csv(zinc_path, header=None, squeeze=True))
# data['in_ZINC'] = data['Can_Smile'].apply(lambda x: x in zinc).astype('int')
# data.to_csv(cansmile_path, index=None)

# data = pd.read_csv(cansmile_path)['Can_Smile']
# data = pd.DataFrame(data[data.notnull()])
# data.to_csv(lung_path, sep=' ', index=False, header=None)

lung = pd.read_csv(cansmile_path)
lung = lung.loc[lung['Can_Smile'].notnull(),['Drug', 'Can_Smile']]
smile = pd.read_csv(lung_path, header=None, names=['smile'])
fp = pd.read_csv(fp_path, sep=' ', header=None)
fp.columns = ['fp'+str(x) for x in fp.columns]
fp.insert(0, column='Can_Smile', value=smile['smile'])
fp = fp.groupby(by='Can_Smile').mean().reset_index()
fp = lung.merge(fp, left_on='Can_Smile', right_on='Can_Smile', how='inner')
# duplicates
idx = fp.duplicated(subset=['Can_Smile'], keep=False)
print(fp.loc[idx,:].sort_values(by='Can_Smile'))
# write
fp.drop('Can_Smile', axis=1, inplace=True)
print(fp)
fp.to_csv(lung_fp_path, index=False)
