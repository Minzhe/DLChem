import glob
import pandas as pd

# files = glob.glob('/work/bioinformatics/s418336/projects/DLChem/data/ZINC/*.smi')
# zinc = []
# for f in files:
#     try:
#         data = pd.read_csv(f, sep=' ')
#         print(len(data.index))
#         zinc.append(data)
#     except:
#         continue

# zinc = pd.concat(zinc)[['smiles']]
# zinc.to_csv('/work/bioinformatics/s418336/projects/DLChem/data/zinc.smi', sep=' ', header=None, index=False)

# zinc = pd.read_csv('/work/bioinformatics/s418336/projects/DLChem/data/zinc.smi', sep=' ', header=None, squeeze=True)
# lung_smile = set(pd.read_csv('/work/bioinformatics/s418336/projects/DLMed/data/curated/Lung/drug_smile.csv')['Can_Smile'])
# idx = zinc.apply(lambda x: x in lung_smile)
# print(sum(idx))
# zinc = pd.DataFrame(zinc[~idx])
# zinc.to_csv('/work/bioinformatics/s418336/projects/DLChem/data/zinc.filter.smi', header=None, index=False)

zinc = pd.read_csv('/work/bioinformatics/s418336/projects/DLChem/data/zinc.filter.smi', sep=' ', header=None)
zinc = zinc.sample(frac=1, replace=False)
n = len(zinc.index)
zinc_train = zinc.iloc[:int(n*0.8),:]
zinc_val = zinc.iloc[int(n*0.8):int(n*0.9),:]
zinc_test = zinc.iloc[int(n*0.9):,:]
zinc_train.to_csv('/work/bioinformatics/s418336/projects/DLChem/data/zinc_train.smi', sep=' ', header=None, index=False)
zinc_val.to_csv('/work/bioinformatics/s418336/projects/DLChem/data/zinc_val.smi', sep=' ', header=None, index=False)
zinc_test.to_csv('/work/bioinformatics/s418336/projects/DLChem/data/zinc_test.smi', sep=' ', header=None, index=False)