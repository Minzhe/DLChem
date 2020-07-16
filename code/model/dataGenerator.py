#################################################################################
###                            dataGenerator.py                               ###
#################################################################################
import pandas as pd
import torch

### ----------------------------  Dataset  ---------------------------- ###
class ResponseDataset(torch.utils.data.Dataset):
    __initialized = False
    def __init__(self, resp, genome_data, chem_data, impute_genome=True):
        '''
        Args:
            resp:
            genomic:
            chemical:
        '''
        self.resp = resp
        self.n = len(self.resp.index)
        if not all(pd.Series(['Cell', 'Drug', 'LOG50']).isin(self.resp.columns.tolist())):
            raise ValueError('Response table must contain "Cell", "Drug", "LOG50".')
        self.resp.index = list(range(self.n))
        self.genome_feat = genome_data.keys()
        self.genome_data = genome_data
        self.chem_data = chem_data
        self.len_feat = {f: len(self.genome_data[f].columns) for f in self.genome_feat}
        self.len_feat['chem'] = len(self.chem_data.columns)
        self.impute_genome = impute_genome
        self.__initialized = True

    def __len__(self):
        '''
        Denotes the number of samples
        '''
        return self.n
    
    def __getitem__(self, index):
        '''
        Generate one batch of data.
        '''
        # Generate indexes of the batch
        cell = self.resp['Cell'][index]
        drug = self.resp['Drug'][index]
        ic50 = self.resp['LOG50'][index]
        # Get data
        data = self.__data_generation(cell, drug)
        return data, torch.tensor(ic50)
    
    def __data_generation(self, cell, drug):
        '''
        Generates data matching the cell and drug.
        '''
        data = dict()
        for f in self.genome_feat:
            if cell in self.genome_data[f].index:
                data[f] = torch.tensor(self.genome_data[f].loc[cell, :].values)
            elif self.impute_genome:
                data[f] = self.__impute_genome(self.genome_data[f])
            else:
                raise ValueError('Cell {} not found in genome data.'.format(cell))
        if drug in self.chem_data.index:
            data['chem'] = torch.tensor(self.chem_data.loc[drug,:].values)
        else:
            raise ValueError('Drug {} not found in chem data'.format(drug))
        # check dimension
        for f in data.keys():
            dim = list(data[f].size())
            assert len(dim) == 1, '{} feature should be one dimensional for {} and {}, but got {}'.format(f, cell, drug, data[f])
            assert dim[0] == self.len_feat[f], '{} feature length does not match with data: {}, {}'.format(f, dim[0], self.len_feat[f])
        return data
    
    def __impute_genome(self, df):
        '''
        Impute with mean value
        '''
        return torch.tensor(df.mean(axis=0).values)
    
