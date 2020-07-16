####################################################################################
###                               train_fcn.py                                   ###
####################################################################################
import pandas as pd
import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
proj_dir = '/work/bioinformatics/s418336/projects/DLChem'
os.chdir(proj_dir)
sys.path.append('code/model')
from dataGenerator import ResponseDataset
from neural_net import FCN, Trainer
arg = sys.argv[1]
##################################      function     ###################################
def check_two_drug(resp, id1, id2):
    # utsw and ccle has overlap id e.g. ERLOTINIB == SW198886
    df1 = resp.loc[resp['Drug'] == id1,:]
    df2 = resp.loc[resp['Drug'] == id2,:]
    cell = list(set(df1['Cell']) & set(df2['Cell']))
    df1 = df1.loc[resp['Cell'].isin(cell),:]
    df2 = df2.loc[resp['Cell'].isin(cell),:]
    print(df1)
    print(df2)


##################################       main      ####################################
### ---------  1. load data  ---------- ###
data_path = '/work/bioinformatics/s418336/projects/DLMed/data/curated/Lung/merge_final_version/'
mut_path = os.path.join(data_path, 'ccle_utsw.lung_Mut_cancergene_table.csv')
expr_path = os.path.join(data_path, 'ccle_utsw.lung_RNAseq_cancergene.csv')
cnv_path = os.path.join(data_path, 'ccle_utsw.lung_CNV_cancergene_table.csv')
fp_path = os.path.join(data_path, 'drug_fingerprint.csv')
drug_path = os.path.join(data_path, 'ccle_utsw.lung_drug.split.csv')

mut = pd.read_csv(mut_path, index_col=0)
expr = pd.read_csv(expr_path, index_col=0)
# cnv = pd.read_csv(cnv_path, index_col=0)
fp = pd.read_csv(fp_path, index_col=0)
resp = pd.read_csv(drug_path)
resp = resp.loc[resp['Drug'].isin(fp.index.to_list()),:]
# check_two_drug(resp, 'ERLOTINIB', 'SW198886')
resp_train = resp.loc[resp['Train_split_'+arg] == 'Train',:]
resp_val = resp.loc[resp['Train_split_'+arg] == 'Val',:]
resp_test = resp.loc[resp['Train_split_'+arg] == 'Inference',:]

### ---------- 2. data loader ----------- ###
train_data = ResponseDataset(resp=resp_train, genome_data={'mut': mut, 'expr': expr}, chem_data=fp, impute_genome=True)
val_data = ResponseDataset(resp=resp_val, genome_data={'mut': mut, 'expr': expr}, chem_data=fp, impute_genome=True)
test_data = ResponseDataset(resp=resp_test, genome_data={'mut': mut, 'expr': expr}, chem_data=fp, impute_genome=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True, num_workers=1)

### ------------ 3. set fcn model ---------------- ###
model_path = '/work/bioinformatics/s418336/projects/DLChem/code/train/model_archive'
weight_path = os.path.join(model_path, 'lung.genome.chem.split_by_'+arg+'.{}.pt')
log_path = os.path.join(model_path, 'lung.genome.chem.split_by_{}.csv'.format(arg))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: {}'.format(device))
f_mut = len(mut.columns)
f_expr = len(expr.columns)
f_chem = len(fp.columns)
# for _, (X, y) in enumerate(test_loader):
#     # print(_)
#     x = {f: data.float() for f, data in X.items()}
#     # x = X['mut'].float()
#     y = y.float().to(device)
#     # print(x)
#     print(y.cpu().detach().numpy())
#     exit()

# set model
resp_model = FCN(feature_layer_sizes={'mut': (f_mut,256,), 'expr': (f_expr,256,), 'chem': (f_chem,256,)}, 
                 fcn_layer_sizes=(768,512,256),
                 dropout=0.3).to(device)
# set training param
gradient_clip = 5
optimizer = torch.optim.Adam(resp_model.parameters(), lr=1e-4)

### ------------ 4. train fcn model ---------------- ###
trainer = Trainer(resp_model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                  optimizer=optimizer, loss='mse', metrics=['mse', 'r2'],
                  gradient_clip=gradient_clip, device=device)
trainer.train(epochs=5000, model_path=weight_path, log_path=log_path, save_freq=20)