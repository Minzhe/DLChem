##################################################################################
###                             neural_net.py                                  ###
##################################################################################
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import pearsonr
import time

####################               network                #######################
class FCN(torch.nn.Module):
    '''
    Network to integrate mutation, expression and chemical fingerprint
    '''
    def __init__(self, feature_layer_sizes, fcn_layer_sizes, dropout):
        '''
        Args:
            layer_sizes: e.g., {"mutation": (10000, 100, 100)} to define the structures, the keys should exist in keys for DataLoader
        '''
        super().__init__()
        # feature layer
        self.features = list(feature_layer_sizes.keys())
        for f in self.features:
            net = []
            in_shape = feature_layer_sizes[f][0]
            for n_neurons in feature_layer_sizes[f][1:]:
                net.append(torch.nn.Linear(in_shape, n_neurons))
                net.append(torch.nn.BatchNorm1d(n_neurons))
                net.append(torch.nn.ReLU6())
                in_shape = n_neurons
            setattr(self, f, torch.nn.ModuleList(net))
        # fcn layer
        in_shape = fcn_layer_sizes[0]
        net = []
        for n_neurons in fcn_layer_sizes[1:]:
            net.append(torch.nn.Linear(in_shape, n_neurons))
            net.append(torch.nn.BatchNorm1d(n_neurons))
            net.append(torch.nn.ReLU6())
            net.append(torch.nn.Dropout(dropout))
            in_shape = n_neurons
        net.append(torch.nn.Linear(in_shape, 1))
        self.fcn = torch.nn.ModuleList(net)

        
    def forward(self, X):
        '''
        Args:
            X: a dict
        '''
        feat = []
        for f in self.features:
            x_ = X[f]
            for layer in getattr(self, f):
                x_ = layer(x_)
            feat.append(x_)
        x_ = torch.cat(feat, dim=-1)
        for layer in self.fcn:
            x_ = layer(x_)
        return x_


####################               trainer                #######################
class Trainer(object):
    '''
    Torch trainer
    '''
    def __init__(self, model, train_loader, val_loader, optimizer, loss, metrics, test_loader=None, gradient_clip=10, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_n = len(train_loader.dataset)
        self.val_n = len(val_loader.dataset)
        self.test_n = len(test_loader.dataset) if test_loader else 0
        self.optimizer = optimizer
        self.gradient_clip = gradient_clip
        self.device = device
        self.log = {'loss': []}
        # loss
        if loss == 'mse':
            self.loss = loss
            self.loss_f = torch.nn.MSELoss()
        else:
            raise ValueError('Loss name not recognized {}'.format(loss))
        if isinstance(metrics, str): metrics = [metrics]
        # metrics
        self.metrics = dict()
        for m in metrics:
            if m == 'r2':
                self.metrics[m] = r2
            elif m == 'mse':
                self.metrics[m] = mse
            else:
                raise ValueError('Metrics name not recognized {}'.format(metrics))
            self.log['train_'+m] = []
            self.log['val_'+m] = []
            if self.test_loader: self.log['test_'+m] = []
        
    
    def train(self, epochs, model_path, log_path, save_freq=20):
        print('Start training on {} data, validate on {} data, test on {} data ...'.format(self.train_n, self.val_n, self.test_n), flush=True)
        for epoc in range(1, epochs+1):
            print('============================================')
            self.train_one_epoch(epoc, print_freq=200)
            self.evaluate('train', self.train_loader)
            self.evaluate('val', self.val_loader)
            if self.test_loader:
                self.evaluate('test', self.test_loader)
            # --------------- save --------------- #
            if epoc % save_freq == 0:
                print('## Saving model to {}'.format(model_path.format(epoc)))
                state = {'model': self.model, 'optimizer': self.optimizer.state_dict(), 'epoch': epoc}
                torch.save(state, model_path.format(epoc))
            hist = self.parse_log()
            hist.to_csv(log_path)
            
    def train_one_epoch(self, epoch, print_freq=None):
        self.model.train()
        loss = 0
        for i, (x, y) in enumerate(self.train_loader):
            x = {f: data.float().to(self.device) for f, data in x.items()}
            y = y.float().to(self.device)

            # ---------------------- forward --------------------- #
            pred = self.model(x).view(-1)
            batch_loss = self.loss_f(y, pred)
            loss += batch_loss.item()

            # ---------------------- backward --------------------- #
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            # torch.cuda.synchronize()
        
            # -------------- print info ---------------- #
            if print_freq and i % print_freq == 0:
                print('Epoch: [{}]/[{}]\t loss: {:.3f}'.format(epoch, i, batch_loss.item()), flush=True)
            
        # -------------- log ---------------- #
        self.log['loss'].append(loss)
    
    def evaluate(self, data, data_loader):
        assert data in ['train', 'val', 'test']
        y_true, y_pred = np.array([]), np.array([])
        self.model.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x = {f: data.float().to(self.device) for f, data in x.items()}
                yhat = self.model(x)
                y_true = np.concatenate([y_true, y.numpy()])
                y_pred = np.concatenate([y_pred, yhat.cpu().detach().numpy().reshape(-1)])
        for m in self.metrics:
            metric = self.metrics[m](y_true, y_pred)
            print('{} {}:\t{:.3f}\t'.format(data, m, metric), end='', flush=True)
            self.log[data+'_'+m].append(metric)
        print('\n', end='', flush=True)
    
    def parse_log(self):
        hist = pd.DataFrame(self.log)
        hist.index = list(range(1, len(hist.index)+1))
        return hist


####################               function                #######################
def r2(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0] ** 2

def mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))