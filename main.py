import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import CheXpertData
from train import CheXpert

def main(train_csv, valid_csv, n_epochs=50, batch_size=64, cdloss=True, cdloss_weight=1.2, scloss=True, scloss_weight=0.9):
    """ parameters """
    lr = 0.001
    beta1 = 0.93
    beta2 = 0.999
    n_classes = 5
    
    train_loader = DataLoader(CheXpertData(train_csv, mode='train'),
                        drop_last=True,shuffle=True,
                        batch_size=batch_size)
    val_loader = DataLoader(CheXpertData(valid_csv, mode='val'),
                        drop_last=False,shuffle=False,
                        batch_size=batch_size)

    chexpert_model = CheXpert()
    chexpert_model.train(train_loader, val_loader)
    

if __name__=='__main__':
    train_csv = 'CheXpert-v1.0-small/train.csv'
    valid_csv = 'CheXpert-v1.0-small/valid.csv'
    num_epochs = 50
    batch_size = 64 
    cdloss = True 
    cdloss_weight = 1.2
    scloss = True 
    scloss_weight = 0.9
    main(train_csv, valid_csv, num_epochs, batch_size, cdloss, cdloss_weight, scloss, scloss_weight)