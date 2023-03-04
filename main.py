from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange
import argparse

from dataset import CheXpertData
from train import CheXpert

def main(args):
    if args.data=='chexpert':
        train_csv = 'CheXpert-v1.0-small/train.csv'
        valid_csv = 'CheXpert-v1.0-small/valid.csv'
        train_data = CheXpertData(train_csv, mode='train')
        val_data = CheXpertData(valid_csv, mode='val')

    train_loader = DataLoader(train_data,
                            drop_last=True,shuffle=True,
                            batch_size=args.batch_size, num_workers=32, pin_memory=True)
    val_loader = DataLoader(val_data,
                            drop_last=True,shuffle=False,
                            batch_size=1, num_workers=32, pin_memory=True)

    if args.data=='chexpert':
        chexpert_model = CheXpert(args)
        chexpert_model.train(train_loader, val_loader, epochs=args.epochs)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_runs', default=1, type=int)
    parser.add_argument('--data', default='chexpert', type=str)
    parser.add_argument('--training_type', default='fully-supervised', type=str)
    parser.add_argument('--cdloss', default=True, type=bool)
    parser.add_argument('--cdloss_weight', default=1.2, type=int)
    parser.add_argument('--scloss', default=True, type=bool)
    parser.add_argument('--scloss_weight', default=1.2, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--beta1', default=0.93, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--n_classes', default=5, type=int)
    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--verbose', default=False, type=bool)

    args = parser.parse_args()
    main(args)