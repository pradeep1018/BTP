from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

from dataset import CheXpertData
from train import CheXpert

def main(train_csv, valid_csv, n_epochs=50, batch_size=32, cdloss=False, cdloss_weight=1.2, scloss=False, scloss_weight=0.9):
    """ parameters """
    lr = 0.001
    beta1 = 0.93
    beta2 = 0.999
    n_classes = 5

    train_data = CheXpertData(train_csv, mode='train')
    train_loader = DataLoader(train_data,
                        drop_last=True,shuffle=True,
                        batch_size=batch_size, num_workers=32, pin_memory=True)
    val_data = CheXpertData(valid_csv, mode='val')
    val_loader = DataLoader(val_data,
                        drop_last=True,shuffle=False,
                        batch_size=batch_size, num_workers=32, pin_memory=True)

    chexpert_model = CheXpert(n_classes=n_classes, lr=lr, beta1=beta1, beta2=beta2, 
                              cdloss=cdloss, cdloss_weight=cdloss_weight, scloss=scloss, scloss_weight=scloss_weight)
    loss, train_auc, val_auc = chexpert_model.train(train_loader, val_loader, epochs=n_epochs)
    return loss, train_auc, val_auc

if __name__=='__main__':
    train_csv = 'CheXpert-v1.0-small/train.csv'
    valid_csv = 'CheXpert-v1.0-small/valid.csv'
    num_epochs = 20
    batch_size = 32

    loss_cel, train_auc_cel, val_auc_cel = main(train_csv, valid_csv, num_epochs, batch_size)

    #loss_cl, train_auc_cl, val_auc_cl = main(train_csv, valid_csv, num_epochs, batch_size, 
    #cdloss=True, cdloss_weight=1.2, scloss=True, scloss_weight=0.9)

    plt.plot(loss_cel, label='crossentropy loss')
    #plt.plot(loss_cl, label='custom loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('results/chexpert_val_loss_plot.png')

    train_auc_cel = train_auc_cel.tolist()
    train_auc_cel.append(sum(train_auc_cel)/len(train_auc_cel))
    #train_auc_cl = train_auc_cl.tolist()
    #train_auc_cl.append(sum(train_auc_cl)/len(train_auc_cl))
    train_auc_df = pd.DataFrame(columns=['Atelectasis','Cardiomegaly','Consolidation','Edema','Plural Effusion','Mean'])
    train_auc_df.loc['train auc(crossentropy loss)'] = train_auc_cel
    #train_auc_df.loc['train auc(custom loss)'] = train_auc_cl
    train_auc_df.to_csv('results/chexpert_train_auc.csv')

    val_auc_cel = val_auc_cel.tolist()
    val_auc_cel.append(sum(val_auc_cel)/len(val_auc_cel))
    #val_auc_cl = val_auc_cl.tolist()
    #val_auc_cl.append(sum(val_auc_cl)/len(val_auc_cl))
    val_auc_df = pd.DataFrame(columns=['Atelectasis','Cardiomegaly','Consolidation','Edema','Plural Effusion','Mean'])
    val_auc_df.loc['val auc(crossentropy loss)'] = val_auc_cel
    #val_auc_df.loc['val auc(custom loss)'] = val_auc_cl
    val_auc_df.to_csv('results/chexpert_val_auc.csv')