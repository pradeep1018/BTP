import torch
import numpy as np
import torch.nn as nn
<<<<<<< HEAD
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
=======
>>>>>>> b4e3bcee0a21677d66cb76ff5f2fb9e9e71a61b9
from captum.attr import LayerGradCam, LayerAttribution
from tqdm import tqdm, trange
from torchvision.models import densenet121

from losses import ClassDistinctivenessLoss, SpatialCoherenceConv
from metrics import AUC
<<<<<<< HEAD
from utils import EarlyStopping

class CheXpert:
    def __init__(self, n_classes=5, lr = 0.001, beta1 = 0.93, beta2 = 0.999, cdloss=True, cdloss_weight=1.2, scloss=True, scloss_weight=0.9):
        self.device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
=======

class CheXpert:
    def __init__(self, n_classes=5, lr = 0.001, beta1 = 0.93, beta2 = 0.999, cdloss=True, cdloss_weight=1.2, scloss=True, scloss_weight=0.9):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
>>>>>>> b4e3bcee0a21677d66cb76ff5f2fb9e9e71a61b9
        self.n_classes = n_classes
        self.cdloss = cdloss
        self.cdloss_weight = cdloss_weight
        self.scloss = scloss
        self.scloss_weight = scloss_weight

        self.model = densenet121(weights='DEFAULT')
        self.model.classifier = nn.Linear(1024, n_classes)
<<<<<<< HEAD
        #for param in self.model.features.parameters():
            #param.requires_grad = False
        #self.model = nn.DataParallel(self.model, [0,1,4,5])
        self.model = self.model.to(self.device)
        
=======
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model = self.model.to(self.device)
>>>>>>> b4e3bcee0a21677d66cb76ff5f2fb9e9e71a61b9

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)#, betas=(beta1, beta2))
        self.sigmoid = nn.Sigmoid()
        self.gradcam = LayerGradCam(self.model, layer=self.model.features.denseblock4.denselayer16.conv2)

        self.metrics = AUC()
<<<<<<< HEAD
        self.early_stopping = EarlyStopping(patience=3, verbose=True)
=======
>>>>>>> b4e3bcee0a21677d66cb76ff5f2fb9e9e71a61b9

        self.iteration = 0
        self.wceloss_scale = 1
        self.scloss_scale = 1
        self.cdloss_scale = 1

        self.train_losses = []
        self.val_losses = []
        self.train_aucs = []
        self.val_aucs = []

    def train_one_epoch(self):
        self.model.train()

        losses = []
        y_true = torch.tensor([]).to(self.device)
        y_pred = torch.tensor([]).to(self.device)

        for inputs, targets in tqdm(self.train_loader):
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            data_hist = np.zeros(self.n_classes)
            for target in targets:
                ind = np.where(target==1)
                data_hist[ind] += 1
            data_hist /= self.train_loader.batch_size

            targets = targets.to(self.device)
            ce_weights = torch.Tensor(data_hist).to(self.device)
            criterion = nn.BCEWithLogitsLoss(weight=ce_weights)

            wceloss = criterion(outputs, targets)
            """
            if self.iteration == 0:
                self.wceloss_scale = 1 / (wceloss.item())
            wceloss *= self.wceloss_scale
            """
            loss = wceloss

            if self.cdloss or self.scloss:
                attr_classes = [torch.Tensor(self.gradcam.attribute(inputs, [i] * inputs.shape[0])) for i in range(self.n_classes)]

                if self.cdloss:
                    cdcriterion = ClassDistinctivenessLoss(device=self.device)
                    cdloss_value = cdcriterion(attr_classes)
                    if self.iteration == 0:
                        self.cdloss_scale = 1 / (cdloss_value.item())
                    cdloss_value *= self.cdloss_scale
                    loss += self.cdloss_weight * cdloss_value

                if self.scloss:
                    upsampled_attr_val = [LayerAttribution.interpolate(attr, inputs[0].shape[-2:]) for
                                        attr in attr_classes]
                    sccriterion = SpatialCoherenceConv(device=self.device, kernel_size=9)
                    scloss_value = sccriterion(upsampled_attr_val, device=self.device)
                    if self.iteration == 0:
                        self.scloss_scale = 1 / (scloss_value.item())
                    scloss_value *= self.scloss_scale
                    loss += self.scloss_weight * scloss_value
            
            predictions = self.sigmoid(outputs).to(self.device)

            y_pred = torch.cat((y_pred, predictions), 0)
            y_true = torch.cat((y_true, targets), 0)
            losses.append(loss.item())
            
            loss.backward()
            self.optimizer.step()

            self.iteration += 1

        self.train_losses.append(sum(losses) / len(losses))
        self.train_aucs.append(self.metrics(y_pred, y_true))

    def evaluate(self):
        self.model.eval()

        losses = []
        y_true = torch.tensor([]).to(self.device)
        y_pred = torch.tensor([]).to(self.device)

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                data_hist = np.zeros(self.n_classes)
                for target in targets:
                    ind = np.where(target==1)
                    data_hist[ind] += 1
                data_hist /= self.train_loader.batch_size

                targets = targets.to(self.device)
                ce_weights = torch.Tensor(data_hist).to(self.device)
                criterion = nn.BCEWithLogitsLoss(weight=ce_weights)

                wceloss = criterion(outputs, targets)
                #wceloss *= self.wceloss_scale
                loss = wceloss

                if self.cdloss or self.scloss:
                    attr_classes = [torch.Tensor(self.gradcam.attribute(inputs, [i] * inputs.shape[0])) for i in range(self.n_classes)]

                    if self.cdloss:
                        cdcriterion = ClassDistinctivenessLoss(device=self.device)
                        cdloss_value = cdcriterion(attr_classes)
                        cdloss_value *= self.cdloss_scale
                        loss += self.cdloss_weight * cdloss_value

                    if self.scloss:
                        upsampled_attr_val = [LayerAttribution.interpolate(attr, inputs[0].shape[-2:]) for
                                            attr in attr_classes]
                        sccriterion = SpatialCoherenceConv(device=self.device, kernel_size=9)
                        scloss_value = sccriterion(upsampled_attr_val, device=self.device)
                        scloss_value *= self.scloss_scale
                        loss += self.scloss_weight * scloss_value
                
                predictions = self.sigmoid(outputs).to(self.device)

                y_pred = torch.cat((y_pred, predictions), 0)
                y_true = torch.cat((y_true, targets), 0)
                losses.append(loss.item())

        self.val_losses.append(sum(losses) / len(losses))
        self.val_aucs.append(self.metrics(y_pred, y_true))

    def train(self, train_loader, val_loader, epochs=50):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

        for epoch in trange(self.epochs):
            self.train_one_epoch()
            self.evaluate()
            
            print("Epoch {0}: Training Loss = {1}, Validation Loss = {2}, Average Training AUC = {3}, Average Validation AUC = {4}".
            format(epoch+1, self.train_losses[-1], self.val_losses[-1], 
            sum(self.train_aucs[-1])/self.n_classes, sum(self.val_aucs[-1])/self.n_classes))

<<<<<<< HEAD
            self.early_stopping(self.val_losses[-1], self.model, epoch)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        return self.val_losses, self.train_aucs[self.early_stopping.best_epoch], self.val_aucs[self.early_stopping.best_epoch]
=======
        return self.val_losses, self.train_aucs[-1], self.val_aucs[-1]
>>>>>>> b4e3bcee0a21677d66cb76ff5f2fb9e9e71a61b9
