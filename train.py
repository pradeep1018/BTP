import torch
import numpy as np
import torch.nn as nn
from captum.attr import LayerGradCam, LayerAttribution
from tqdm import trange

from losses import ClassDistinctivenessLoss, SpatialCoherenceConv
from models import DenseNet121
from metrics import AUC
from utils import EarlyStopping

class CheXpert:
    def __init__(self, n_classes=5, lr = 0.001, beta1 = 0.93, beta2 = 0.999, cdloss=True, cdloss_weight=1.2, scloss=True, scloss_weight=0.9):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        self.cdloss = cdloss
        self.cdloss_weight = cdloss_weight
        self.scloss = scloss
        self.scloss_weight = scloss_weight

        self.model = DenseNet121(self.n_classes)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, beta2))
        self.sigmoid = nn.Sigmoid()
        self.gradcam = LayerGradCam(self.model, layer=self.model.features.denseblock4.denselayer16.conv2)
        self.earlystopping = EarlyStopping(patience=3, cdloss=self.cdloss, scloss=self.scloss)

        self.metric = AUC()

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

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            data_hist = np.zeros(self.n_classes)
            for target in targets:
                ind = np.where(target==1)
                data_hist[ind] += 1
            data_hist /= self.train_loader.batch_size

            ce_weights = torch.Tensor(data_hist).to(self.device)
            criterion = nn.BCEWithLogitsLoss(weight=ce_weights)

            wceloss = criterion(outputs, targets)
            if self.iteration == 0:
                self.wceloss_scale = 1 / wceloss.item()
            wceloss *= self.wceloss_scale
            loss = wceloss

            if self.cdloss or self.scloss:
                attr_classes = [torch.Tensor(self.gradcam.attribute(inputs, [i] * inputs.shape[0])) for i in range(self.n_classes)]

                if self.cdloss:
                    cdcriterion = ClassDistinctivenessLoss(device=self.device)
                    cdloss_value = cdcriterion(attr_classes)
                    if self.iteration == 0:
                        self.cdloss_scale = 1 / cdloss_value.item()
                    cdloss_value *= self.cdloss_scale
                    loss += self.cdloss_weight * cdloss_value

                if self.scloss:
                    upsampled_attr_val = [LayerAttribution.interpolate(attr, inputs[0].shape[-2:]) for
                                        attr in attr_classes]
                    sccriterion = SpatialCoherenceConv(device=self.device, kernel_size=9)
                    scloss_value = sccriterion(upsampled_attr_val, device=self.device)
                    if self.iteration == 0:
                        self.scloss_scale = 1 / scloss_value.item()
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
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)

                data_hist = np.zeros(self.n_classes)
                for target in targets:
                    ind = np.where(target==1)
                    data_hist[ind] += 1
                data_hist /= self.train_loader.batch_size

                ce_weights = torch.Tensor(data_hist).to(self.device)
                criterion = nn.BCEWithLogitsLoss(weight=ce_weights)

                wceloss = criterion(outputs, targets)
                wceloss *= self.wceloss_scale
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

        self.train_losses.append(sum(losses) / len(losses))
        self.train_aucs.append(self.metrics(y_pred, y_true))

    def train(self, train_loader, val_loader, epochs=50):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

        for epoch in trange(self.epochs):
            self.train_one_epoch()
            self.evaluate()
            
            print("Epoch {0}: Training Loss = {1}, Validation Loss = {2}, Average Training AUC = {2}, Average Validation AUC = {3}".
            format(epoch+1, self.train_losses[-1], self.val_losses[-1], 
            sum(self.train_aucs[-1])/self.n_classes, sum(self.val_aucs[-1])/self.n_classes))
            
            self.earlystopping(self.val_losses[-1], self.model)
            if self.earlystopping.early_stop:
                print('Early Stopping')
                break