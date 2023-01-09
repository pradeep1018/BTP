import torch.nn as nn
from torchvision.models import densenet121

class DenseNet121(nn.Module):
    def __init__(self, n_classes=5):
        super(DenseNet121, self).__init__()
        pretrained_model = densenet121(weights='DEFAULT')
        self.features = pretrained_model.features
        self.classifier = nn.Linear(in_features = pretrained_model.classifier.in_features, out_features = n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x