import torch.nn as nn
from torchvision import models


class DogBreedResNet(nn.Module):
    def __init__(self, num_classes=120, pretrained=True):
        super().__init__()

        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)