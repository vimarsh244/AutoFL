import torch
import torch.nn as nn
from models.ResnetModel import (
    resnet50 as resnet50_custom,
    resnet34 as resnet34_custom,
    resnet18 as resnet18_custom,
)


class ResNet(nn.Module):
    def __init__(self, num_classes: int = 10, arch: str = "resnet18"):
        super(ResNet, self).__init__()

        arch = (arch or "resnet18").lower()
        if arch == "resnet18":
            self.model = resnet18_custom(num_classes=num_classes)
        elif arch == "resnet34":
            self.model = resnet34_custom(num_classes=num_classes)
        elif arch == "resnet50":
            self.model = resnet50_custom(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported ResNet architecture: {arch}")

    def forward(self, x):
        return self.model(x)


def ResNet18(num_classes=10):
    return ResNet(num_classes=num_classes, arch="resnet18")


def ResNet34(num_classes=10):
    return ResNet(num_classes=num_classes, arch="resnet34")
