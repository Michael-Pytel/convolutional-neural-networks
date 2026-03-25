import torch
import torch.nn as nn
import torchvision.models as models
from few_shot import ProtoNet

class CNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),


            nn.Conv2d(256, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
    

class ResNet18(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 10)
        )


    def forward(self, x):
        return self.model(x)

class BackboneWrapper(nn.Module):
    def __init__(self, model, model_type):
        super().__init__()
        self.model_type = model_type

        if model_type == "cnn":
            self.features = model.features
            self.pool = model.pool

        elif model_type == "resnet18":
            self.features = nn.Sequential(*list(model.model.children())[:-1])

        else:
            raise ValueError("Unsupported model type")

    def forward(self, x):
        x = self.features(x)

        if self.model_type == "cnn":
            x = self.pool(x)

        return torch.flatten(x, 1)

def get_model(name, dropout, for_protonet=False):

    if name == "cnn":
        model = CNN(dropout)
        feat_dim = 384

    elif name == "resnet18":
        model = ResNet18(dropout)
        feat_dim = 512

    else:
        raise ValueError(f"Unknown model: {name}")

    if for_protonet:
        backbone = BackboneWrapper(model, name)

        return ProtoNet(
            backbone=backbone,
            feat_dim=feat_dim,
            proj_dim=128
        )

    return model