import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, features=[32, 64, 128, 256, 512]):
        super(UNetClassifier, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(features[-1]*2, num_classes)

    def forward(self, x):
        for down in self.downs:
            x = down(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class FPFNPenalizedCELoss(nn.Module):
    def __init__(self, fn_weight=3.0, fp_weight=3.0):
        super().__init__()
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets):
        loss = self.ce(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        fn_mask = (preds == 0) & (targets == 1)
        fp_mask = (preds == 1) & (targets == 0)
        loss[fn_mask] *= self.fn_weight
        loss[fp_mask] *= self.fp_weight
        return loss.mean()
