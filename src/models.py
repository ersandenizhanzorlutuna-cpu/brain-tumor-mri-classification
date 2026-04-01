
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class BrainTumorCNN(nn.Module):
    """
    Baseline CNN for brain tumor classification
    Input:  [batch, 3, 224, 224]
    Output: [batch, 4]
    """
    def __init__(self, num_classes=4, dropout=0.5):

        super(BrainTumorCNN, self).__init__()

        # Feature extractor — 224 → 112 → 56 → 28 → 14
        self.conv_features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_features(x)  # feature extraction
        x = self.gap(x)            # global average pool
        x = self.classifier(x)     # classify
        return x
