
import torch
import torch.nn as nn
from torchvision import models

class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B3 fine-tuned for brain tumor classification
    Strategy: freeze backbone → train head → unfreeze top layers
    """
    def __init__(self, num_classes=4, dropout=0.4):
      super(EfficientNetClassifier, self).__init__()


      # Load Pretrained model

      self.backbone = models.efficientnet_b3(
          weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
      )

      in_features = self.backbone.classifier[1].in_features # Get the number of features before classifier

      # replacing the classifier head with our own

      self.backbone.classifier = nn.Sequential(
          nn.Dropout(p=dropout),
          nn.Linear(in_features, 256),
          nn.ReLU(inplace=True),
          nn.Dropout(dropout),
          nn.Linear(256, num_classes)


      )

    def freeze_backbone(self):
      """ Freeze all layers except our classifier head """

      for param in self.backbone.parameters():
        param.requires_grad = False

      # Unfreeze classifier head
      for param in self.backbone.classifier.parameters():
        param.requires_grad = True
      print("Backbone freezed -- classifier head only!")


    def unfreeze_backbone(self, num_blocks=3):
      """ Unfreeze last N blocks for fine tuning """

      for param in self.backbone.parameters():
        param.requires_grad = False

      # unfreeze last num_blocks for fine tuning

      blocks = list(self.backbone.children())

      for block in blocks[-num_blocks:]:
        for param in block.parameters():
          param.requires_grad = True

          print(f"Last {num_blocks} frozen blocks are unfrozen")

      for param in self.backbone.classifier.parameters():
        param.requires_grad = True

      trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

      print(f"Unfroze top {num_blocks} blocks"
      f" trianable {trainable} parameters" )

    def forward(self, x):
      return self.backbone(x)
