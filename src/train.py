
import os
import sys
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import BrainTumorDataset, CLASS_NAMES
from src.models import BrainTumorCNN


def set_seed(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic=True
  torch.backends.cudnn.benchmark=False
  print(f"set seed {seed}")
  
# Device #
def get_device():
  if torch.cuda.is_available():
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    return torch.device('cuda')

  print("GPU used")
  return torch.device('cpu')

# Transformer #

def get_transformer():
  train_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(degrees=15),
      transforms.ColorJitter(brightness=0.3, contrast=0.3),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225 ]

      )

  ])

  val_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485,0.456, 0.406 ],
          std=[0.229,0.224, 0.225 ]
      )

  ])

  return train_transform, val_transform

def get_dataloaders(data_dir, batch_size=32):

  train_transform, val_transform = get_transformer()

  full_dataset = BrainTumorDataset(
      root_dir = f'{data_dir}/Training',
      transform = None
  )

  all_paths = [s[0]for s in full_dataset.samples]
  all_labels = [s[1]for s in full_dataset.samples]


  train_paths, val_paths, train_labels, val_labels = train_test_split(
      all_paths,
      all_labels,
      test_size=0.3,
      stratify=all_labels,
      random_state=42

  )
  val_paths, test_paths, val_labels, test_labels = train_test_split(
      val_paths,
      val_labels,
      test_size=0.5,
      stratify=val_labels,
      random_state=42
  )

  train_dataset = BrainTumorDataset(
    root_dir = f'{data_dir}/Training',
    transform = train_transform
  )

  train_dataset.samples = list(zip(train_paths, train_labels))

  test_dataset = BrainTumorDataset(
      root_dir = f'{data_dir}/Training',
      transform = val_transform
  )

  test_dataset.samples = list(zip(test_paths, test_labels))

  val_dataset = BrainTumorDataset(
      root_dir = f'{data_dir}/Training',
      transform = val_transform
  )

  val_dataset.samples = list(zip(val_paths, val_labels))

  # Dataloaders #

  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=2
  )

  val_loader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=2
  )

  test_loader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=2
  )


  print(f"Train:{len(train_dataset)} |" f"Val: {len(val_dataset)} |" f"Test: {len(test_dataset)}" )

  return train_loader, test_loader, val_loader



