
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import BrainTumorDataset, CLASS_NAMES
from src.models import BrainTumorCNN

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
          mean=[0.485, 0.406, 0.456],
          std=[0.229, 0.224, 0.225 ]

      )

  ])

  val_transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485,0.406, 0.456 ],
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




# Training loop 
def training_one_epoch(model, loader, criterion, optimizer, device):
  model.train()
  running_loss = 0.0
  correct = 0.0
  total = 0.0 

  for images, labels in loader:
    images = images.to(device)
    labels = labels.to(device)

    # forward pass
    outputs = model(images)
    loss = criterion(outputs,labels)

    # Clear gradients,updates 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Tracking the metrics

    running_loss += loss.item()
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()
   
  epoch_loss = running_loss / len(loader)
  epoch_acc = 100.0 * correct / total
  return epoch_loss, epoch_acc

# Evaluation loop 
def evaluation(model, loader, criterion, device):
  model.eval()
  running_loss = 0.0
  correct = 0.0
  total= 0.0 

  with torch.no_grad():
    for images, labels in loader:

      images = images.to(device)
      labels = images.to(device)

      outputs = model(images)
      loss = criterion(outputs, labels)
      
      running_loss += loss.item()
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total 
    return epoch_loss, epoch_acc

# Main Training loop 

def train(model, train_loader, val_loader,
          num_epochs = 25, learning_rate=1e-3,
          save_path ='/content/repo/results/baseline_cnn.pth'):
  
  device = get_device()
  model = model.to(device)

  # loss function, optimizer 

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # learning rate scheduler 

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor = 0.5, patience=3
  )
  
  history = {
      "train_loss":[], "val_loss":[],
      "train_acc":[], "val_acc":[]
  }

  best_val_loss = float('inf')
  patience_counter = 0 
  early_stopping_patience = 7

  print(f"\nTraining for {num_epochs} epochs ")
  print("="*60)

  for epoch in range(num_epochs):

    train_loss, train_acc = training_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluation(
        model, val_loader, criterion, device
    )

    # step scheduler 
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)


    print(f"Epochs [{epoch+1:02d}/{num_epochs}]"
    f"Train loss:{train_loss:.4f}"
    f"Train acc: {train_acc:.1f}%|"
    f"Val loss: {val_loss:.4f} |"
    f"Val acc: {val_acc:.1f}%")

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      torch.save(model.state_dict(), save_path)
      print(f"Best model is saved (val_loss): {val_loss:.4f}")
    
    # Early stopping mechanism
    else :
      patience_counter += 1
      if patience_counter >= early_stopping_patience:
        print(f"Early stoppingg triggered (epoch: {epoch+1})")
        break

  return history
  

# Plot curves history

def plot_history(history, save_path = '/content/repo/results/'):

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  # Loss curves 
  ax1.plot(history['train_loss'], label='train loss',
           color='#5DCAA5', linewidth=2)
  ax1.plot(history['val_loss'], label= 'val loss',
           color='#D85A30', linewidth=2)
  
  ax1.set_title('loss curve', fontweight='bold')
  ax1.set_xlabel('epoch', fontweight='bold')
  ax1.set_ylabel('loss', fontweight ='bold')
  ax1.legend()
  ax1.grid(alpha=0.3)
  
  # Accuracy curves

  ax2.plot(history['train_acc'], label='train acc',
          color='#5DCAA5', linewidth=2 )
  ax2.plot(history['val_acc'], label='val acc',
           color='#D85A30', linewidth=2)
  
  ax2.set_title('accuracy curve', fontweight='bold')
  ax2.set_xlabel('epoch', fontweight='bold')
  ax2.set_ylabel('accuracy', fontweight='bold')
  ax2.legend()
  ax2.grid(alpha=0.3)

  plt.tight_layout()
  plt.savefig(f'{save_path}/baseline_training_curves.png',
              dpi=150, bbox_inches='tight')
  plt.show()
  print("Training curves saved!")


















    

# Training loop 
def training_one_epoch(model, loader, criterion, optimizer, device):
  model.train()
  running_loss = 0.0
  correct = 0.0
  total = 0.0 

  for images, labels in loader:
    images = images.to(device)
    labels = labels.to(device)

    # forward pass
    outputs = model(images)
    loss = criterion(outputs,labels)

    # Clear gradients,updates 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Tracking the metrics

    running_loss += loss.item()
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()
   
  epoch_loss = running_loss / len(loader)
  epoch_acc = 100.0 * correct / total
  return epoch_loss, epoch_acc

# Evaluation loop 
def evaluation(model, loader, criterion, device):
  model.eval()
  running_loss = 0.0
  correct = 0.0
  total= 0.0 

  with torch.no_grad():
    for images, labels in loader:

      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      loss = criterion(outputs, labels)
      
      running_loss += loss.item()
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total 
    return epoch_loss, epoch_acc

# Main Training loop 

def train(model, train_loader, val_loader,
          num_epochs = 25, learning_rate=1e-3,
          save_path ='/content/repo/results/baseline_cnn.pth'):
  
  device = get_device()
  model = model.to(device)

  # loss function, optimizer 

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # learning rate scheduler 

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor = 0.5, patience=3
  )
  
  history = {
      "train_loss":[], "val_loss":[],
      "train_acc":[], "val_acc":[]
  }

  best_val_loss = float('inf')
  patience_counter = 0 
  early_stopping_patience = 7

  print(f"\nTraining for {num_epochs} epochs ")
  print("="*60)

  for epoch in range(num_epochs):

    train_loss, train_acc = training_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc = evaluation(
        model, val_loader, criterion, device
    )

    # step scheduler 
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)


    print(f"Epochs [{epoch+1:02d}/{num_epochs}]"
    f"Train loss:{train_loss:.4f}"
    f"Train acc: {train_acc:.1f}%|"
    f"Val loss: {val_loss:.4f} |"
    f"Val acc: {val_acc:.1f}%")

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      torch.save(model.state_dict(), save_path)
      print(f"Best model is saved (val_loss): {val_loss:.4f}")
    
    # Early stopping mechanism
    else :
      patience_counter += 1
      if patience_counter >= early_stopping_patience:
        print(f"Early stoppingg triggered (epoch: {epoch+1})")
        break

  return history
  

# Plot curves history

def plot_history(history, save_path = '/content/repo/results/'):

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  # Loss curves 
  ax1.plot(history['train_loss'], label='train loss',
           color='#5DCAA5', linewidth=2)
  ax1.plot(history['val_loss'], label= 'val loss',
           color='#D85A30', linewidth=2)
  
  ax1.set_title('loss curve', fontweight='bold')
  ax1.set_xlabel('epoch', fontweight='bold')
  ax1.set_ylabel('loss', fontweight ='bold')
  ax1.legend()
  ax1.grid(alpha=0.3)
  
  # Accuracy curves

  ax2.plot(history['train_acc'], label='train acc',
          color='#5DCAA5', linewidth=2 )
  ax2.plot(history['val_acc'], label='val acc',
           color='#D85A30', linewidth=2)
  
  ax2.set_title('accuracy curve', fontweight='bold')
  ax2.set_xlabel('epoch', fontweight='bold')
  ax2.set_ylabel('accuracy', fontweight='bold')
  ax2.legend()
  ax2.grid(alpha=0.3)

  plt.tight_layout()
  plt.savefig(f'{save_path}/baseline_training_curves.png',
              dpi=150, bbox_inches='tight')
  plt.show()
  print("Training curves saved!")


  
