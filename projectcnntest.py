# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:17:37 2026

@author: Sabina
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

# Set styling and random seed for reproducibility
sns.set_theme(style="whitegrid")
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Splitting the dataset
#defining the dataset as the folder
dataset = ImageFolder("catdog_data")

#defining indices
indices = list(range(len(dataset)))
#defining labels
labels = dataset.targets

#first split to create the test dataset
train_val_idx, test_idx = train_test_split(
    indices, #splitting on the indices
    test_size=0.15, #15% will be test data
    stratify=labels, #stratifying on labels
    random_state=42 
)


#defining labels again
train_val_labels = [labels[i] for i in train_val_idx]

#second split to create training and validation datasets
train_idx, val_idx = train_test_split(
    train_val_idx, #splitting on new indices
    test_size=0.15,  #15% will be validation data
    stratify=train_val_labels, #stratifying on new labels
    random_state=42
)


torch.save( # Har fjernet dine subsets, da jeg laver dem senere.
    {"train": train_idx, # Dog kan vi stadig bruge de her indexer/SMHM
     "val": val_idx,
     "test": test_idx,
    },"split.pt"
)

# Defining transformations
train_transform = transforms.Compose([
    transforms.Resize((128, 128)), # Standard valg / SMHM
    transforms.RandomHorizontalFlip(p=0.5), # Vores on-the-fly augmentation / SMHM
    transforms.ToTensor()])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()])

train_dataset = ImageFolder(
    root = "catdog_data",
    transform = train_transform)

test_dataset = ImageFolder(
    root = "catdog_data",
    transform = test_transform)

# Applying split
train_ds = Subset(train_dataset, train_idx)
val_ds = Subset(test_dataset, val_idx)
test_ds = Subset(test_dataset, test_idx)

# DataLoader

train_loader = torch.utils.data.DataLoader(
    dataset = train_ds,
    batch_size = 64,
    shuffle = True)

test_loader = torch.utils.data.DataLoader(
    dataset = test_ds,
    batch_size = 64,
    shuffle = False)

val_loader = torch.utils.data.DataLoader(
    dataset = val_ds,
    batch_size = 64,
    shuffle = False)


# Dataset preårocessing

# Output dataset dimensions and image sizes
print(f"Numver of training images : {len(train_ds):,}")
print(f"Number of test images : {len(test_ds):,}")
print(f"Number of validation images : {len(val_ds):,}")
#print(f"Images size               : {train_ds[1]} x {train_ds[2]} pixels") # relevant? /SMHM
#print(f"Images dtype & range      : {train_dataset.data.dtype}, min={train_dataset.data.min().item()}, max={train_dataset.data.max().item()}") # relevant? /SMHM

sample_tensor, sample_label = train_dataset[0]
print(f"Transformed image tensor  : {sample_tensor.shape} (Channels x Height x Width)")
print(f"Transformed dtype & range : {sample_tensor.dtype}, min={sample_tensor.min():.3f}, max={sample_tensor.max():.3f}")

# Hvis vi vil have noget visuelt ift fordelingen /SMHM
train_labels = np.array([dataset.targets[i] for i in train_idx])
val_labels = np.array([dataset.targets[i] for i in val_idx])
test_labels = np.array([dataset.targets[i] for i in test_idx])

classes = dataset.classes

train_counts = np.bincount(train_labels, minlength=2)
val_counts = np.bincount(val_labels, minlength=2)
test_counts = np.bincount(test_labels, minlength=2)

print(f"Classes: {classes}")
print(f"Train distribution: {train_counts}")
print(f"Validation distribution: {val_counts}")
print(f"Test distribution: {test_counts}")


fig, ax = plt.subplots(figsize=(10, 5))

bar_width = 0.25
x = np.arange(len(classes))

bars_train = ax.bar(x - bar_width, train_counts, width=bar_width, label='Train')
bars_val = ax.bar(x, val_counts, width=bar_width, label='Validation')
bars_test = ax.bar(x + bar_width, test_counts, width=bar_width, label='Test')

ax.set_xlabel('Class')
ax.set_ylabel('Sample Count')
ax.set_title('Cat and Dog Distribution')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

plt.tight_layout()
plt.show() # Visualisering af fordeling slutter her /SMHM

# Train a CNN
# Tager udgangspunkt i facit til exercise 4 /SMHM
class CatDogCNN(nn.Module):
    def __init__(self, dropout_rate_conv=0.25, dropout_rate_fc=0.5):
        super(CatDogCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
            kernel_size=3, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
            kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(p=dropout_rate_conv)

        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.dropout2 = nn.Dropout(p=dropout_rate_fc)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):       
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


model = CatDogCNN().to(device)
print(model)

# Training the CNN
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 25
history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': [],
    #val /SMHM
    'val_loss': [],
    'val_acc': []
}

print(f"Starting training on {device} for {num_epochs} epochs...")

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
        
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = (correct_train / total_train) * 100.0
    
    
# Validation    
    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0
    # val /SMHM
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_test_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)
            
            #val /SMHM
            running_val_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            
 
# Del jeg er i tvivl om /SMHM
# test eller val?            
    epoch_test_loss = running_test_loss / total_test
    epoch_test_acc = (correct_test / total_test) * 100.0
    epoch_val_loss = running_val_loss / total_val
    epoch_val_acc = (correct_val / total_val) * 100.0

    
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['test_loss'].append(epoch_test_loss)
    history['test_acc'].append(epoch_test_acc)
#Tilføjer lige val så vi har /SMHM
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    
    print(f"Epoch [{epoch:02d}/{num_epochs:02d}] | "
          f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
          f"Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%"
          # Tilføjer val /SMHM
          f"Val Loss: {epoch_val_loss:.4f} | "
          f"Val Acc: {epoch_val_acc:.2f}%")

print()
final_test_acc = history['test_acc'][-1]
print(f"Final Test Accuracy: {final_test_acc:.2f}%")
if final_test_acc >= 95.0:
    print("SUCCESS: Target accuracy (>= 95%) achieved!")
else:
    print("WARNING: Target accuracy of 95% was not reached.")

# Save the trained model
save_path = "catdog_cnn.pth"
torch.save(model.state_dict(), save_path)