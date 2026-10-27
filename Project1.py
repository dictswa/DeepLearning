from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

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
    transforms.Resize((150, 150)), # Har valgt random tal, skal nok justeres
    transforms.RandomHorizontalFlip(p=0.5), # Vores on-the-fly augmentation / SMHM
    transforms.ToTensor()])

test_transform = transforms.Compose([
    transforms.Resize((150, 150)), # Samme her / SMHM
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
