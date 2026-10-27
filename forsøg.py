# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:33:54 2026

@author: dicte
"""

from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split
import torch

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

#subsetting the full dataset into the three using their indices
train_ds = Subset(dataset, train_idx)
val_ds = Subset(dataset, val_idx)
test_ds = Subset(dataset, test_idx)

torch.save(
    {"train": train_ds.indices,
     "val": val_ds.indices,
     "test": test_ds.indices,
    },"split.pt"
)

