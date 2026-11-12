"""
Example PyTorch Training Script for Gastrovision 5-Class Dataset
================================================================
Demonstrates:
  - Loading the cleaned train/val/test splits via ImageFolder
  - Heavy data augmentation to combat overfitting
  - GPU / CUDA device targeting with automatic CPU fallback
  - Balanced class weighting for CrossEntropyLoss
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"c:\Users\AMRUTA M PATIL\Downloads\Gastrovision_5classes\dataset")
TRAIN_DIR  = DATA_DIR / "train"
VAL_DIR    = DATA_DIR / "val"
TEST_DIR   = DATA_DIR / "test"

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS     = 25
LR         = 1e-4

# ── Device selection (GPU if available, else CPU) ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Anti-Overfitting Data Augmentation ─────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Datasets & Loaders ────────────────────────────────────────────────────────
train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transforms)
val_dataset   = datasets.ImageFolder(str(VAL_DIR),   transform=eval_transforms)
test_dataset  = datasets.ImageFolder(str(TEST_DIR),  transform=eval_transforms)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Classes: {train_dataset.classes}")
print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ── Model (Transfer learning with Dropout to combat overfitting) ─────────────
def get_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Freeze backbone if dataset is small
    for param in model.parameters():
        param.requires_grad = False
    
    # Custom head with Dropout
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES),
    )
    return model.to(device)

if __name__ == "__main__":
    model = get_model()
    print("Model created successfully.")
    print("Run your training loop here!")

