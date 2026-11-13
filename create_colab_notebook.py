"""
Creates the complete, self-contained Google Colab notebook (.ipynb)
for Gastrointestinal Disease Image Classification with DenseNet121 + CBAM + FPN.
"""

import json

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "name": "GastroVision_DenseNet121_CBAM_FPN_Complete.ipynb",
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "accelerator": "GPU"
    },
    "cells": []
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(code):
    nb["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in code.split("\n")]
    })

# --- Notebook Header ---
add_md("""# Gastrointestinal Disease Classification Using DenseNet121, CBAM & FPN
### Deep Learning & Medical Computer Vision Research Pipeline (Free T4 GPU)
**Study Overview:**
This notebook implements, trains, evaluates, and compares two deep learning architectures for Computer-Aided Diagnosis (CAD) on the **GastroVision** 5-class endoscopic dataset:
1. **Baseline Model:** Pretrained DenseNet121 Backbone &rarr; Global Average Pooling &rarr; Classification Head &rarr; Softmax
2. **Full Proposed Model (Flowchart):** Input Image &rarr; CLAHE Preprocessing &rarr; DenseNet121 Multi-Level Backbone &rarr; CBAM Attention Modules &rarr; Multi-Scale Feature Fusion (FPN) &rarr; Classification Head &rarr; Softmax

**Experimental Safeguards:**
- Strict deterministic 70% / 15% / 15% train/val/test split with fixed seed 42.
- Anti-overfitting controls: CLAHE on L-channel of LAB space, Differential LRs, Backbone Freezing (Epochs 1-8), AdamW, ReduceLROnPlateau, and Early Stopping (patience=12).
- Scientific evaluation suite: Loss/Accuracy curves, Confusion Matrix verification (sum=678), One-vs-Rest ROC-AUC curves, Precision-Recall curves, per-class F1 metrics, and automated Google Drive checkpointing.""")

# --- Cell 1: Setup & Dependencies ---
add_md("## Step 1: Install & Import Dependencies, Mount Google Drive")
add_code("""# Mount Google Drive
from google.colab import drive
import os
drive.mount('/content/drive')

# Create project output directory in Google Drive
SAVE_DIR = '/content/drive/MyDrive/Gastrovision_Project_Results'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Results will be permanently saved to: {SAVE_DIR}")

# Standard imports
import sys, random, copy, time, zipfile, shutil
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from google.colab import files

# Device configuration (GPU on Colab T4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Deterministic Seed
SEED = 42
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
print(f"Deterministic seed set to {SEED}")""")

# --- Cell 2: Dataset Extraction & Deterministic Splitting ---
add_md("## Step 2: Locate Dataset (Auto-Search in Drive / GitHub Fallback) & Split")
add_code("""# ==============================================================================
# CELL 2: SMART DATASET FINDER & UNZIP (HANDLES DRIVE FOLDERS & RENAMES)
# ==============================================================================
import os, glob, zipfile, shutil, random

RAW_DATA_DIR = '/content/raw_data'
if os.path.exists(RAW_DATA_DIR):
    shutil.rmtree(RAW_DATA_DIR)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# 1. Search for dataset zip in Google Drive
ZIP_PATH = None
candidate_names = [
    '/content/drive/MyDrive/Gastrovision_5classes.zip',
    '/content/drive/MyDrive/Gastrovision_5classes (2).zip',
    '/content/drive/MyDrive/Gastrovision_5classes (1).zip',
    '/content/drive/MyDrive/Gastrovision.zip',
]

for p in candidate_names:
    if os.path.exists(p):
        ZIP_PATH = p
        break

# If not in the root of MyDrive, scan subfolders of MyDrive
if ZIP_PATH is None and os.path.exists('/content/drive/MyDrive'):
    print("Searching for Gastrovision zip in Google Drive subfolders...")
    for root, dirs, files in os.walk('/content/drive/MyDrive'):
        for f in files:
            if f.lower().endswith('.zip') and ('gastro' in f.lower() or '5class' in f.lower()):
                ZIP_PATH = os.path.join(root, f)
                break
        if ZIP_PATH:
            break

# 2. Extract or Fallback
if ZIP_PATH and os.path.exists(ZIP_PATH):
    print(f"Found dataset zip at: {ZIP_PATH}")
    print(f"Extracting {os.path.basename(ZIP_PATH)} ...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)
    print("Extraction complete!")
else:
    print("Could not find a Gastrovision .zip directly in Google Drive.")
    print("Showing all .zip files currently detected in your MyDrive:")
    all_zips = glob.glob('/content/drive/MyDrive/**/*.zip', recursive=True)
    if all_zips:
        for z in all_zips[:10]:
            print(f"  - {z}")
    else:
        print("  (No .zip files found in MyDrive)")
    
    # Fallback to cloning directly from your GitHub repository (already pushed!)
    print("\\n>>> FALLBACK: Fetching dataset directly from your GitHub repository...")
    REPO_URL = "https://github.com/Amruta09-eng/GastroVision-GI-Classification-Using-DenseNet121-CBAM-FPN.git"
    !git clone --depth 1 {REPO_URL} /content/github_repo
    if os.path.exists('/content/github_repo/dataset'):
        print("Successfully loaded dataset directly from GitHub!")
        RAW_DATA_DIR = '/content/github_repo/dataset'
    else:
        raise FileNotFoundError(
            "Please upload Gastrovision_5classes.zip to your Google Drive root (MyDrive) and re-run this cell."
        )

# 3. Handle potential nested directory structure inside raw_data
if 'github_repo' in RAW_DATA_DIR:
    # If pulled from GitHub, dataset is already partitioned in train/val/test
    candidate_splits = os.path.join(RAW_DATA_DIR, 'train')
    if os.path.exists(candidate_splits):
        BASE_CLASSES_DIR = candidate_splits
    else:
        BASE_CLASSES_DIR = RAW_DATA_DIR
else:
    subdirs = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    if len(subdirs) == 1 and ('Gastrovision' in subdirs[0] or 'classes' in subdirs[0] or 'raw' in subdirs[0]):
        BASE_CLASSES_DIR = os.path.join(RAW_DATA_DIR, subdirs[0])
    else:
        BASE_CLASSES_DIR = RAW_DATA_DIR

CLASSES = sorted([
    d for d in os.listdir(BASE_CLASSES_DIR)
    if os.path.isdir(os.path.join(BASE_CLASSES_DIR, d)) and not d.startswith(('.', '_'))
])
print(f"Found {len(CLASSES)} target classes: {CLASSES}")

# 4. Deterministic 70% / 15% / 15% Split
SPLIT_DIR = '/content/dataset_split'
if os.path.exists(SPLIT_DIR):
    shutil.rmtree(SPLIT_DIR)

split_records = {'train': [], 'val': [], 'test': []}
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Check if dataset is already pre-split in GitHub repo format
if os.path.exists('/content/github_repo/dataset/test'):
    print("Using pre-stratified splits from repository...")
    for s_name in ['train', 'val', 'test']:
        s_dir = os.path.join('/content/github_repo/dataset', s_name)
        for c_idx, c_name in enumerate(CLASSES):
            c_dir = os.path.join(s_dir, c_name)
            if os.path.exists(c_dir):
                for fname in os.listdir(c_dir):
                    if fname.lower().endswith(IMG_EXTENSIONS):
                        split_records[s_name].append((os.path.join(c_dir, fname), c_idx))
else:
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_folder = os.path.join(BASE_CLASSES_DIR, cls_name)
        files_list = sorted([
            f for f in os.listdir(cls_folder)
            if f.lower().endswith(IMG_EXTENSIONS)
        ])
        
        # Shuffle deterministically per class
        rng = random.Random(SEED + cls_idx)
        rng.shuffle(files_list)
        
        n_total = len(files_list)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        
        train_files = files_list[:n_train]
        val_files = files_list[n_train:n_train + n_val]
        test_files = files_list[n_train + n_val:]
        
        for split_name, f_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
            dst_folder = os.path.join(SPLIT_DIR, split_name, cls_name)
            os.makedirs(dst_folder, exist_ok=True)
            for fname in f_list:
                src = os.path.join(cls_folder, fname)
                dst = os.path.join(dst_folder, fname)
                shutil.copy2(src, dst)
                split_records[split_name].append((dst, cls_idx))
                
        print(f"Class '{cls_name[:32]:32s}': Total={n_total:4d} | Train={len(train_files):4d} | Val={len(val_files):3d} | Test={len(test_files):3d}")

print("-" * 75)
print(f"TOTALS: Train={len(split_records['train'])} | Val={len(split_records['val'])} | Test={len(split_records['test'])}")
print("Dataset successfully ready and verified!")""")

# --- Cell 3: Preprocessing & CLAHE Transform ---
add_md("## Step 3: Preprocessing (Resize + CLAHE on L Channel of LAB)")
add_code("""import cv2
import numpy as np
from PIL import Image
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CLAHETransform:
    \"\"\"
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the L-channel of the LAB color space.
    Enhances mucosal texture and microvascular structures without color shift.
    \"\"\"
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def __call__(self, img_pil):
        # Convert PIL (RGB) to OpenCV (BGR)
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Convert BGR to LAB
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)
        
        # Merge back and convert to RGB PIL
        lab_clahe = cv2.merge((l_clahe, a, b))
        bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        rgb_clahe = cv2.cvtColor(bgr_clahe, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_clahe)

# Data Transforms
IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom PyTorch Dataset
class GastroDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = GastroDataset(split_records['train'], transform=train_transform)
val_dataset   = GastroDataset(split_records['val'],   transform=eval_transform)
test_dataset  = GastroDataset(split_records['test'],  transform=eval_transform)

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Compute Class Weights for Imbalance (Loss weighting)
train_labels = [label for _, label in split_records['train']]
class_counts = Counter(train_labels)
total_samples = len(train_labels)
num_classes = len(CLASSES)

class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class counts in training: {[class_counts[i] for i in range(num_classes)]}")
print(f"Calculated class weights: {[round(w, 3) for w in class_weights]}")""")

# --- Cell 4: Model Architectures ---
add_md("## Step 4: Model Architectures (Baseline vs Full Model with CBAM + FPN)")
add_code("""# ==========================================
# 1. BASELINE MODEL: DenseNet121 Standard
# ==========================================
class DenseNet121_Baseline(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.densenet(x)


# ==========================================
# 2. CBAM ATTENTION MODULES
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ==========================================
# 3. FULL MODEL: DenseNet121 + CBAM + FPN
# ==========================================
class DenseNet121_CBAM_FPN(nn.Module):
    \"\"\"
    Flowchart Implementation:
    Input -> Preprocessing -> DenseNet121 Backbone (extract 3 scales)
          -> CBAM Attention (on each scale)
          -> Multi-Scale Feature Fusion (FPN top-down + lateral convs)
          -> Global Average Pooling & Concatenation (768-D)
          -> Fully Connected Classifier -> Softmax
    \"\"\"
    def __init__(self, num_classes=5, fpn_channels=256):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        features = densenet.features

        # Scale 1: up to denseblock2 (output channels = 512)
        self.stage1 = nn.Sequential(
            features.conv0, features.norm0, features.relu0, features.pool0,
            features.denseblock1, features.transition1,
            features.denseblock2
        )
        # Scale 2: transition2 + denseblock3 (output channels = 1024)
        self.stage2 = nn.Sequential(
            features.transition2,
            features.denseblock3
        )
        # Scale 3: transition3 + denseblock4 + norm5 (output channels = 1024)
        self.stage3 = nn.Sequential(
            features.transition3,
            features.denseblock4,
            features.norm5
        )

        # CBAM Attention on each multi-scale tap
        self.cbam1 = CBAM(512, ratio=16)
        self.cbam2 = CBAM(1024, ratio=16)
        self.cbam3 = CBAM(1024, ratio=16)

        # FPN Lateral 1x1 convs (bring all to fpn_channels = 256)
        self.lat3 = nn.Conv2d(1024, fpn_channels, kernel_size=1)
        self.lat2 = nn.Conv2d(1024, fpn_channels, kernel_size=1)
        self.lat1 = nn.Conv2d(512,  fpn_channels, kernel_size=1)

        # FPN 3x3 Smooth Convolutions
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)

        # Global Pooling and Final Classifier Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fpn_channels * 3, 256), # 256*3 = 768
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 1. Feature extraction through DenseNet stages
        c1 = self.stage1(x)  # (B, 512, H/8, W/8)
        c2 = self.stage2(c1) # (B, 1024, H/16, W/16)
        c3 = self.stage3(c2) # (B, 1024, H/32, W/32)

        # 2. CBAM Attention applied to each feature map
        a1 = self.cbam1(c1)
        a2 = self.cbam2(c2)
        a3 = self.cbam3(c3)

        # 3. FPN Top-Down Pathway with Lateral Connections
        p3 = self.lat3(a3)
        p2 = self.lat2(a2) + F.interpolate(p3, size=a2.shape[-2:], mode='nearest')
        p1 = self.lat1(a1) + F.interpolate(p2, size=a1.shape[-2:], mode='nearest')

        # 4. Smooth Convolutions
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        # 5. Global Average Pooling & Multi-Scale Concatenation
        g3 = torch.flatten(self.gap(p3), 1) # (B, 256)
        g2 = torch.flatten(self.gap(p2), 1) # (B, 256)
        g1 = torch.flatten(self.gap(p1), 1) # (B, 256)
        fused = torch.cat([g1, g2, g3], dim=1) # (B, 768)

        # 6. Final Fully Connected Layer -> Logits
        out = self.classifier(fused)
        return out

# Parameter count check
m_base = DenseNet121_Baseline(num_classes=5)
m_full = DenseNet121_CBAM_FPN(num_classes=5)

p_base = sum(p.numel() for p in m_base.parameters())
p_full = sum(p.numel() for p in m_full.parameters())
print(f"DenseNet121 Baseline Parameters: {p_base:,} (~{p_base/1e6:.2f}M)")
print(f"Full Model (DenseNet+CBAM+FPN):   {p_full:,} (~{p_full/1e6:.2f}M)")
print(f"Added attention & fusion params: {p_full - p_base:,} (~{(p_full-p_base)/1e6:.2f}M)")""")

# --- Cell 5: Training Function with Early Stopping ---
add_md("## Step 5: Training Loop with Backbone Freezing & Early Stopping")
add_code("""def train_model(model, model_name, train_loader, val_loader, class_weights,
                max_epochs=60, freeze_epochs=8, patience=12):
    print(f\"\\n{'='*70}\\nSTARTING TRAINING: {model_name}\\n{'='*70}\")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Separate parameters: backbone vs new layers
    if isinstance(model, DenseNet121_Baseline):
        backbone_params = [p for name, p in model.densenet.named_parameters() if 'classifier' not in name]
        head_params = list(model.densenet.classifier.parameters())
    else:
        backbone_params = (
            list(model.stage1.parameters()) +
            list(model.stage2.parameters()) +
            list(model.stage3.parameters())
        )
        head_params = (
            list(model.cbam1.parameters()) + list(model.cbam2.parameters()) + list(model.cbam3.parameters()) +
            list(model.lat1.parameters()) + list(model.lat2.parameters()) + list(model.lat3.parameters()) +
            list(model.smooth1.parameters()) + list(model.smooth2.parameters()) + list(model.smooth3.parameters()) +
            list(model.classifier.parameters())
        )

    # Initial freeze of backbone
    for p in backbone_params:
        p.requires_grad = False

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 5e-5, 'weight_decay': 1e-4},
        {'params': head_params,     'lr': 5e-4, 'weight_decay': 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_model_weights = None
    best_epoch = 0
    patience_counter = 0

    checkpoint_path = os.path.join(SAVE_DIR, f"{model_name}_best.pth")

    for epoch in range(1, max_epochs + 1):
        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs + 1:
            print(f"\\n>>> Epoch {epoch}: Unfreezing backbone layers for fine-tuning!")
            for p in backbone_params:
                p.requires_grad = True

        # Training Phase
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
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels).item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels).item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        current_lr = optimizer.param_groups[1]['lr']
        scheduler.step(epoch_val_acc)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch:02d}/{max_epochs} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:5.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:5.2f}% | LR: {current_lr:.1e}", end="")

        # Early Stopping Check on Validation Accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, checkpoint_path)
            patience_counter = 0
            print("  --> Best Model Saved!")
        else:
            patience_counter += 1
            print(f"  [Patience {patience_counter}/{patience}]")

        if patience_counter >= patience:
            print(f"\\nEarly stopping triggered at Epoch {epoch}! Best Val Acc: {best_val_acc*100:.2f}% at Epoch {best_epoch}.")
            break

    # Restore best checkpoint
    print(f"Restoring best weights from Epoch {best_epoch} (Val Acc: {best_val_acc*100:.2f}%)...")
    model.load_state_dict(best_model_weights)
    return model, history, best_epoch""")

# --- Cell 6: Run Training ---
add_md("## Step 6: Train Baseline and Full Model")
add_code("""# Train Baseline Model
baseline_model = DenseNet121_Baseline(num_classes=5)
baseline_model, baseline_history, baseline_best_ep = train_model(
    baseline_model, "DenseNet121_Baseline",
    train_loader, val_loader, class_weights_tensor,
    max_epochs=60, freeze_epochs=8, patience=12
)

# Train Full Model (DenseNet121 + CBAM + FPN)
seed_everything(SEED) # Reset seed for exact reproducibility
full_model = DenseNet121_CBAM_FPN(num_classes=5)
full_model, full_history, full_best_ep = train_model(
    full_model, "DenseNet121_CBAM_FPN",
    train_loader, val_loader, class_weights_tensor,
    max_epochs=60, freeze_epochs=8, patience=12
)""")

# --- Cell 7: Evaluation Function ---
add_md("## Step 7: Comprehensive Evaluation on 678 Test Images")
add_code("""def evaluate_model(model, test_loader, model_name):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 1. Verification Safeguard: Confusion Matrix Row-Sums
    cm = confusion_matrix(all_targets, all_preds)
    row_sums = cm.sum(axis=1)
    print(f"\\n{'='*70}\\nEVALUATION: {model_name}\\n{'='*70}")
    print(f"Confusion Matrix Row Sums: {row_sums.tolist()}")
    print(f"Total Test Predictions:   {len(all_preds)} (Row sum check: {row_sums.sum() == 678})")

    # 2. Accuracy & Macro Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, support = precision_recall_fscore_support(all_targets, all_preds, average=None)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')

    # 3. One-vs-Rest ROC & AUC
    y_true_bin = np.zeros((len(all_targets), len(CLASSES)))
    for i, t in enumerate(all_targets):
        y_true_bin[i, t] = 1

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(CLASSES)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    macro_auc = np.mean([roc_auc[i] for i in range(len(CLASSES))])

    # 4. Precision-Recall Curves
    pr_curve_data = dict()
    for i in range(len(CLASSES)):
        p, r, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
        ap = average_precision_score(y_true_bin[:, i], all_probs[:, i])
        pr_curve_data[i] = (r, p, ap)

    # Print Report Table
    df_report = pd.DataFrame({
        'Class': CLASSES,
        'Support': support,
        'Precision': [round(x, 4) for x in prec],
        'Recall':    [round(x, 4) for x in rec],
        'F1-Score':  [round(x, 4) for x in f1],
        'ROC-AUC':   [round(roc_auc[i], 4) for i in range(len(CLASSES))]
    })
    print(df_report.to_string(index=False))
    print(f"\\nOVERALL ACCURACY: {acc*100:.2f}% | MACRO F1: {macro_f1:.4f} | MACRO AUC: {macro_auc:.4f}")

    results = {
        'model_name': model_name,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'macro_auc': macro_auc,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'per_class_table': df_report,
        'cm': cm,
        'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
        'pr_curve_data': pr_curve_data,
        'all_targets': all_targets,
        'all_preds': all_preds,
        'all_probs': all_probs
    }
    return results

base_results = evaluate_model(baseline_model, test_loader, "DenseNet121_Baseline")
full_results = evaluate_model(full_model,     test_loader, "DenseNet121_CBAM_FPN")""")

# --- Cell 8: Plotting Curves ---
add_md("## Step 8: Generate All Teacher-Requested Plots & Curves")
add_code("""plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_paths = []

# 1. Loss & Accuracy Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_base = range(1, len(baseline_history['train_loss']) + 1)
epochs_full = range(1, len(full_history['train_loss']) + 1)

# Loss
ax1.plot(epochs_base, baseline_history['train_loss'], 'b--', label='Base Train Loss')
ax1.plot(epochs_base, baseline_history['val_loss'],   'b-',  label='Base Val Loss')
ax1.plot(epochs_full, full_history['train_loss'],     'r--', label='Full Train Loss')
ax1.plot(epochs_full, full_history['val_loss'],       'r-',  label='Full Val Loss')
ax1.axvline(8, color='gray', linestyle=':', label='Unfreeze Backbone (Ep 8)')
ax1.set_title("Training & Validation Loss Curve", fontsize=12, fontweight='bold')
ax1.set_xlabel("Epochs")
ax1.set_ylabel("CrossEntropy Loss")
ax1.legend(loc='upper right')

# Accuracy
ax2.plot(epochs_base, [x*100 for x in baseline_history['train_acc']], 'b--', label='Base Train Acc')
ax2.plot(epochs_base, [x*100 for x in baseline_history['val_acc']],   'b-',  label='Base Val Acc')
ax2.plot(epochs_full, [x*100 for x in full_history['train_acc']],     'r--', label='Full Train Acc')
ax2.plot(epochs_full, [x*100 for x in full_history['val_acc']],       'r-',  label='Full Val Acc')
ax2.axvline(8, color='gray', linestyle=':', label='Unfreeze Backbone (Ep 8)')
ax2.set_title("Training & Validation Accuracy Curve", fontsize=12, fontweight='bold')
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy (%)")
ax2.legend(loc='lower right')

plt.tight_layout()
p1 = os.path.join(SAVE_DIR, "loss_accuracy_curves.png")
plt.savefig(p1, dpi=300)
fig_paths.append(p1)
plt.show()

# 2. Confusion Matrices Side-by-Side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
short_classes = ['Polyps', 'Mucosa/Bowel', 'Stomach', 'Pylorus', 'Small Bowel']

sns.heatmap(base_results['cm'], annot=True, fmt='d', cmap='Blues',
            xticklabels=short_classes, yticklabels=short_classes, ax=ax1)
ax1.set_title(f"Baseline Confusion Matrix (Acc: {base_results['accuracy']*100:.2f}%)", fontweight='bold')
ax1.set_xlabel("Predicted Label")
ax1.set_ylabel("Ground Truth Label")

sns.heatmap(full_results['cm'], annot=True, fmt='d', cmap='Greens',
            xticklabels=short_classes, yticklabels=short_classes, ax=ax2)
ax2.set_title(f"Full Model (CBAM+FPN) CM (Acc: {full_results['accuracy']*100:.2f}%)", fontweight='bold')
ax2.set_xlabel("Predicted Label")
ax2.set_ylabel("Ground Truth Label")

plt.tight_layout()
p2 = os.path.join(SAVE_DIR, "confusion_matrices.png")
plt.savefig(p2, dpi=300)
fig_paths.append(p2)
plt.show()

# 3. One-vs-Rest ROC Curves Side-by-Side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for i, cls in enumerate(short_classes):
    ax1.plot(base_results['fpr'][i], base_results['tpr'][i],
             label=f"{cls} (AUC = {base_results['roc_auc'][i]:.3f})")
ax1.plot(base_results['fpr']['micro'], base_results['tpr']['micro'],
         'k:', label=f"Micro-avg (AUC = {base_results['roc_auc']['micro']:.3f})", linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_title(f"Baseline ROC Curves (Macro AUC = {base_results['macro_auc']:.4f})", fontweight='bold')
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend(loc='lower right', fontsize=9)

for i, cls in enumerate(short_classes):
    ax2.plot(full_results['fpr'][i], full_results['tpr'][i],
             label=f"{cls} (AUC = {full_results['roc_auc'][i]:.3f})")
ax2.plot(full_results['fpr']['micro'], full_results['tpr']['micro'],
         'k:', label=f"Micro-avg (AUC = {full_results['roc_auc']['micro']:.3f})", linewidth=2)
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax2.set_title(f"Full Model ROC Curves (Macro AUC = {full_results['macro_auc']:.4f})", fontweight='bold')
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend(loc='lower right', fontsize=9)

plt.tight_layout()
p3 = os.path.join(SAVE_DIR, "roc_curves.png")
plt.savefig(p3, dpi=300)
fig_paths.append(p3)
plt.show()

# 4. Precision-Recall Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for i, cls in enumerate(short_classes):
    r, p, ap = base_results['pr_curve_data'][i]
    ax1.plot(r, p, label=f"{cls} (AP = {ap:.3f})")
ax1.set_title(f"Baseline Precision-Recall Curves", fontweight='bold')
ax1.set_xlabel("Recall")
ax1.set_ylabel("Precision")
ax1.legend(loc='lower left', fontsize=9)

for i, cls in enumerate(short_classes):
    r, p, ap = full_results['pr_curve_data'][i]
    ax2.plot(r, p, label=f"{cls} (AP = {ap:.3f})")
ax2.set_title(f"Full Model (CBAM+FPN) PR Curves", fontweight='bold')
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend(loc='lower left', fontsize=9)

plt.tight_layout()
p4 = os.path.join(SAVE_DIR, "precision_recall_curves.png")
plt.savefig(p4, dpi=300)
fig_paths.append(p4)
plt.show()
print(f"All 4 figure panels successfully generated and saved to: {SAVE_DIR}")""")

# --- Cell 9: Comparison Table & Download Zip ---
add_md("## Step 9: Final Comparison Table and Downloadable Results Zip Archive")
add_code("""# Final Comparison Table
comparison_df = pd.DataFrame({
    'Metric / Property': [
        'Model Parameters',
        'Overall Test Accuracy',
        'Macro F1-Score',
        'Macro ROC-AUC',
        'Pylorus Class F1 (Smallest Class)',
        'Epochs Run (Early Stop Trigger)',
        'Best Checkpoint Restored'
    ],
    'DenseNet121 Baseline': [
        f\"{p_base:,} (~{p_base/1e6:.2f}M)\",
        f\"{base_results['accuracy']*100:.2f}%\",
        f\"{base_results['macro_f1']:.4f}\",
        f\"{base_results['macro_auc']:.4f}\",
        f\"{base_results['per_class_table'].loc[base_results['per_class_table']['Class'] == 'Pylorus', 'F1-Score'].values[0]:.4f}\",
        f\"{len(baseline_history['train_loss'])} epochs\",
        f\"Epoch {baseline_best_ep}\"
    ],
    'DenseNet121 + CBAM + FPN': [
        f\"{p_full:,} (~{p_full/1e6:.2f}M)\",
        f\"{full_results['accuracy']*100:.2f}%\",
        f\"{full_results['macro_f1']:.4f}\",
        f\"{full_results['macro_auc']:.4f}\",
        f\"{full_results['per_class_table'].loc[full_results['per_class_table']['Class'] == 'Pylorus', 'F1-Score'].values[0]:.4f}\",
        f\"{len(full_history['train_loss'])} epochs\",
        f\"Epoch {full_best_ep}\"
    ],
    'Delta (Gain)': [
        f\"+{p_full - p_base:,} (+{(p_full-p_base)/p_base*100:.1f}%)\",
        f\"+{(full_results['accuracy'] - base_results['accuracy'])*100:.2f}%\xb2\",
        f\"+{(full_results['macro_f1'] - base_results['macro_f1']):.4f}\",
        f\"+{(full_results['macro_auc'] - base_results['macro_auc']):.4f}\",
        f\"+{(full_results['per_class_table'].loc[full_results['per_class_table']['Class'] == 'Pylorus', 'F1-Score'].values[0] - base_results['per_class_table'].loc[base_results['per_class_table']['Class'] == 'Pylorus', 'F1-Score'].values[0]):.4f}\",
        \"-\",
        \"-\"
    ]
})

print(\"\\n\" + \"=\"*80)
print(\"FINAL MODEL COMPARISON SUMMARY (678 Test Images, Seed 42)\")
print(\"=\"*80)
print(comparison_df.to_string(index=False))

# Save CSV reports
comparison_df.to_csv(os.path.join(SAVE_DIR, "final_model_comparison.csv"), index=False)
base_results['per_class_table'].to_csv(os.path.join(SAVE_DIR, "baseline_per_class.csv"), index=False)
full_results['per_class_table'].to_csv(os.path.join(SAVE_DIR, "full_model_per_class.csv"), index=False)

# Create Zip Archive for one-click download
ZIP_OUTPUT = '/content/Gastrovision_DenseNet121_CBAM_Results.zip'
shutil.make_archive('/content/Gastrovision_DenseNet121_CBAM_Results', 'zip', SAVE_DIR)
shutil.copy2(ZIP_OUTPUT, os.path.join(SAVE_DIR, "Gastrovision_DenseNet121_CBAM_Results.zip"))

print(f\"\\nZip archive created at: {ZIP_OUTPUT}\")
print(\"Initiating automatic download to your local machine...\")
files.download(ZIP_OUTPUT)
print(\"\\nAll tasks completed successfully!\")""")

# Save notebook to workspace
notebook_path = r"c:\Users\AMRUTA M PATIL\Downloads\Gastrovision_5classes\GastroVision_DenseNet121_CBAM_FPN_Complete.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Jupyter Notebook successfully created at: {notebook_path}")

