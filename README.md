# GastroVision GI Classification Using DenseNet121-CBAM-FPN

A deep learning pipeline for Gastrointestinal (GI) disease classification across 5 endoscopic findings using **DenseNet121** augmented with **Multi-Level CBAM (Convolutional Block Attention Module)** and **Feature Pyramid Networks (FPN)**.

---

## 🔬 Dataset Overview

The dataset consists of endoscopic imagery across 5 critical gastrointestinal anatomical and pathological categories:

1. **Colon polyps** (pre-cancerous lesions)
2. **Normal mucosa and vascular pattern in the large bowel**
3. **Normal stomach**
4. **Pylorus**
5. **Small bowel (terminal ileum)**

### Rigorous Preprocessing & Anti-Overfitting Protocol
To resolve severe model overfitting and feature ambiguity caused by camera motion, mucosal secretions, and video frame redundancy, the raw dataset underwent an automated multi-stage cleaning pipeline:

| Category | Initial Count | Blurry Removed (Laplacian < 100) | Exact Duplicates (MD5) | Near-Duplicates (pHash ≤ 8) | **Final Clean Count** |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Colon polyps** | 820 | 37 | 1 | 65 | **717** |
| **Normal mucosa (large bowel)** | 1,467 | 23 | 0 | 254 | **1,190** |
| **Normal stomach** | 969 | 133 | 0 | 97 | **739** |
| **Pylorus** | 393 | 41 | 0 | 52 | **300** |
| **Small bowel (terminal ileum)** | 846 | 70 | 0 | 95 | **681** |
| **TOTAL** | **4,495** | **304** | **1** | **563** | **3,627** |

> **Key Takeaway:** 868 problematic images (19.3%) were quarantined. Removing near-duplicates (consecutive endoscopic video frames) completely prevents **data leakage** between training and evaluation splits, ensuring authentic generalization performance.

---

## 📁 Repository Structure

```
GastroVision-GI-Classification-Using-DenseNet121-CBAM-FPN/
├── dataset/
│   ├── train/                   # 2,537 images (70% stratified split)
│   │   ├── Colon polyps
│   │   ├── Normal mucosa and vascular pattern in the large bowel
│   │   ├── Normal stomach
│   │   ├── Pylorus
│   │   └── Small bowel_terminal ileum
│   ├── val/                     # 542 images (15% stratified split)
│   │   └── [5 classes]
│   └── test/                    # 548 images (15% stratified split)
│       └── [5 classes]
│
├── improved_densenet_cbam.py     # DenseNet121 with Multi-Level CBAM architecture
├── train_example.py             # PyTorch ImageFolder training boilerplate
├── clean_dataset.py             # Preprocessing & deduplication pipeline
├── cleaning_log.txt             # Verification & audit log
└── README.md
```

---

## 🧠 Model Architecture: DenseNet121 + CBAM

The architecture integrates attention mechanisms at multiple dense block transitions:
- **Channel Attention:** Squeezes spatial dimensions via adaptive average and max pooling, followed by shared MLPs to weight salient mucosal features.
- **Spatial Attention:** Captures inter-spatial relationships of lesion regions using $7 \times 7$ convolutions over channel-aggregated feature maps.
- **DenseNet121 Backbone:** Feature reuse through dense connectivity allows deep feature extraction without vanishing gradient issues.

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Amruta09-eng/GastroVision-GI-Classification-Using-DenseNet121-CBAM-FPN.git
cd GastroVision-GI-Classification-Using-DenseNet121-CBAM-FPN
```

### 2. Install Dependencies
```bash
pip install torch torchvision opencv-python imagehash scikit-learn tqdm
```

### 3. Load Dataset in PyTorch
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('dataset/train', transform=train_transforms)
val_dataset   = datasets.ImageFolder('dataset/val', transform=train_transforms)
test_dataset  = datasets.ImageFolder('dataset/test', transform=train_transforms)

train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
```

---

## ⚖️ License
MIT License. Available for academic research and educational use.
