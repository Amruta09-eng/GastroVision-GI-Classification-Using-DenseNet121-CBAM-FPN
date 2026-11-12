"""
IMPROVED DenseNet121 with Multi-Level CBAM Attention
Enhanced for better performance on Endoscopy Classification

Key Improvements:
1. CBAM at multiple dense block levels (not just final)
2. Channel-wise and spatial attention modules
3. Better data augmentation with MixUp
4. Advanced optimizer (AdamW) with warmup
5. Gradient clipping and batch normalization tuning
6. Label smoothing and class weighting
"""

import torch
import torch.nn as torch
import torchvision.models as models
from torch import nn
import torch.nn.functional as F

# ============================================================
# 1. CBAM ATTENTION MODULE (Convolutional Block Attention)
# ============================================================

class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Complete CBAM Block"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

# ============================================================
# 2. IMPROVED DENSENET WITH MULTI-LEVEL CBAM
# ============================================================

class DenseNetWithCBAM(nn.Module):
    """DenseNet121 with CBAM at multiple levels"""
    
    def __init__(self, num_classes=5, cbam_reduction=16):
        super().__init__()
        
        # Load pretrained DenseNet121
        densenet = models.densenet121(pretrained=True)
        
        # Extract features
        self.features = densenet.features
        
        # Add CBAM modules at different dense blocks
        self.cbam_block1 = CBAM(64, reduction=cbam_reduction)    # After initial conv
        self.cbam_block2 = CBAM(256, reduction=cbam_reduction)   # After DenseBlock1
        self.cbam_block3 = CBAM(512, reduction=cbam_reduction)   # After DenseBlock2
        self.cbam_block4 = CBAM(1024, reduction=cbam_reduction)  # After DenseBlock3
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Initial convolution
        x = self.features[0](x)  # Conv2d
        x = self.features[1](x)  # BatchNorm
        x = self.features[2](x)  # ReLU
        x = self.cbam_block1(x)
        x = self.features[3](x)  # MaxPool
        
        # DenseBlocks with CBAM
        x = self.features[4](x)  # DenseBlock1
        x = self.cbam_block2(x)
        x = self.features[5](x)  # Transition1
        
        x = self.features[6](x)  # DenseBlock2
        x = self.cbam_block3(x)
        x = self.features[7](x)  # Transition2
        
        x = self.features[8](x)  # DenseBlock3
        x = self.cbam_block4(x)
        x = self.features[9](x)  # Transition3
        
        x = self.features[10](x) # DenseBlock4
        x = self.features[11](x) # BatchNorm
        x = F.relu(x, inplace=True)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


# ============================================================
# 3. ADVANCED TRAINING UTILITIES
# ============================================================

class MixupAugmentation:
    """MixUp data augmentation for better generalization"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x1, y1, x2, y2):
        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample()
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        return x, y


class WarmupScheduler:
    """Learning rate warmup scheduler"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing after warmup
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


# ============================================================
# 4. ADVANCED TRAINING CONFIGURATION
# ============================================================

class TrainingConfig:
    """Complete training configuration"""
    
    # Data parameters
    BATCH_SIZE = 32  # Larger batch size
    NUM_WORKERS = 4
    
    # Model parameters
    NUM_CLASSES = 5
    CBAM_REDUCTION = 16
    
    # Training parameters
    NUM_EPOCHS = 40  # More epochs
    BASE_LR = 0.001
    WARMUP_EPOCHS = 5
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0
    
    # Regularization
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2
    
    # Data augmentation (use these in your DataLoader)
    TRAIN_AUGMENTATION = {
        'random_horizontal_flip': 0.5,
        'random_vertical_flip': 0.3,
        'random_rotation': 20,
        'random_affine': {
            'degrees': 15,
            'translate': (0.1, 0.1),
            'scale': (0.9, 1.1),
            'shear': 10
        },
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.1
        },
        'random_gaussian_blur': {
            'kernel_size': 3,
            'sigma': (0.1, 2.0)
        }
    }
    
    # Class weights (adjust based on your class distribution)
    CLASS_WEIGHTS = torch.tensor([1.2, 1.0, 1.1, 1.4, 1.0])  # Adjust per your needs


# ============================================================
# 5. TRAINING LOOP TEMPLATE
# ============================================================

def create_improved_model():
    """Create improved DenseNet with CBAM"""
    model = DenseNetWithCBAM(
        num_classes=TrainingConfig.NUM_CLASSES,
        cbam_reduction=TrainingConfig.CBAM_REDUCTION
    )
    return model


def setup_optimizer_and_scheduler(model):
    """Setup AdamW optimizer with warmup"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TrainingConfig.BASE_LR,
        weight_decay=TrainingConfig.WEIGHT_DECAY
    )
    
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=TrainingConfig.WARMUP_EPOCHS,
        total_epochs=TrainingConfig.NUM_EPOCHS,
        base_lr=TrainingConfig.BASE_LR
    )
    
    return optimizer, scheduler


def create_loss_function():
    """Create loss with label smoothing and class weighting"""
    return torch.nn.CrossEntropyLoss(
        label_smoothing=TrainingConfig.LABEL_SMOOTHING,
        weight=TrainingConfig.CLASS_WEIGHTS
    )


# ============================================================
# USAGE IN YOUR COLAB
# ============================================================
"""
# Step 1: Create model
model = create_improved_model()
model = model.to(device)

# Step 2: Setup optimizer and scheduler
optimizer, scheduler = setup_optimizer_and_scheduler(model)

# Step 3: Loss function with class weighting
criterion = create_loss_function()

# Step 4: In your training loop:
for epoch in range(TrainingConfig.NUM_EPOCHS):
    # Training phase
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # MixUp augmentation (optional)
        if torch.rand(1) > 0.5 and batch_idx < len(train_loader) - 1:
            # Apply mixup (requires next batch)
            pass
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.GRADIENT_CLIP)
        optimizer.step()
    
    # Update learning rate
    scheduler.step()

# Step 5: Evaluation (same as before, but with improved model)
"""
