import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.dyg_stmt import FastSpatialEncoder

class SpatialClassifier(nn.Module):
    def __init__(self, encoder, out_dim=128):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat).squeeze(-1)

def train_encoder_on_frames(imgs, labels, name="eye", epochs=15, batch_size=128, lr=1e-3):
    print(f"\n==================== Training {name.upper()} Spatial Encoder ====================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Flatten sequence frames: sample every 5th frame for fast diverse training
    N, T, C, H, W = imgs.shape
    sampled_imgs = imgs[:, ::5, :, :, :].reshape(-1, C, H, W)
    sampled_labels = labels[:, ::5].reshape(-1)
    
    # Filter valid frames
    valid_mask = ~np.isnan(sampled_labels)
    sampled_imgs = sampled_imgs[valid_mask]
    sampled_labels = sampled_labels[valid_mask]
    
    print(f"Total training frames for {name}: {len(sampled_labels)} (Positives: {int(np.sum(sampled_labels > 0.5))})")
    
    # Train / Val Split
    n_total = len(sampled_labels)
    n_train = int(0.85 * n_total)
    perm = np.random.permutation(n_total)
    
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    
    train_ds = TensorDataset(
        torch.from_numpy(sampled_imgs[train_idx]).float() / 255.0,
        torch.from_numpy(sampled_labels[train_idx]).float()
    )
    val_ds = TensorDataset(
        torch.from_numpy(sampled_imgs[val_idx]).float() / 255.0,
        torch.from_numpy(sampled_labels[val_idx]).float()
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    encoder = FastSpatialEncoder(out_dim=128)
    model = SpatialClassifier(encoder, out_dim=128).to(device)
    
    pos_weight = torch.tensor([(len(sampled_labels) - np.sum(sampled_labels)) / max(1, np.sum(sampled_labels))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(by)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == by).sum().item()
            total += len(by)
            
        scheduler.step()
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == by).sum().item()
                val_total += len(by)
                
        val_acc = val_correct / max(1, val_total)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = encoder.state_dict()
            
    print(f"Finished {name} training! Best Val Acc: {best_val_acc*100:.2f}%")
    return best_weights

def main():
    os.makedirs('checkpoints', exist_ok=True)
    print("Loading data_cache/train_consolidated.npz...")
    data = np.load('data_cache/train_consolidated.npz', mmap_mode='r')
    
    eye_weights = train_encoder_on_frames(data['eye_imgs'], data['label_eye'], name="eye", epochs=15)
    torch.save(eye_weights, 'checkpoints/pretrained_eye_encoder.pt')
    print("Saved checkpoints/pretrained_eye_encoder.pt")
    
    mouth_weights = train_encoder_on_frames(data['mouth_imgs'], data['label_mouth'], name="mouth", epochs=15)
    torch.save(mouth_weights, 'checkpoints/pretrained_mouth_encoder.pt')
    print("Saved checkpoints/pretrained_mouth_encoder.pt")

if __name__ == '__main__':
    main()
