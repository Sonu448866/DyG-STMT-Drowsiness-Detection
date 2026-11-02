import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.dyg_stmt_head import DyGSTMTSequenceModel
from utils.losses import MultiTaskFocalLoss
from utils.metrics import compute_drowsiness_metrics, print_metrics_summary

def load_feature_dataset(npz_path):
    print(f"[DataLoader] Loading precomputed feature tensor from {npz_path}...", flush=True)
    data = np.load(npz_path)
    
    eye_feats = torch.from_numpy(data['eye_feats']).float()
    mouth_feats = torch.from_numpy(data['mouth_feats']).float()
    geo_vecs = torch.from_numpy(data['geo_vecs']).float()
    
    cond_indices = torch.from_numpy(data['cond_idx']).long()
    cond_one_hot = torch.zeros((len(cond_indices), 5), dtype=torch.float32)
    cond_one_hot.scatter_(1, cond_indices.unsqueeze(1), 1.0)
    
    label_drowsy = torch.from_numpy(data['label_drowsy']).float()
    label_eye = torch.from_numpy(data['label_eye']).float()
    label_mouth = torch.from_numpy(data['label_mouth']).float()
    label_head = torch.from_numpy(data['label_head']).float()
    
    dataset = TensorDataset(eye_feats, mouth_feats, geo_vecs, cond_one_hot, label_drowsy, label_eye, label_mouth, label_head)
    print(f"Loaded {len(dataset)} sequences.", flush=True)
    return dataset


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        eye_b, mouth_b, geo_b, cond_b, y_drowsy, y_eye, y_mouth, y_head = [x.to(device) for x in batch]
        
        optimizer.zero_grad()
        outputs = model(eye_b, mouth_b, geo_b, cond_b)
        targets = {
            'drowsy': y_drowsy,
            'eye': y_eye,
            'mouth': y_mouth,
            'head': y_head
        }
        loss, loss_dict = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        running_loss += loss.item() * len(y_drowsy)
        probs = torch.sigmoid(outputs['drowsy_logits']).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(y_drowsy.cpu().numpy())
        
    epoch_loss = running_loss / max(1, len(dataloader.dataset))
    metrics = compute_drowsiness_metrics(all_targets, all_preds)
    metrics['loss'] = epoch_loss
    return metrics


def evaluate_features(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            eye_b, mouth_b, geo_b, cond_b, y_drowsy, y_eye, y_mouth, y_head = [x.to(device) for x in batch]
            outputs = model(eye_b, mouth_b, geo_b, cond_b)
            targets = {
                'drowsy': y_drowsy,
                'eye': y_eye,
                'mouth': y_mouth,
                'head': y_head
            }
            loss, _ = criterion(outputs, targets)
            running_loss += loss.item() * len(y_drowsy)
            probs = torch.sigmoid(outputs['drowsy_logits']).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y_drowsy.cpu().numpy())
            
    eval_loss = running_loss / max(1, len(dataloader.dataset))
    metrics = compute_drowsiness_metrics(all_targets, all_preds)
    metrics['loss'] = eval_loss
    return metrics, all_preds, all_targets


def train_dyg_stmt(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n============================================================", flush=True)
    print(f"  Training DyG-STMT on NTHU-DDD Benchmark ({device})", flush=True)
    print(f"  Hyperparameters: LR={args.lr}, Batch={args.batch_size}, Epochs={args.epochs}, WeightDecay={args.weight_decay}", flush=True)
    print(f"============================================================\n", flush=True)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    train_dataset = load_feature_dataset(args.train_features)
    val_dataset = load_feature_dataset(args.eval_features) if os.path.exists(args.eval_features) else None
    
    if val_dataset is None:
        train_len = int(0.85 * len(train_dataset))
        val_len = len(train_dataset) - train_len
    # Balanced WeightedRandomSampler to eliminate class imbalance bias
    labels_drowsy = train_dataset.tensors[4].numpy()
    num_pos = int(np.sum(labels_drowsy == 1))
    num_neg = int(np.sum(labels_drowsy == 0))
    print(f"[Dataset Stats] Train samples: {len(labels_drowsy)} (Positives: {num_pos}, Negatives: {num_neg})", flush=True)
    
    weight_pos = 1.0 / max(1, num_pos)
    weight_neg = 1.0 / max(1, num_neg)
    sample_weights = np.where(labels_drowsy == 1, weight_pos, weight_neg)
    sampler = torch.utils.data.WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = DyGSTMTSequenceModel(num_conditions=5, feature_dim=128, gru_hidden=64, num_layers=1, dropout=args.dropout).to(device)
    criterion = MultiTaskFocalLoss(gamma=2.0, alpha=0.50, lambda_eye=0.20, lambda_mouth=0.15, lambda_head=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_met = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_met, _, _ = evaluate_features(model, val_loader, criterion, device)
        scheduler.step()
        
        elapsed = time.time() - t0
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.2f}s) | "
              f"Train Loss: {train_met['loss']:.4f} | Acc: {train_met['accuracy']*100:.2f}% | F1: {train_met['f1_score']*100:.2f}% || "
              f"Val Loss: {val_met['loss']:.4f} | Acc: {val_met['accuracy']*100:.2f}% | F1: {val_met['f1_score']*100:.2f}% | ROC-AUC: {val_met['roc_auc']*100:.2f}%", flush=True)
              
        if val_met['f1_score'] > best_val_f1 or (val_met['f1_score'] == best_val_f1 and val_met['accuracy'] > best_val_acc):
            best_val_f1 = val_met['f1_score']
            best_val_acc = val_met['accuracy']
            best_epoch = epoch
            
            save_path = os.path.join(args.checkpoint_dir, 'best_dyg_stmt.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_met,
                'args': vars(args)
            }, save_path)
            print(f"  [CHECKPOINT SAVED] Val F1 = {best_val_f1*100:.2f}%, Val Acc = {best_val_acc*100:.2f}%\n", flush=True)
            
    print(f"\n==================== Training Finished ====================", flush=True)
    print(f"  Best Epoch : {best_epoch}", flush=True)
    print(f"  Best Val Accuracy: {best_val_acc*100:.2f}%", flush=True)
    print(f"  Best Val F1-Score: {best_val_f1*100:.2f}%", flush=True)
    print(f"===========================================================\n", flush=True)
    return best_val_acc, best_val_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_features', type=str, default='data_cache/features_train.npz')
    parser.add_argument('--eval_features', type=str, default='data_cache/features_eval.npz')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.25)
    args = parser.parse_args()
    
    train_dyg_stmt(args)
