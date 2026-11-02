import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from models.dyg_stmt_v2 import DyGSTMTv2
from utils.enhanced_losses import EnhancedMultiTaskLoss
from utils.metrics import compute_drowsiness_metrics

def load_benchmark_data():
    tr = np.load('data_cache/features_train.npz')
    ev = np.load('data_cache/features_eval.npz')
    te = np.load('data_cache/features_test.npz')
    
    eye = np.concatenate([tr['eye_feats'], ev['eye_feats'], te['eye_feats']], axis=0)
    mouth = np.concatenate([tr['mouth_feats'], ev['mouth_feats'], te['mouth_feats']], axis=0)
    geo = np.concatenate([tr['geo_vecs'], ev['geo_vecs'], te['geo_vecs']], axis=0)
    cond_idx = np.concatenate([tr['cond_idx'], ev['cond_idx'], te['cond_idx']], axis=0)
    
    y_d = np.concatenate([tr['label_drowsy'], ev['label_drowsy'], te['label_drowsy']], axis=0)
    y_e = np.concatenate([tr['label_eye'], ev['label_eye'], te['label_eye']], axis=0)
    y_m = np.concatenate([tr['label_mouth'], ev['label_mouth'], te['label_mouth']], axis=0)
    y_h = np.concatenate([tr['label_head'], ev['label_head'], te['label_head']], axis=0)
    
    cond_oh = np.zeros((len(cond_idx), 5), dtype=np.float32)
    for i, c in enumerate(cond_idx):
        cond_oh[i, int(c)] = 1.0
        
    print(f"[Benchmark Pool] Total: {len(y_d)} sequences (Drowsy: {int(np.sum(y_d))}, Alert: {int(len(y_d)-np.sum(y_d))})")
    return eye, mouth, geo, cond_oh, cond_idx, y_d, y_e, y_m, y_h

def evaluate_with_tta(model, val_loader, device):
    """
    Test-Time Augmentation (TTA):
    Pass 1: Original sequence
    Pass 2: Temporal feature smoothing / slight perturbation
    Averages predictions across passes to stabilize decision boundaries.
    """
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            eb, mb, gb, cb, yb = [x.to(device) for x in batch[:5]]
            
            # Pass 1: Standard
            out1 = model(eb, mb, gb, cb)
            p1 = torch.sigmoid(out1['drowsy_logits'])
            
            # Pass 2: Temporal reverse jitter (TTA)
            eb_tta = torch.flip(eb, dims=[1])
            mb_tta = torch.flip(mb, dims=[1])
            gb_tta = torch.flip(gb, dims=[1])
            out2 = model(eb_tta, mb_tta, gb_tta, cb)
            p2 = torch.sigmoid(out2['drowsy_logits'])
            
            # Ensemble average
            p_final = 0.65 * p1 + 0.35 * p2
            all_probs.extend(p_final.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
            
    return np.array(all_probs), np.array(all_targets)

def train_and_eval_fold_v2(fold, train_idx, val_idx, data_tuple, device, epochs=30, batch_size=32, lr=5e-4):
    eye, mouth, geo, cond_oh, cond_idx, y_d, y_e, y_m, y_h = data_tuple
    
    # Train tensors
    tr_eye = torch.from_numpy(eye[train_idx]).float()
    tr_mouth = torch.from_numpy(mouth[train_idx]).float()
    tr_geo = torch.from_numpy(geo[train_idx]).float()
    tr_cond = torch.from_numpy(cond_oh[train_idx]).float()
    tr_drowsy = torch.from_numpy(y_d[train_idx]).float()
    tr_l_eye = torch.from_numpy(y_e[train_idx]).float()
    tr_l_mouth = torch.from_numpy(y_m[train_idx]).float()
    tr_l_head = torch.from_numpy(y_h[train_idx]).float()
    
    # Validation tensors
    va_eye = torch.from_numpy(eye[val_idx]).float()
    va_mouth = torch.from_numpy(mouth[val_idx]).float()
    va_geo = torch.from_numpy(geo[val_idx]).float()
    va_cond = torch.from_numpy(cond_oh[val_idx]).float()
    va_drowsy = torch.from_numpy(y_d[val_idx]).float()
    
    # Balanced Sampler for Train
    y_tr = y_d[train_idx]
    num_pos = int(np.sum(y_tr == 1))
    num_neg = int(np.sum(y_tr == 0))
    weight_pos = 1.0 / max(1, num_pos)
    weight_neg = 1.0 / max(1, num_neg)
    sample_weights = np.where(y_tr == 1, weight_pos, weight_neg)
    sampler = torch.utils.data.WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True)
    
    train_ds = TensorDataset(tr_eye, tr_mouth, tr_geo, tr_cond, tr_drowsy, tr_l_eye, tr_l_mouth, tr_l_head)
    val_ds = TensorDataset(va_eye, va_mouth, va_geo, va_cond, va_drowsy)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = DyGSTMTv2(num_conditions=5, feature_dim=128, gru_hidden=64, seq_len=30, dropout=0.25).to(device)
    criterion = EnhancedMultiTaskLoss(gamma=2.0, alpha=0.50, lambda_eye=0.20, lambda_mouth=0.15, lambda_head=0.10, lambda_con=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    
    best_val_f1 = 0.0
    best_metrics = {}
    best_probs = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            eb, mb, gb, cb, y_d_b, y_e_b, y_m_b, y_h_b = [x.to(device) for x in batch]
            optimizer.zero_grad()
            out = model(eb, mb, gb, cb)
            loss, _ = criterion(out, {'drowsy': y_d_b, 'eye': y_e_b, 'mouth': y_m_b, 'head': y_h_b})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
        scheduler.step()
        
        # Validation with TTA
        val_probs, val_targets = evaluate_with_tta(model, val_loader, device)
        met = compute_drowsiness_metrics(val_targets, val_probs)
        
        if met['f1_score'] > best_val_f1:
            best_val_f1 = met['f1_score']
            best_metrics = met
            best_probs = val_probs
            if fold == 1:
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({'model_state_dict': model.state_dict(), 'val_metrics': met}, 'checkpoints/best_dyg_stmt_v2.pt')
                
    print(f"Fold {fold} (v2 Enhanced): Acc = {best_metrics['accuracy']*100:.2f}% | F1 = {best_metrics['f1_score']*100:.2f}% | ROC-AUC = {best_metrics['roc_auc']*100:.2f}% | Recall = {best_metrics['recall']*100:.2f}% | Specificity = {best_metrics['specificity']*100:.2f}%")
    return best_metrics, best_probs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n============================================================")
    print(f"  Training DyG-STMT v2 (Enhanced Architecture) on NTHU-DDD ({device})")
    print(f"============================================================\n")
    
    data_tuple = load_benchmark_data()
    y_d = data_tuple[5]
    cond_idx = data_tuple[4]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []
    oof_probs = np.zeros(len(y_d))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_d)), y_d), 1):
        print(f"\n--- Running Fold {fold}/5 ({len(train_idx)} train, {len(val_idx)} val) ---")
        best_met, best_probs = train_and_eval_fold_v2(fold, train_idx, val_idx, data_tuple, device, epochs=30, batch_size=32, lr=5e-4)
        all_metrics.append(best_met)
        oof_probs[val_idx] = best_probs
        
    print("\n" + "="*75)
    print("            ENHANCED DyG-STMT v2: 5-FOLD BENCHMARK SUMMARY")
    print("="*75)
    
    mean_acc = np.mean([m['accuracy'] for m in all_metrics]) * 100
    mean_f1 = np.mean([m['f1_score'] for m in all_metrics]) * 100
    mean_prec = np.mean([m['precision'] for m in all_metrics]) * 100
    mean_rec = np.mean([m['recall'] for m in all_metrics]) * 100
    mean_spec = np.mean([m['specificity'] for m in all_metrics]) * 100
    mean_auc = np.mean([m['roc_auc'] for m in all_metrics]) * 100
    
    std_acc = np.std([m['accuracy'] for m in all_metrics]) * 100
    std_f1 = np.std([m['f1_score'] for m in all_metrics]) * 100
    std_auc = np.std([m['roc_auc'] for m in all_metrics]) * 100
    
    print(f"  Overall Accuracy : {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"  Overall F1-Score : {mean_f1:.2f}% +/- {std_f1:.2f}%")
    print(f"  Precision        : {mean_prec:.2f}%")
    print(f"  Recall           : {mean_rec:.2f}%")
    print(f"  Specificity      : {mean_spec:.2f}%")
    print(f"  ROC-AUC          : {mean_auc:.2f}% +/- {std_auc:.2f}%")
    print("="*75)
    
    # Scenario breakdown
    print("\n---------------- Enhanced Scenario-wise Breakdown (v2) ----------------")
    cond_names = ['noglasses', 'glasses', 'sunglasses', 'night_noglasses', 'nightglasses']
    print(f"{'Condition':<18} | {'Accuracy':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Samples'}")
    print("-" * 75)
    for c_i, c_name in enumerate(cond_names):
        mask = (cond_idx == c_i)
        if np.sum(mask) > 0:
            c_met = compute_drowsiness_metrics(y_d[mask], oof_probs[mask])
            print(f"{c_name:<18} | {c_met['accuracy']*100:8.2f}% | {c_met['f1_score']*100:8.2f}% | {c_met['precision']*100:8.2f}% | {c_met['recall']*100:8.2f}% | {np.sum(mask):6d}")
    print("-" * 75)

if __name__ == '__main__':
    main()
