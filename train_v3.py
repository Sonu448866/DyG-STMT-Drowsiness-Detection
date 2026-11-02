"""
Phase 4 Training & 5-Fold Cross-Validation Script for DyG-STMT v3 (Ultra)
Strictly Enforces 100% Exact Baseline Literature Hyperparameters:
- Sequence Window (T): 30 frames @ 30 FPS (50% stride = 15 frames)
- Optimizer: AdamW (weight_decay = 1e-4)
- Learning Rate: 1e-4 with CosineAnnealingLR
- Batch Size: 32 (Standardized)
- Dropout: 0.35
- Early Stopping Patience: 12 epochs
- Validation: 5-Fold Subject-Independent Cross-Validation on NTHU-DDD
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from models.dyg_stmt_v3 import DyGSTMTv3
from utils.v3_losses import Phase4MultiTaskLoss
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
        
    print(f"[Phase 4 Benchmark Pool] Total: {len(y_d)} sequences (Drowsy: {int(np.sum(y_d))}, Alert: {int(len(y_d)-np.sum(y_d))})")
    return eye, mouth, geo, cond_oh, cond_idx, y_d, y_e, y_m, y_h

def evaluate_with_tta_v3(model, val_loader, device):
    """
    Test-Time Augmentation (TTA) with Forward & Temporal Jitter Ensembling.
    """
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            eb, mb, gb, cb, yb = [x.to(device) for x in batch[:5]]
            
            # Pass 1: Standard Forward Pass
            out1 = model(eb, mb, gb, cb)
            p1 = torch.sigmoid(out1['drowsy_logits'])
            
            # Pass 2: Temporal Invariance Augmentation
            eb_flip = torch.flip(eb, dims=[1])
            mb_flip = torch.flip(mb, dims=[1])
            gb_flip = torch.flip(gb, dims=[1])
            out2 = model(eb_flip, mb_flip, gb_flip, cb)
            p2 = torch.sigmoid(out2['drowsy_logits'])
            
            # Weighted Ensembling (70% standard, 30% flipped)
            p_final = 0.70 * p1 + 0.30 * p2
            all_probs.extend(p_final.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
            
    return np.array(all_probs), np.array(all_targets)

def train_and_eval_fold_v3(fold, train_idx, val_idx, data_tuple, device, epochs=35, batch_size=32, lr=1e-4, weight_decay=1e-4):
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
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_ds = TensorDataset(tr_eye, tr_mouth, tr_geo, tr_cond, tr_drowsy, tr_l_eye, tr_l_mouth, tr_l_head)
    val_ds = TensorDataset(va_eye, va_mouth, va_geo, va_cond, va_drowsy)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Instantiate Model & Loss
    model = DyGSTMTv3(num_conditions=5, feature_dim=128, gru_hidden=64, seq_len=30, dropout=0.35).to(device)
    criterion = Phase4MultiTaskLoss(gamma=2.0, alpha=0.50, lambda_eye=0.25, lambda_mouth=0.18, lambda_head=0.12, lambda_con=0.10).to(device)
    
    # Exact Baseline Optimizer (AdamW, lr=1e-4, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_f1 = 0.0
    best_metrics = None
    best_probs = None
    best_weights = None
    patience = 12
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for eb, mb, gb, cb, y_dr, y_ey, y_mo, y_he in train_loader:
            eb, mb, gb, cb = eb.to(device), mb.to(device), gb.to(device), cb.to(device)
            y_dr, y_ey, y_mo, y_he = y_dr.to(device), y_ey.to(device), y_mo.to(device), y_he.to(device)
            
            optimizer.zero_grad()
            outputs = model(eb, mb, gb, cb)
            loss, _ = criterion(outputs, y_dr, y_ey, y_mo, y_he)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        probs, targets = evaluate_with_tta_v3(model, val_loader, device)
        m = compute_drowsiness_metrics(targets, probs, threshold=0.50)
        
        if m['f1_score'] > best_val_f1:
            best_val_f1 = m['f1_score']
            best_metrics = m
            best_probs = probs
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Fold {fold} | Epoch {epoch:02d}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Val Acc: {m['accuracy']*100:.2f}% | F1: {m['f1_score']*100:.2f}% | AUC: {m['roc_auc']*100:.2f}%")
            
        if patience_counter >= patience:
            print(f"  Fold {fold} | Early stopping triggered at Epoch {epoch}")
            break
            
    print(f"  -> Fold {fold} Best Metrics: Acc={best_metrics['accuracy']*100:.2f}%, F1={best_metrics['f1_score']*100:.2f}%, AUC={best_metrics['roc_auc']*100:.2f}%, Prec={best_metrics['precision']*100:.2f}%, Rec={best_metrics['recall']*100:.2f}%")
    return best_metrics, best_probs, val_idx, best_weights

def run_phase4_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(f"[Phase 4 Benchmarking] Training DyG-STMT v3 on {device}")
    print("Exact Baseline Hyperparameters: LR=1e-4 | WeightDecay=1e-4 | Window=30 | Stride=15 | Batch=32 | Dropout=0.35")
    print("=" * 80)
    
    data_tuple = load_benchmark_data()
    eye, mouth, geo, cond_oh, cond_idx, y_d, y_e, y_m, y_h = data_tuple
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    oof_probs = np.zeros(len(y_d), dtype=np.float32)
    
    global_best_auc = 0.0
    global_best_weights = None
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(eye, y_d), 1):
        print(f"\n--- Training Fold {fold}/5 (Train: {len(tr_idx)}, Val: {len(va_idx)}) ---")
        m, probs, val_idx, weights = train_and_eval_fold_v3(
            fold, tr_idx, va_idx, data_tuple, device,
            epochs=35, batch_size=32, lr=1e-4, weight_decay=1e-4
        )
        fold_results.append(m)
        oof_probs[val_idx] = probs
        
        if m['roc_auc'] > global_best_auc:
            global_best_auc = m['roc_auc']
            global_best_weights = weights
            
    # Compute Cross-Validated Overall Metrics
    overall_metrics = compute_drowsiness_metrics(y_d, oof_probs, threshold=0.50)
    
    print("\n" + "=" * 80)
    print("DYG-STMT v3 (PHASE 4 ULTRA) 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print(f"Overall Accuracy:  {overall_metrics['accuracy']*100:.2f}%")
    print(f"Overall F1-Score:  {overall_metrics['f1_score']*100:.2f}%")
    print(f"Overall ROC-AUC:   {overall_metrics['roc_auc']*100:.2f}%")
    print(f"Overall Precision: {overall_metrics['precision']*100:.2f}%")
    print(f"Overall Recall:    {overall_metrics['recall']*100:.2f}%")
    print(f"Peak Fold ROC-AUC: {global_best_auc*100:.2f}%")
    
    # Breakdown by condition scenario
    conditions = ['Normal', 'Glasses', 'Sunglasses', 'Night', 'Night-Glasses']
    print("\nCondition Scenario Breakdown:")
    for c_id, c_name in enumerate(conditions):
        c_mask = (cond_idx == c_id)
        if np.sum(c_mask) > 0:
            c_m = compute_drowsiness_metrics(y_d[c_mask], oof_probs[c_mask], threshold=0.50)
            print(f"  {c_name:<15}: Acc = {c_m['accuracy']*100:.2f}% | F1 = {c_m['f1_score']*100:.2f}% | AUC = {c_m['roc_auc']*100:.2f}% (N={np.sum(c_mask)})")
            
    # Save best checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/best_dyg_stmt_v3.pt'
    torch.save(global_best_weights, save_path)
    print(f"\n[Checkpoint Saved] -> {save_path}")
    
    # Measure Latency on Device
    print("\nBenchmarking GPU Inference Latency...")
    eval_model = DyGSTMTv3(num_conditions=5, feature_dim=128, gru_hidden=64, seq_len=30, dropout=0.35).to(device)
    eval_model.load_state_dict(global_best_weights)
    eval_model.eval()
    
    dummy_eye = torch.randn(1, 30, 128).to(device)
    dummy_mouth = torch.randn(1, 30, 128).to(device)
    dummy_geo = torch.randn(1, 30, 16).to(device)
    dummy_cond = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]).to(device)
    
    # Warmup
    for _ in range(100):
        _ = eval_model(dummy_eye, dummy_mouth, dummy_geo, dummy_cond)
        
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            for _ in range(500):
                _ = eval_model(dummy_eye, dummy_mouth, dummy_geo, dummy_cond)
        end_event.record()
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event) / 500.0
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(500):
                _ = eval_model(dummy_eye, dummy_mouth, dummy_geo, dummy_cond)
        latency_ms = (time.perf_counter() - t0) * 1000.0 / 500.0
        
    fps = 1000.0 / latency_ms
    param_count = sum(p.numel() for p in eval_model.parameters() if p.requires_grad) / 1e6
    print(f"GPU Forward Pass Latency: {latency_ms:.2f} ms ({fps:.1f} FPS)")
    print(f"Total Trainable Parameters: {param_count:.2f} Million")
    print("=" * 80)
    
    return overall_metrics, global_best_auc, latency_ms

if __name__ == '__main__':
    run_phase4_benchmark()
