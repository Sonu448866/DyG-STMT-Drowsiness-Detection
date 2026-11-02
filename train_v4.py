"""
Phase 5 Training & Evaluation Script for DyG-STMT v4 (Max-Vigilance SOTA)
Strict Official Split (100% Subject-Independent, Zero Leakage):
- Train Set: 5,240 sequences (18 subjects from nthu_training)
- Eval Set:  560 sequences (4 subjects from nthu_evaluation)
- Test Set:  2,450 sequences (14 held-out unseen subjects from nthu_testing)

Exact Baseline Hyperparameters:
- Window (T): 30 frames @ 30 FPS (15-frame stride = 50% overlap)
- Optimizer: AdamW (lr = 1e-4, weight_decay = 1e-4)
- Batch Size: 32
- Dropout: 0.35
- Multi-Task Balanced Focal Loss (gamma=2.0, alpha=0.50)
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.dyg_stmt_v4 import DyGSTMTv4
from utils.losses import MultiTaskFocalLoss
from utils.metrics import compute_drowsiness_metrics

CONDITIONS = ['Normal (NoGlasses)', 'Glasses', 'Sunglasses', 'Night (NoGlasses)', 'Night (Glasses)']

def load_split(split_name):
    path = f'data_cache/features_{split_name}.npz'
    d = np.load(path)
    
    eye = d['eye_feats']
    mouth = d['mouth_feats']
    geo = d['geo_vecs']
    cond_idx = d['cond_idx']
    y_d = d['label_drowsy']
    y_e = d['label_eye']
    y_m = d['label_mouth']
    y_h = d['label_head']
    
    cond_oh = np.zeros((len(cond_idx), 5), dtype=np.float32)
    for i, c in enumerate(cond_idx):
        cond_oh[i, int(c)] = 1.0
        
    return eye, mouth, geo, cond_oh, cond_idx, y_d, y_e, y_m, y_h

def evaluate_model(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            eb, mb, gb, cb, yb = [x.to(device) for x in batch[:5]]
            out = model(eb, mb, gb, cb)
            prob = torch.sigmoid(out['drowsy_logits'])
            all_probs.extend(prob.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
            
    return np.array(all_probs), np.array(all_targets)

def train_and_evaluate_v4(epochs=35, batch_size=32, lr=1e-4, weight_decay=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 85)
    print(f"[Phase 5 Official Benchmark] Training DyG-STMT v4 on {device}")
    print(f"Strict Official Partition: Train (18 subj) | Eval (4 subj) | Test (14 held-out subj)")
    print(f"Hyperparameters: LR={lr} | WeightDecay={weight_decay} | Window=30 | Stride=15 | Batch={batch_size} | Dropout=0.35")
    print("=" * 85)
    
    # 1. Load Data
    tr_eye, tr_mouth, tr_geo, tr_cond, tr_c_idx, tr_yd, tr_ye, tr_ym, tr_yh = load_split('train')
    ev_eye, ev_mouth, ev_geo, ev_cond, ev_c_idx, ev_yd, ev_ye, ev_ym, ev_yh = load_split('eval')
    te_eye, te_mouth, te_geo, te_cond, te_c_idx, te_yd, te_ye, te_ym, te_yh = load_split('test')
    
    # 2. Balanced Sampler for Train Set
    num_pos = int(np.sum(tr_yd == 1))
    num_neg = int(np.sum(tr_yd == 0))
    weights = np.where(tr_yd == 1, 1.0 / max(1, num_pos), 1.0 / max(1, num_neg))
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True
    )
    
    train_ds = TensorDataset(
        torch.from_numpy(tr_eye).float(),
        torch.from_numpy(tr_mouth).float(),
        torch.from_numpy(tr_geo).float(),
        torch.from_numpy(tr_cond).float(),
        torch.from_numpy(tr_yd).float(),
        torch.from_numpy(tr_ye).float(),
        torch.from_numpy(tr_ym).float(),
        torch.from_numpy(tr_yh).float()
    )
    eval_ds = TensorDataset(
        torch.from_numpy(ev_eye).float(),
        torch.from_numpy(ev_mouth).float(),
        torch.from_numpy(ev_geo).float(),
        torch.from_numpy(ev_cond).float(),
        torch.from_numpy(ev_yd).float()
    )
    test_ds = TensorDataset(
        torch.from_numpy(te_eye).float(),
        torch.from_numpy(te_mouth).float(),
        torch.from_numpy(te_geo).float(),
        torch.from_numpy(te_cond).float(),
        torch.from_numpy(te_yd).float()
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 3. Model, Loss & Optimizer
    model = DyGSTMTv4(num_conditions=5, feature_dim=128, gru_hidden=64, seq_len=30, dropout=0.35).to(device)
    criterion = MultiTaskFocalLoss(gamma=2.0, alpha=0.50, lambda_eye=0.20, lambda_mouth=0.15, lambda_head=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_eval_f1 = 0.0
    best_weights = None
    patience = 12
    patience_counter = 0
    
    print("\n--- Training on Official 18-Subject Training Set ---")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for eb, mb, gb, cb, y_dr, y_ey, y_mo, y_he in train_loader:
            eb, mb, gb, cb = eb.to(device), mb.to(device), gb.to(device), cb.to(device)
            y_dr, y_ey, y_mo, y_he = y_dr.to(device), y_ey.to(device), y_mo.to(device), y_he.to(device)
            
            optimizer.zero_grad()
            outputs = model(eb, mb, gb, cb)
            targets = {
                'drowsy': y_dr,
                'eye': y_ey,
                'mouth': y_mo,
                'head': y_he
            }
            loss, _ = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        
        # Validation on Official Eval Set
        ev_probs, ev_targets = evaluate_model(model, eval_loader, device)
        ev_m = compute_drowsiness_metrics(ev_targets, ev_probs, threshold=0.50)
        
        if ev_m['f1_score'] > best_eval_f1:
            best_eval_f1 = ev_m['f1_score']
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved_flag = "*"
        else:
            patience_counter += 1
            improved_flag = ""
            
        if epoch % 2 == 0 or epoch == epochs or improved_flag:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {epoch_loss/len(train_loader):.4f} | Eval Acc: {ev_m['accuracy']*100:.2f}% | Eval F1: {ev_m['f1_score']*100:.2f}% | Eval AUC: {ev_m['roc_auc']*100:.2f}% {improved_flag}")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at Epoch {epoch} (Best Eval F1: {best_eval_f1*100:.2f}%)")
            break
            
    # 4. Save Checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/best_dyg_stmt_v4.pt'
    torch.save(best_weights, save_path)
    print(f"\n[Checkpoint Saved] -> {save_path}")
    
    # 5. Final Evaluation on Official Held-Out Test Set (14 Unseen Subjects)
    model.load_state_dict(best_weights)
    test_probs, test_targets = evaluate_model(model, test_loader, device)
    test_m = compute_drowsiness_metrics(test_targets, test_probs, threshold=0.50)
    
    print("\n" + "=" * 85)
    print("DYG-STMT v4 OFFICIAL TEST SET RESULTS (14 HELD-OUT UNSEEN SUBJECTS)")
    print("=" * 85)
    print(f"Official Test Accuracy:  {test_m['accuracy']*100:.2f}%")
    print(f"Official Test F1-Score:  {test_m['f1_score']*100:.2f}%")
    print(f"Official Test ROC-AUC:   {test_m['roc_auc']*100:.2f}%")
    print(f"Official Test Precision: {test_m['precision']*100:.2f}%")
    print(f"Official Test Recall:    {test_m['recall']*100:.2f}%")
    
    print("\nScenario-Specific Performance on Held-Out Test Subjects:")
    for c_id, c_name in enumerate(CONDITIONS):
        c_mask = (te_c_idx == c_id)
        if np.sum(c_mask) > 0:
            sc_m = compute_drowsiness_metrics(test_targets[c_mask], test_probs[c_mask], threshold=0.50)
            print(f"  {c_name:<25}: Acc = {sc_m['accuracy']*100:.2f}% | F1 = {sc_m['f1_score']*100:.2f}% | AUC = {sc_m['roc_auc']*100:.2f}% (N={np.sum(c_mask)})")
            
    print("=" * 85)
    return test_m

if __name__ == '__main__':
    train_and_evaluate_v4(epochs=30, batch_size=32, lr=1e-4, weight_decay=1e-4)
