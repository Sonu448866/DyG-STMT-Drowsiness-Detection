import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from models.dyg_stmt_head import DyGSTMTSequenceModel
from utils.losses import MultiTaskFocalLoss
from utils.metrics import compute_drowsiness_metrics, print_metrics_summary

def load_full_benchmark():
    tr = np.load('data_cache/features_train.npz')
    ev = np.load('data_cache/features_eval.npz')
    te = np.load('data_cache/features_test.npz')
    
    eye_feats = np.concatenate([tr['eye_feats'], ev['eye_feats'], te['eye_feats']], axis=0)
    mouth_feats = np.concatenate([tr['mouth_feats'], ev['mouth_feats'], te['mouth_feats']], axis=0)
    geo_vecs = np.concatenate([tr['geo_vecs'], ev['geo_vecs'], te['geo_vecs']], axis=0)
    cond_indices = np.concatenate([tr['cond_idx'], ev['cond_idx'], te['cond_idx']], axis=0)
    
    label_drowsy = np.concatenate([tr['label_drowsy'], ev['label_drowsy'], te['label_drowsy']], axis=0)
    label_eye = np.concatenate([tr['label_eye'], ev['label_eye'], te['label_eye']], axis=0)
    label_mouth = np.concatenate([tr['label_mouth'], ev['label_mouth'], te['label_mouth']], axis=0)
    label_head = np.concatenate([tr['label_head'], ev['label_head'], te['label_head']], axis=0)
    
    cond_one_hot = np.zeros((len(cond_indices), 5), dtype=np.float32)
    for i, c in enumerate(cond_indices):
        cond_one_hot[i, int(c)] = 1.0
        
    print(f"[Benchmark Pool] Total sequences: {len(label_drowsy)} (Drowsy: {int(np.sum(label_drowsy))}, Alert: {int(len(label_drowsy) - np.sum(label_drowsy))})")
    return eye_feats, mouth_feats, geo_vecs, cond_one_hot, cond_indices, label_drowsy, label_eye, label_mouth, label_head

def train_and_eval_fold(fold, train_idx, val_idx, data_tuple, device, epochs=25, batch_size=32, lr=5e-4):
    eye_feats, mouth_feats, geo_vecs, cond_one_hot, cond_indices, label_drowsy, label_eye, label_mouth, label_head = data_tuple
    
    # Train tensors
    tr_eye = torch.from_numpy(eye_feats[train_idx]).float()
    tr_mouth = torch.from_numpy(mouth_feats[train_idx]).float()
    tr_geo = torch.from_numpy(geo_vecs[train_idx]).float()
    tr_cond = torch.from_numpy(cond_one_hot[train_idx]).float()
    tr_drowsy = torch.from_numpy(label_drowsy[train_idx]).float()
    tr_l_eye = torch.from_numpy(label_eye[train_idx]).float()
    tr_l_mouth = torch.from_numpy(label_mouth[train_idx]).float()
    tr_l_head = torch.from_numpy(label_head[train_idx]).float()
    
    # Validation tensors
    va_eye = torch.from_numpy(eye_feats[val_idx]).float()
    va_mouth = torch.from_numpy(mouth_feats[val_idx]).float()
    va_geo = torch.from_numpy(geo_vecs[val_idx]).float()
    va_cond = torch.from_numpy(cond_one_hot[val_idx]).float()
    va_drowsy = torch.from_numpy(label_drowsy[val_idx]).float()
    va_l_eye = torch.from_numpy(label_eye[val_idx]).float()
    va_l_mouth = torch.from_numpy(label_mouth[val_idx]).float()
    va_l_head = torch.from_numpy(label_head[val_idx]).float()
    
    # Balanced Sampler for Train
    y_tr = label_drowsy[train_idx]
    num_pos = int(np.sum(y_tr == 1))
    num_neg = int(np.sum(y_tr == 0))
    weight_pos = 1.0 / max(1, num_pos)
    weight_neg = 1.0 / max(1, num_neg)
    sample_weights = np.where(y_tr == 1, weight_pos, weight_neg)
    sampler = torch.utils.data.WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True)
    
    train_ds = TensorDataset(tr_eye, tr_mouth, tr_geo, tr_cond, tr_drowsy, tr_l_eye, tr_l_mouth, tr_l_head)
    val_ds = TensorDataset(va_eye, va_mouth, va_geo, va_cond, va_drowsy, va_l_eye, va_l_mouth, va_l_head)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = DyGSTMTSequenceModel(num_conditions=5, feature_dim=128, gru_hidden=64, num_layers=1, dropout=0.3).to(device)
    criterion = MultiTaskFocalLoss(gamma=2.0, alpha=0.50, lambda_eye=0.20, lambda_mouth=0.15, lambda_head=0.10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_f1 = 0.0
    best_metrics = {}
    best_preds = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            eb, mb, gb, cb, y_d, y_e, y_m, y_h = [x.to(device) for x in batch]
            optimizer.zero_grad()
            out = model(eb, mb, gb, cb)
            loss, _ = criterion(out, {'drowsy': y_d, 'eye': y_e, 'mouth': y_m, 'head': y_h})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
        scheduler.step()
        
        # Eval
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                eb, mb, gb, cb, y_d, _, _, _ = [x.to(device) for x in batch]
                out = model(eb, mb, gb, cb)
                probs = torch.sigmoid(out['drowsy_logits']).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(y_d.cpu().numpy())
                
        met = compute_drowsiness_metrics(all_targets, all_preds)
        if met['f1_score'] > best_val_f1:
            best_val_f1 = met['f1_score']
            best_metrics = met
            best_preds = np.array(all_preds)
            if fold == 1:
                torch.save({'model_state_dict': model.state_dict(), 'val_metrics': met}, 'checkpoints/best_dyg_stmt.pt')
                
    print(f"Fold {fold} Best: Acc = {best_metrics['accuracy']*100:.2f}% | F1 = {best_metrics['f1_score']*100:.2f}% | ROC-AUC = {best_metrics['roc_auc']*100:.2f}% | Recall = {best_metrics['recall']*100:.2f}% | Specificity = {best_metrics['specificity']*100:.2f}%")
    return best_metrics, best_preds

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n============================================================")
    print(f"  Running 5-Fold Cross-Validation for DyG-STMT Benchmark ({device})")
    print(f"============================================================\n")
    
    data_tuple = load_full_benchmark()
    label_drowsy = data_tuple[5]
    cond_indices = data_tuple[4]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_fold_metrics = []
    oof_preds = np.zeros(len(label_drowsy))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(label_drowsy)), label_drowsy), 1):
        print(f"\n--- Training Fold {fold}/5 ({len(train_idx)} train, {len(val_idx)} val) ---")
        best_met, best_preds = train_and_eval_fold(fold, train_idx, val_idx, data_tuple, device, epochs=25, batch_size=32, lr=5e-4)
        all_fold_metrics.append(best_met)
        oof_preds[val_idx] = best_preds
        
    print("\n" + "="*70)
    print("                5-FOLD BENCHMARK SUMMARY FOR DyG-STMT")
    print("="*70)
    
    mean_acc = np.mean([m['accuracy'] for m in all_fold_metrics]) * 100
    mean_f1 = np.mean([m['f1_score'] for m in all_fold_metrics]) * 100
    mean_prec = np.mean([m['precision'] for m in all_fold_metrics]) * 100
    mean_rec = np.mean([m['recall'] for m in all_fold_metrics]) * 100
    mean_spec = np.mean([m['specificity'] for m in all_fold_metrics]) * 100
    mean_auc = np.mean([m['roc_auc'] for m in all_fold_metrics]) * 100
    
    std_acc = np.std([m['accuracy'] for m in all_fold_metrics]) * 100
    std_f1 = np.std([m['f1_score'] for m in all_fold_metrics]) * 100
    std_auc = np.std([m['roc_auc'] for m in all_fold_metrics]) * 100
    
    print(f"  Accuracy    : {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"  F1-Score    : {mean_f1:.2f}% +/- {std_f1:.2f}%")
    print(f"  Precision   : {mean_prec:.2f}%")
    print(f"  Recall      : {mean_rec:.2f}%")
    print(f"  Specificity : {mean_spec:.2f}%")
    print(f"  ROC-AUC     : {mean_auc:.2f}% +/- {std_auc:.2f}%")
    print("="*70)
    
    # Condition breakdown on OOF predictions
    print("\n---------------- Condition-wise Performance Breakdown ----------------")
    cond_names = ['noglasses', 'glasses', 'sunglasses', 'night_noglasses', 'nightglasses']
    print(f"{'Condition':<18} | {'Accuracy':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Samples'}")
    print("-" * 75)
    for c_idx, c_name in enumerate(cond_names):
        mask = (cond_indices == c_idx)
        if np.sum(mask) > 0:
            c_met = compute_drowsiness_metrics(label_drowsy[mask], oof_preds[mask])
            print(f"{c_name:<18} | {c_met['accuracy']*100:8.2f}% | {c_met['f1_score']*100:8.2f}% | {c_met['precision']*100:8.2f}% | {c_met['recall']*100:8.2f}% | {np.sum(mask):6d}")
    print("-" * 75)

if __name__ == '__main__':
    main()
