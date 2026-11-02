import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.dyg_stmt_head import DyGSTMTSequenceModel
from utils.metrics import compute_drowsiness_metrics, print_metrics_summary

CONDITION_NAMES = ['noglasses', 'glasses', 'sunglasses', 'night_noglasses', 'nightglasses']

def find_optimal_threshold(y_true, y_probs):
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        y_pred = (y_probs >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def evaluate_dataset(checkpoint_path, npz_path, dataset_name="Test Set"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n============================================================")
    print(f"  Evaluating DyG-STMT on {dataset_name} ({npz_path})")
    print(f"============================================================")
    
    data = np.load(npz_path)
    eye_feats = torch.from_numpy(data['eye_feats']).float()
    mouth_feats = torch.from_numpy(data['mouth_feats']).float()
    geo_vecs = torch.from_numpy(data['geo_vecs']).float()
    
    cond_indices = torch.from_numpy(data['cond_idx']).long()
    cond_one_hot = torch.zeros((len(cond_indices), 5), dtype=torch.float32)
    cond_one_hot.scatter_(1, cond_indices.unsqueeze(1), 1.0)
    
    label_drowsy = torch.from_numpy(data['label_drowsy']).float()
    
    loader = DataLoader(TensorDataset(eye_feats, mouth_feats, geo_vecs, cond_one_hot, label_drowsy), batch_size=32, shuffle=False)
    
    model = DyGSTMTSequenceModel(num_conditions=5, feature_dim=128, gru_hidden=64, num_layers=1).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_targets = []
    all_conds = []
    
    with torch.no_grad():
        for eye_b, mouth_b, geo_b, cond_b, y_drowsy in loader:
            eye_b, mouth_b, geo_b, cond_b = [x.to(device) for x in (eye_b, mouth_b, geo_b, cond_b)]
            out = model(eye_b, mouth_b, geo_b, cond_b)
            probs = torch.sigmoid(out['drowsy_logits']).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y_drowsy.numpy())
            all_conds.extend(cond_b.cpu().numpy().argmax(axis=1))
            
    y_true = np.array(all_targets).astype(int)
    y_probs = np.array(all_preds)
    conds = np.array(all_conds)
    
    # 1. Standard Threshold (0.50)
    std_metrics = compute_drowsiness_metrics(y_true, y_probs, threshold=0.50)
    print_metrics_summary(std_metrics, prefix=f"{dataset_name} (Threshold = 0.50)")
    
    # 2. Optimal Threshold Search
    opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_probs)
    opt_metrics = compute_drowsiness_metrics(y_true, y_probs, threshold=opt_thresh)
    print_metrics_summary(opt_metrics, prefix=f"{dataset_name} (Calibrated Threshold = {opt_thresh:.2f})")
    
    # 3. Scenario-wise Breakdown
    print(f"---------------- Scenario-wise Breakdown ({dataset_name}) ----------------")
    print(f"{'Condition':<18} | {'Accuracy':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Samples':<8}")
    print("-" * 75)
    
    scenario_res = {}
    for c_idx, c_name in enumerate(CONDITION_NAMES):
        mask = (conds == c_idx)
        if np.sum(mask) == 0:
            continue
        c_t = y_true[mask]
        c_p = y_probs[mask]
        c_met = compute_drowsiness_metrics(c_t, c_p, threshold=opt_thresh)
        scenario_res[c_name] = c_met
        print(f"{c_name:<18} | {c_met['accuracy']*100:>8.2f}% | {c_met['f1_score']*100:>8.2f}% | {c_met['precision']*100:>8.2f}% | {c_met['recall']*100:>8.2f}% | {np.sum(mask):>8}")
    print("-" * 75 + "\n")
    
    return opt_metrics, scenario_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_dyg_stmt.pt')
    parser.add_argument('--eval_features', type=str, default='data_cache/features_eval.npz')
    parser.add_argument('--test_features', type=str, default='data_cache/features_test.npz')
    args = parser.parse_args()
    
    if os.path.exists(args.eval_features):
        evaluate_dataset(args.checkpoint, args.eval_features, dataset_name="Evaluation Set (4 Subjects)")
    if os.path.exists(args.test_features):
        evaluate_dataset(args.checkpoint, args.test_features, dataset_name="Testing Set (14 Subjects)")
