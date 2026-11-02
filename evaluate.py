import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.dyg_stmt import DyGSTMTNet
from utils.losses import MultiTaskFocalLoss
from utils.metrics import compute_drowsiness_metrics, print_metrics_summary
from train import load_cached_dataset

CONDITION_NAMES = ['noglasses', 'glasses', 'sunglasses', 'night_noglasses', 'nightglasses']

def evaluate_model(checkpoint_path, test_cache_path, batch_size=32, device_str='auto'):
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
        
    print(f"\n[Evaluation Engine] Evaluating on Device: {device}")
    print(f"[Evaluation Engine] Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DyGSTMTNet(num_conditions=5, feature_dim=128, gru_hidden=128).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"[Evaluation Engine] Loading test dataset from: {test_cache_path}")
    test_dataset = load_cached_dataset(test_cache_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_conditions = []
    
    with torch.no_grad():
        for batch in test_loader:
            eye_b, mouth_b, geo_b, cond_b, y_drowsy, _, _, _ = [x.to(device) for x in batch]
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(eye_b, mouth_b, geo_b, cond_b)
                
            probs = torch.sigmoid(outputs['drowsy_logits']).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y_drowsy.cpu().numpy())
            all_conditions.extend(cond_b.cpu().numpy().argmax(axis=1))
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_conditions = np.array(all_conditions)
    
    # 1. Overall Metrics
    overall_metrics = compute_drowsiness_metrics(all_targets, all_preds)
    print_metrics_summary(overall_metrics, prefix="Overall NTHU-DDD Benchmark")
    
    # 2. Scenario-wise Breakdown
    print("\n---------------- Scenario-wise Breakdown ----------------")
    print(f"{'Condition':<18} | {'Accuracy':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Samples':<8}")
    print("-" * 75)
    
    scenario_metrics = {}
    for c_idx, c_name in enumerate(CONDITION_NAMES):
        mask = (all_conditions == c_idx)
        if np.sum(mask) == 0:
            continue
        c_targets = all_targets[mask]
        c_preds = all_preds[mask]
        c_met = compute_drowsiness_metrics(c_targets, c_preds)
        scenario_metrics[c_name] = c_met
        print(f"{c_name:<18} | {c_met['accuracy']*100:>8.2f}% | {c_met['f1_score']*100:>8.2f}% | {c_met['precision']*100:>8.2f}% | {c_met['recall']*100:>8.2f}% | {np.sum(mask):>8}")
    print("-" * 75 + "\n")
    
    return overall_metrics, scenario_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate DyG-STMT model on test/evaluation split")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_dyg_stmt.pt')
    parser.add_argument('--test_cache', type=str, default='data_cache/test_sequences.npz')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.test_cache, batch_size=args.batch_size)
