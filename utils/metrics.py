import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

def compute_drowsiness_metrics(y_true, y_probs, threshold=0.5):
    """
    Computes comprehensive evaluation metrics for drowsiness detection benchmark:
    - Accuracy
    - Precision
    - Recall (Sensitivity)
    - Specificity
    - F1-Score
    - ROC-AUC
    - PR-AUC (Average Precision)
    - Confusion Matrix (TN, FP, FN, TP)
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_true, y_probs)
    except Exception:
        pr_auc = prec
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / (tn + fp + 1e-8)
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'specificity': float(spec),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': {
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        }
    }

def print_metrics_summary(metrics, prefix="Evaluation"):
    print(f"\n==================== {prefix} Results ====================")
    print(f"  Accuracy    : {metrics['accuracy'] * 100:.2f}%")
    print(f"  F1-Score    : {metrics['f1_score'] * 100:.2f}%")
    print(f"  Precision   : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall      : {metrics['recall'] * 100:.2f}%")
    print(f"  Specificity : {metrics['specificity'] * 100:.2f}%")
    print(f"  ROC-AUC     : {metrics['roc_auc'] * 100:.2f}%")
    print(f"  PR-AUC      : {metrics['pr_auc'] * 100:.2f}%")
    cm = metrics['confusion_matrix']
    print(f"  Confusion Matrix: TP={cm['TP']} | FP={cm['FP']} | TN={cm['TN']} | FN={cm['FN']}")
    print("============================================================\n")
