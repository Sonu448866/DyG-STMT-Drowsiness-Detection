import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.dyg_stmt_head import DyGSTMTSequenceModel
from utils.losses import MultiTaskFocalLoss
from utils.metrics import compute_drowsiness_metrics
from train_features import load_feature_dataset, train_one_epoch, evaluate_features
from evaluate_features import find_optimal_threshold

class AblativeSequenceModel(nn.Module):
    def __init__(self, remove_cross_attention=False, remove_gating=False, remove_film=False, remove_temporal_attn=False, modality_mode='all'):
        super().__init__()
        self.remove_cross_attention = remove_cross_attention
        self.remove_gating = remove_gating
        self.remove_film = remove_film
        self.remove_temporal_attn = remove_temporal_attn
        self.modality_mode = modality_mode
        
        self.base = DyGSTMTSequenceModel(num_conditions=5, feature_dim=128, gru_hidden=128)
        if remove_cross_attention:
            self.concat_proj = nn.Linear(128 * 3, 128)

    def forward(self, eye_feats, mouth_feats, geo_vectors, condition_vec):
        B, T, _ = eye_feats.shape
        geo_flat = geo_vectors.view(B * T, 16)
        h_feat = self.base.geo_encoder(geo_flat).view(B, T, 128)
        
        if not self.remove_film:
            e_feat = self.base.film_eye(eye_feats, condition_vec)
            m_feat = self.base.film_mouth(mouth_feats, condition_vec)
            h_feat = self.base.film_geo(h_feat, condition_vec)
        else:
            e_feat = eye_feats
            m_feat = mouth_feats
            
        if self.modality_mode == 'eye_only':
            fused_spatial = e_feat
        elif self.modality_mode == 'mouth_only':
            fused_spatial = m_feat
        elif self.modality_mode == 'head_only':
            fused_spatial = h_feat
        else:
            if self.remove_cross_attention:
                concat = torch.cat([e_feat, m_feat, h_feat], dim=-1)
                fused_spatial = self.concat_proj(concat)
            elif self.remove_gating:
                e_t = e_feat.reshape(B * T, 1, 128)
                m_t = m_feat.reshape(B * T, 1, 128)
                h_t = h_feat.reshape(B * T, 1, 128)
                tokens = torch.cat([e_t, m_t, h_t], dim=1)
                attn_out, _ = self.base.cross_modal_fusion.mha(tokens, tokens, tokens)
                fused_spatial = attn_out.mean(dim=1).reshape(B, T, 128)
            else:
                fused_spatial, _ = self.base.cross_modal_fusion(e_feat, m_feat, h_feat)
                
        gru_out, _ = self.base.temporal_encoder.bigru(fused_spatial)
        
        if self.remove_temporal_attn:
            temp_context = torch.mean(gru_out, dim=1)
        else:
            score = torch.tanh(self.base.temporal_encoder.attn_dense(gru_out))
            weights = torch.softmax(self.base.temporal_encoder.attn_vec(score), dim=1)
            temp_context = torch.sum(gru_out * weights, dim=1)
            
        drowsy_logits = self.base.main_head(temp_context).squeeze(-1)
        aux_eye_logits = self.base.aux_eye_head(gru_out).squeeze(-1)
        aux_mouth_logits = self.base.aux_mouth_head(gru_out).squeeze(-1)
        aux_head_logits = self.base.aux_head_head(gru_out).squeeze(-1)
        
        return {
            'drowsy_logits': drowsy_logits,
            'aux_eye_logits': aux_eye_logits,
            'aux_mouth_logits': aux_mouth_logits,
            'aux_head_logits': aux_head_logits
        }


def run_ablation_study(train_path='data_cache/features_train.npz', eval_path='data_cache/features_eval.npz', output_md='Ablation_Study_Results.md'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n============================================================")
    print(f"  Running Systematic Ablation Study for DyG-STMT ({device})")
    print("============================================================\n")
    
    train_ds = load_feature_dataset(train_path)
    val_ds = load_feature_dataset(eval_path)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    std_crit = MultiTaskFocalLoss(gamma=2.0, alpha=0.65, lambda_eye=0.25, lambda_mouth=0.20, lambda_head=0.15)
    no_aux_crit = MultiTaskFocalLoss(gamma=2.0, alpha=0.65, lambda_eye=0.0, lambda_mouth=0.0, lambda_head=0.0)
    
    variants = [
        ("Full Proposed Architecture (DyG-STMT)", AblativeSequenceModel(), std_crit),
        ("w/o Cross-Modal Attention (Concat Fusion)", AblativeSequenceModel(remove_cross_attention=True), std_crit),
        ("w/o Dynamic Reliability Gating", AblativeSequenceModel(remove_gating=True), std_crit),
        ("w/o FiLM Condition Modulation", AblativeSequenceModel(remove_film=True), std_crit),
        ("w/o Temporal Self-Attention (Mean Pool)", AblativeSequenceModel(remove_temporal_attn=True), std_crit),
        ("w/o Multi-Task Auxiliary Loss (λ=0)", AblativeSequenceModel(), no_aux_crit),
        ("Single Modality: Eye Only", AblativeSequenceModel(modality_mode='eye_only'), no_aux_crit),
        ("Single Modality: Mouth Only", AblativeSequenceModel(modality_mode='mouth_only'), no_aux_crit),
        ("Single Modality: Head Pose Only", AblativeSequenceModel(modality_mode='head_only'), no_aux_crit),
    ]
    
    results = []
    for name, model, crit in variants:
        print(f"\n--- Testing Configuration: [{name}] ---")
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        best_met = {}
        best_f1 = 0.0
        
        for epoch in range(1, 16):
            _ = train_one_epoch(model, train_loader, opt, crit, device)
            met, probs, targets = evaluate_features(model, val_loader, crit, device)
            
            # Calibrated metric check
            opt_th, f1_opt = find_optimal_threshold(np.array(targets), np.array(probs))
            opt_met = compute_drowsiness_metrics(targets, probs, threshold=opt_th)
            
            if opt_met['f1_score'] > best_f1:
                best_f1 = opt_met['f1_score']
                best_met = opt_met
                
        print(f"[{name}] -> Accuracy: {best_met['accuracy']*100:.2f}%, F1: {best_met['f1_score']*100:.2f}%, ROC-AUC: {best_met['roc_auc']*100:.2f}%")
        results.append((name, best_met))
        
    print("\n==================== ABLATION STUDY SUMMARY ====================")
    lines = [
        "# 🔬 Systematic Ablation Study Results — DyG-STMT\n",
        "> **Dataset:** NTHU-DDD Benchmark | **Evaluation:** Calibrated Metrics on 4 Evaluation Subjects\n\n",
        "| Architecture Configuration | Accuracy | F1-Score | Precision | Recall | ROC-AUC | Δ Accuracy (Drop) |\n",
        "|---|---|---|---|---|---|---|\n"
    ]
    
    base_acc = results[0][1]['accuracy']
    for name, met in results:
        drop = met['accuracy'] - base_acc
        drop_str = "0.00% (Baseline)" if drop == 0 else f"{drop*100:+.2f}%"
        row = f"| **{name}** | **{met['accuracy']*100:.2f}%** | {met['f1_score']*100:.2f}% | {met['precision']*100:.2f}% | {met['recall']*100:.2f}% | {met['roc_auc']*100:.2f}% | {drop_str} |\n"
        lines.append(row)
        print(row.strip())
        
    with open(output_md, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"\nAblation table saved to {output_md}\n")


if __name__ == '__main__':
    run_ablation_study()
