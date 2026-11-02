import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.dyg_stmt import DyGSTMTNet
from utils.losses import MultiTaskFocalLoss
from utils.metrics import compute_drowsiness_metrics
from train import load_cached_dataset, train_one_epoch, evaluate

class AblativeDyGSTMTNet(nn.Module):
    """
    Configurable Ablation Model allowing removal of key components:
    - remove_cross_attention: uses simple concatenation instead of MHA
    - remove_gating: disables dynamic reliability gating
    - remove_film: bypasses condition-aware FiLM modulation
    - remove_temporal_attn: uses average pooling instead of attention pooling
    - modality_mode: 'all', 'eye_only', 'mouth_only', 'head_only'
    """
    def __init__(self, remove_cross_attention=False, remove_gating=False, remove_film=False, remove_temporal_attn=False, modality_mode='all'):
        super().__init__()
        self.remove_cross_attention = remove_cross_attention
        self.remove_gating = remove_gating
        self.remove_film = remove_film
        self.remove_temporal_attn = remove_temporal_attn
        self.modality_mode = modality_mode
        
        # Base components
        self.base_model = DyGSTMTNet(num_conditions=5, feature_dim=128, gru_hidden=128)
        
        if remove_cross_attention:
            self.concat_proj = nn.Linear(128 * 3, 128)

    def forward(self, eye_imgs, mouth_imgs, geo_vectors, condition_vec):
        B, T, C, H, W = eye_imgs.shape
        
        eye_flat = eye_imgs.view(B * T, C, H, W)
        mouth_flat = mouth_imgs.view(B * T, C, H, W)
        geo_flat = geo_vectors.view(B * T, 16)
        
        e_feat = self.base_model.eye_encoder(eye_flat).view(B, T, 128)
        m_feat = self.base_model.mouth_encoder(mouth_flat).view(B, T, 128)
        h_feat = self.base_model.geo_encoder(geo_flat).view(B, T, 128)
        
        # FiLM Condition Modulation
        if not self.remove_film:
            e_feat = self.base_model.film_eye(e_feat, condition_vec)
            m_feat = self.base_model.film_mouth(m_feat, condition_vec)
            h_feat = self.base_model.film_geo(h_feat, condition_vec)
            
        # Modality selection for solo ablations
        if self.modality_mode == 'eye_only':
            fused_spatial = e_feat
        elif self.modality_mode == 'mouth_only':
            fused_spatial = m_feat
        elif self.modality_mode == 'head_only':
            fused_spatial = h_feat
        else:
            if self.remove_cross_attention:
                # Baseline concatenation
                concat_tokens = torch.cat([e_feat, m_feat, h_feat], dim=-1) # (B, T, 384)
                fused_spatial = self.concat_proj(concat_tokens)
            elif self.remove_gating:
                # Cross-attention without dynamic gating
                eye_flat_t = e_feat.reshape(B * T, 1, 128)
                mouth_flat_t = m_feat.reshape(B * T, 1, 128)
                head_flat_t = h_feat.reshape(B * T, 1, 128)
                tokens = torch.cat([eye_flat_t, mouth_flat_t, head_flat_t], dim=1)
                attn_out, _ = self.base_model.cross_modal_fusion.mha(tokens, tokens, tokens)
                fused_spatial = attn_out.mean(dim=1).reshape(B, T, 128)
            else:
                fused_spatial, _ = self.base_model.cross_modal_fusion(e_feat, m_feat, h_feat)
                
        # Temporal Modeling
        gru_out, _ = self.base_model.temporal_encoder.bigru(fused_spatial)
        
        if self.remove_temporal_attn:
            temp_context = torch.mean(gru_out, dim=1)
        else:
            score = torch.tanh(self.base_model.temporal_encoder.attn_dense(gru_out))
            weights = torch.softmax(self.base_model.temporal_encoder.attn_vec(score), dim=1)
            temp_context = torch.sum(gru_out * weights, dim=1)
            
        drowsy_logits = self.base_model.main_head(temp_context).squeeze(-1)
        aux_eye_logits = self.base_model.aux_eye_head(gru_out).squeeze(-1)
        aux_mouth_logits = self.base_model.aux_mouth_head(gru_out).squeeze(-1)
        aux_head_logits = self.base_model.aux_head_head(gru_out).squeeze(-1)
        
        return {
            'drowsy_logits': drowsy_logits,
            'aux_eye_logits': aux_eye_logits,
            'aux_mouth_logits': aux_mouth_logits,
            'aux_head_logits': aux_head_logits
        }


def run_ablation_experiment(name, model, train_loader, val_loader, criterion, epochs=25, lr=1e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    best_f1 = 0.0
    best_acc = 0.0
    best_metrics = {}
    
    print(f"\n--- Running Ablation Variant: [{name}] ---")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        
        if val_metrics['f1_score'] > best_f1:
            best_f1 = val_metrics['f1_score']
            best_acc = val_metrics['accuracy']
            best_metrics = val_metrics
            
    print(f"[{name}] Completed -> Best Val Accuracy: {best_acc*100:.2f}%, Best Val F1: {best_f1*100:.2f}%, AUC: {best_metrics.get('roc_auc', 0)*100:.2f}%")
    return best_metrics


def run_full_ablation_study(train_cache, val_cache, output_md="Ablation_Study_Results.md"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("[Ablation] Loading data shards...")
    train_dataset = load_cached_dataset(train_cache)
    val_dataset = load_cached_dataset(val_cache)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    standard_criterion = MultiTaskFocalLoss(gamma=2.0, alpha=0.65, lambda_eye=0.25, lambda_mouth=0.20, lambda_head=0.15)
    no_aux_criterion = MultiTaskFocalLoss(gamma=2.0, alpha=0.65, lambda_eye=0.0, lambda_mouth=0.0, lambda_head=0.0)
    
    variants = [
        ("Full Proposed Architecture (DyG-STMT)", AblativeDyGSTMTNet(), standard_criterion),
        ("w/o Cross-Modal Attention (Concat Fusion)", AblativeDyGSTMTNet(remove_cross_attention=True), standard_criterion),
        ("w/o Dynamic Reliability Gating", AblativeDyGSTMTNet(remove_gating=True), standard_criterion),
        ("w/o FiLM Condition Modulation", AblativeDyGSTMTNet(remove_film=True), standard_criterion),
        ("w/o Temporal Self-Attention (Mean Pool)", AblativeDyGSTMTNet(remove_temporal_attn=True), standard_criterion),
        ("w/o Multi-Task Auxiliary Loss (λ=0)", AblativeDyGSTMTNet(), no_aux_criterion),
        ("Single Modality: Eye Only", AblativeDyGSTMTNet(modality_mode='eye_only'), no_aux_criterion),
        ("Single Modality: Mouth Only", AblativeDyGSTMTNet(modality_mode='mouth_only'), no_aux_criterion),
        ("Single Modality: Head Pose Only", AblativeDyGSTMTNet(modality_mode='head_only'), no_aux_criterion),
    ]
    
    results = []
    for name, model, crit in variants:
        met = run_ablation_experiment(name, model, train_loader, val_loader, crit, epochs=20, lr=1e-4, device=device)
        results.append((name, met))
        
    # Write Markdown Ablation Table
    print("\n==================== ABLATION STUDY SUMMARY TABLE ====================")
    lines = [
        "# 🔬 Systematic Ablation Study Results\n",
        "> Benchmark: NTHU-DDD Dataset | Framework: DyG-STMT\n\n",
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
        
    print(f"\nAblation report successfully written to {output_md}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run systematic ablation study for DyG-STMT")
    parser.add_argument('--train_cache', type=str, default='data_cache/train_sequences.npz')
    parser.add_argument('--val_cache', type=str, default='data_cache/eval_sequences.npz')
    parser.add_argument('--output_md', type=str, default='Ablation_Study_Results.md')
    args = parser.parse_args()
    
    run_full_ablation_study(args.train_cache, args.val_cache, args.output_md)
