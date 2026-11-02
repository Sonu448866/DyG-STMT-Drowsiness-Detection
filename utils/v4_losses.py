"""
Loss Functions for DyG-STMT v4 (Phase 5 Max-Vigilance Ultra SOTA):
1. Asymmetric Focal Loss (ASL) with class-specific gamma focusing:
   - gamma_pos = 1.5 (prevents over-penalizing hard fatigued micro-events)
   - gamma_neg = 2.5 (heavily suppresses easy alert driving frames / glances)
2. Angular-Margin Supervised Contrastive Regularization (Arc-SupCon, m=0.20, tau=0.07)
3. Multi-Task Supervision on frame-level eye blinks, mouth yawns, and head micro-nods
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss (ASL):
    Differentiates focusing dynamics between positive (drowsy) and negative (alert) instances.
    """
    def __init__(self, gamma_neg=2.5, gamma_pos=1.5, clip=0.05, alpha=0.50):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.alpha = alpha

    def forward(self, logits, targets):
        # targets: (B,) binary 0 or 1
        probs = torch.sigmoid(logits)
        
        # Asymmetric probability clipping for negative class
        p_neg = probs
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)
            
        # Compute loss components
        loss_pos = targets * torch.log(probs.clamp(min=1e-8)) * torch.pow(1.0 - probs, self.gamma_pos)
        loss_neg = (1.0 - targets) * torch.log((1.0 - p_neg).clamp(min=1e-8)) * torch.pow(p_neg, self.gamma_neg)
        
        loss = - (self.alpha * loss_pos + (1.0 - self.alpha) * loss_neg)
        return loss.mean()


class ArcSupConLoss(nn.Module):
    """
    Angular-Margin Supervised Contrastive Loss (Arc-SupCon):
    Enforces strict angular separation margin m=0.20 on the latent unit hypersphere.
    """
    def __init__(self, temperature=0.07, margin=0.20):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, projections, targets):
        device = projections.device
        batch_size = projections.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)
        
        sim_matrix = torch.matmul(projections, projections.T)
        targets_col = targets.contiguous().view(-1, 1)
        pos_mask = torch.eq(targets_col, targets_col.T).float().to(device)
        
        diag_mask = torch.eye(batch_size, device=device).bool()
        pos_mask.masked_fill_(diag_mask, 0.0)
        
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        cos_sim = torch.clamp(sim_matrix, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_sim)
        marginal_sim = torch.cos(theta + self.margin * pos_mask)
        
        logits = marginal_sim / self.temperature
        logits_mask = torch.ones_like(sim_matrix) - torch.eye(batch_size, device=device)
        exp_logits = torch.exp(logits) * logits_mask
        
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-7)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-7)
        loss = -mean_log_prob_pos.mean()
        return loss


class Phase5MultiTaskLoss(nn.Module):
    """
    Phase 5 Multi-Task Loss:
    L_total = L_asl(drowsy) + 0.25*L_bce(eye) + 0.18*L_bce(mouth) + 0.12*L_bce(head) + 0.12*L_arc_supcon
    """
    def __init__(self, lambda_eye=0.25, lambda_mouth=0.18, lambda_head=0.12, lambda_con=0.12):
        super().__init__()
        self.asl = AsymmetricFocalLoss(gamma_neg=2.5, gamma_pos=1.5, clip=0.05, alpha=0.50)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.arc_supcon = ArcSupConLoss(temperature=0.07, margin=0.20)
        
        self.lambda_eye = lambda_eye
        self.lambda_mouth = lambda_mouth
        self.lambda_head = lambda_head
        self.lambda_con = lambda_con

    def forward(self, outputs, y_drowsy, y_eye, y_mouth, y_head):
        # 1. Primary Drowsiness Asymmetric Focal Loss
        loss_drowsy = self.asl(outputs['drowsy_logits'], y_drowsy)
        
        # 2. Auxiliary Frame-level Losses
        loss_eye = self.bce(outputs['aux_eye_logits'], y_eye).mean()
        loss_mouth = self.bce(outputs['aux_mouth_logits'], y_mouth).mean()
        loss_head = self.bce(outputs['aux_head_logits'], y_head).mean()
        
        # 3. Arc-SupCon Regularization
        loss_con = torch.tensor(0.0, device=y_drowsy.device)
        if 'latent_proj' in outputs and self.lambda_con > 0:
            loss_con = self.arc_supcon(outputs['latent_proj'], y_drowsy)
            
        total_loss = (
            loss_drowsy
            + self.lambda_eye * loss_eye
            + self.lambda_mouth * loss_mouth
            + self.lambda_head * loss_head
            + self.lambda_con * loss_con
        )
        
        return total_loss, {
            'loss_drowsy': loss_drowsy.item(),
            'loss_eye': loss_eye.item(),
            'loss_mouth': loss_mouth.item(),
            'loss_head': loss_head.item(),
            'loss_con': loss_con.item(),
            'total_loss': total_loss.item()
        }
