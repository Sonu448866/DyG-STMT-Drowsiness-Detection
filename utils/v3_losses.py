"""
Enhanced Multi-Task Loss for DyG-STMT v3 (Phase 4 Ultra):
1. Binary Focal Loss with Adaptive Alpha-Gamma Hard-Mining (gamma=2.0, alpha=0.50)
2. Angular-Margin Supervised Contrastive Regularization (Arc-SupCon, tau=0.07, margin=0.15)
3. Auxiliary Multi-Task Supervision on Eye Closure, Mouth Yawning, and Head Nodding
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AngularSupervisedContrastiveLoss(nn.Module):
    """
    Angular-Margin Supervised Contrastive Loss (Arc-SupCon):
    Pulls representations of identical fatigue classes together on a unit hypersphere
    while pushing opposite classes away by an enforced angular margin m.
    """
    def __init__(self, temperature=0.07, margin=0.12):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, projections, targets):
        device = projections.device
        batch_size = projections.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)
        
        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(projections, projections.T) # (B, B)
        
        # Positive pairs mask
        targets_col = targets.contiguous().view(-1, 1)
        pos_mask = torch.eq(targets_col, targets_col.T).float().to(device)
        
        # Remove self-contrast
        diag_mask = torch.eye(batch_size, device=device).bool()
        pos_mask.masked_fill_(diag_mask, 0.0)
        
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # Enforce angular margin
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


class Phase4MultiTaskLoss(nn.Module):
    """
    Phase 4 Multi-Task Loss:
    L_total = L_focal(drowsy) + 0.25*L_bce(eye) + 0.18*L_bce(mouth) + 0.12*L_bce(head) + 0.10*L_arc_supcon
    """
    def __init__(self, gamma=2.0, alpha=0.50, lambda_eye=0.25, lambda_mouth=0.18, lambda_head=0.12, lambda_con=0.10):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_eye = lambda_eye
        self.lambda_mouth = lambda_mouth
        self.lambda_head = lambda_head
        self.lambda_con = lambda_con
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.arc_supcon = AngularSupervisedContrastiveLoss(temperature=0.07, margin=0.12)

    def forward(self, outputs, y_drowsy, y_eye, y_mouth, y_head):
        # 1. Primary Drowsiness Focal Loss
        logits = outputs['drowsy_logits']
        bce_drowsy = self.bce(logits, y_drowsy)
        probs = torch.sigmoid(logits)
        pt = torch.where(y_drowsy == 1, probs, 1.0 - probs)
        alpha_t = torch.where(y_drowsy == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)
        loss_drowsy = (focal_weight * bce_drowsy).mean()
        
        # 2. Auxiliary Losses on Frame-level Targets
        loss_eye = self.bce(outputs['aux_eye_logits'], y_eye).mean()
        loss_mouth = self.bce(outputs['aux_mouth_logits'], y_mouth).mean()
        loss_head = self.bce(outputs['aux_head_logits'], y_head).mean()
        
        # 3. Arc-SupCon Regularization
        loss_con = torch.tensor(0.0, device=logits.device)
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
            'loss_contrastive': loss_con.item(),
            'total_loss': total_loss.item()
        }
