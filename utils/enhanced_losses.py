import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Pulls sequences of the same fatigue class together across subjects and lighting conditions,
    while pushing alert and drowsy sequences apart.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (B, D) normalized latent embeddings
        labels: (B,) binary targets (0 or 1)
        """
        device = features.device
        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Similarity matrix
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask-out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-7)
        loss = -mean_log_prob_pos.mean()
        return loss


class EnhancedMultiTaskLoss(nn.Module):
    """
    Enhanced Multi-Task Loss combining:
    1. Focal Loss with adaptive alpha weighting (gamma=2.0)
    2. Auxiliary BCE losses on Eye, Mouth, and Head prediction heads
    3. Supervised Contrastive Latent Regularization
    """
    def __init__(self, gamma=2.0, alpha=0.50, lambda_eye=0.20, lambda_mouth=0.15, lambda_head=0.10, lambda_con=0.10):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_eye = lambda_eye
        self.lambda_mouth = lambda_mouth
        self.lambda_head = lambda_head
        self.lambda_con = lambda_con
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.supcon = SupervisedContrastiveLoss(temperature=0.08)

    def forward(self, outputs, targets):
        logits = outputs['drowsy_logits']
        y_drowsy = targets['drowsy']
        
        # 1. Main Focal Loss
        bce_loss = self.bce(logits, y_drowsy)
        probs = torch.sigmoid(logits)
        pt = torch.where(y_drowsy == 1, probs, 1.0 - probs)
        alpha_t = torch.where(y_drowsy == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)
        loss_drowsy = (focal_weight * bce_loss).mean()
        
        # 2. Auxiliary Losses
        loss_eye = self.bce(outputs['aux_eye_logits'], targets['eye']).mean()
        loss_mouth = self.bce(outputs['aux_mouth_logits'], targets['mouth']).mean()
        loss_head = self.bce(outputs['aux_head_logits'], targets['head']).mean()
        
        # 3. Supervised Contrastive Loss on Latent Projections
        loss_con = torch.tensor(0.0, device=logits.device)
        if 'latent_proj' in outputs and self.lambda_con > 0:
            loss_con = self.supcon(outputs['latent_proj'], y_drowsy)
            
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
            'loss_contrastive': loss_con.item()
        }
