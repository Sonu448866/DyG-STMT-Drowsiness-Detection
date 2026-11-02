import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskFocalLoss(nn.Module):
    """
    Combined Multi-Task Loss with Focal Loss for the main Drowsiness classification
    and Binary Cross-Entropy with Label Smoothing for the auxiliary tasks.
    
    L_total = L_focal(drowsy) + λ_eye * L_bce(eye) + λ_mouth * L_bce(mouth) + λ_head * L_bce(head)
    """
    def __init__(self, gamma=2.0, alpha=0.65, lambda_eye=0.25, lambda_mouth=0.20, lambda_head=0.15, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_eye = lambda_eye
        self.lambda_mouth = lambda_mouth
        self.lambda_head = lambda_head
        self.label_smoothing = label_smoothing

    def focal_loss(self, logits, targets):
        """
        Numerically stable binary focal loss using BCEWithLogits.
        """
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = torch.pow(1.0 - p_t + 1e-6, self.gamma)
        
        focal = alpha_factor * modulating_factor * bce_loss
        return focal.mean()

    def forward(self, outputs, targets):
        """
        outputs: dict with 'drowsy_logits', 'aux_eye_logits', 'aux_mouth_logits', 'aux_head_logits'
        targets: dict with 'drowsy', 'eye', 'mouth', 'head'
        """
        loss_main = self.focal_loss(outputs['drowsy_logits'], targets['drowsy'])
        
        # Auxiliary losses averaged across time steps
        loss_eye = F.binary_cross_entropy_with_logits(outputs['aux_eye_logits'], targets['eye'].float())
        loss_mouth = F.binary_cross_entropy_with_logits(outputs['aux_mouth_logits'], targets['mouth'].float())
        loss_head = F.binary_cross_entropy_with_logits(outputs['aux_head_logits'], targets['head'].float())
        
        total_loss = loss_main + self.lambda_eye * loss_eye + self.lambda_mouth * loss_mouth + self.lambda_head * loss_head
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'loss_main': loss_main.item(),
            'loss_eye': loss_eye.item(),
            'loss_mouth': loss_mouth.item(),
            'loss_head': loss_head.item()
        }
