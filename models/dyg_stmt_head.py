import torch
import torch.nn as nn
import torch.nn.functional as F
from .dyg_stmt import FiLMConditionModulation, GeometricKinematicMLP, CrossModalAttentionGating, BiGRUTemporalAttention

class MultiScaleTemporalConv(nn.Module):
    """
    Multi-Scale 1D Temporal Convolutions to extract frequency-specific micro-temporal features:
    - Kernel 1: Instantaneous frame representation
    - Kernel 3: Rapid micro-blinks (100ms)
    - Kernel 5: Head nodding dynamics (160ms)
    - Kernel 7: Yawning expansions (230ms)
    """
    def __init__(self, in_dim=128, out_dim=128):
        super().__init__()
        branch_dim = out_dim // 4
        self.conv1 = nn.Conv1d(in_dim, branch_dim, kernel_size=1)
        self.conv3 = nn.Conv1d(in_dim, branch_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_dim, branch_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_dim, branch_dim, kernel_size=7, padding=3)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        x_t = x.transpose(1, 2)
        c1 = self.conv1(x_t)
        c3 = self.conv3(x_t)
        c5 = self.conv5(x_t)
        c7 = self.conv7(x_t)
        concat = torch.cat([c1, c3, c5, c7], dim=1)
        out = self.act(self.norm(concat)) + x_t
        return out.transpose(1, 2)


class DyGSTMTSequenceModel(nn.Module):
    """
    Core Sequence & Multi-Task Model for DyG-STMT.
    Takes pre-extracted spatial embeddings e_t, m_t, h_t and driving condition c,
    and applies:
    1. FiLM condition modulation
    2. Cross-Modal Attention with Dynamic Reliability Gating
    3. Multi-Scale Temporal Convolutions (MS-TCN)
    4. 2-Layer BiGRU with Additive Temporal Attention
    5. Main Drowsiness Head + 3 Multi-Task Heads
    """
    def __init__(self, num_conditions=5, feature_dim=128, gru_hidden=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 16-D Geometric/Kinematic MLP
        self.geo_encoder = GeometricKinematicMLP(in_dim=16, out_dim=feature_dim)
        
        # FiLM Condition Modulations
        self.film_eye = FiLMConditionModulation(num_conditions, feature_dim)
        self.film_mouth = FiLMConditionModulation(num_conditions, feature_dim)
        self.film_geo = FiLMConditionModulation(num_conditions, feature_dim)
        
        # Cross-Modal Scaled Dot-Product Attention with Reliability Gating
        self.cross_modal_fusion = CrossModalAttentionGating(embed_dim=feature_dim, num_heads=4, dropout=dropout)
        
        # Multi-Scale 1D Temporal Convolutions
        self.ms_tcn = MultiScaleTemporalConv(in_dim=feature_dim, out_dim=feature_dim)
        
        # Bidirectional GRU + Additive Temporal Attention
        self.temporal_encoder = BiGRUTemporalAttention(in_dim=feature_dim, hidden_dim=gru_hidden, num_layers=num_layers, dropout=dropout)
        
        # --- Multi-Task Prediction Heads ---
        # Main Head (Global Drowsiness classification)
        self.main_head = nn.Sequential(
            nn.Linear(gru_hidden * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Auxiliary Head 1: Frame-level Eye State
        self.aux_eye_head = nn.Linear(gru_hidden * 2, 1)
        
        # Auxiliary Head 2: Frame-level Mouth State
        self.aux_mouth_head = nn.Linear(gru_hidden * 2, 1)
        
        # Auxiliary Head 3: Frame-level Head State
        self.aux_head_head = nn.Linear(gru_hidden * 2, 1)

    def forward(self, eye_feats, mouth_feats, geo_vectors, condition_vec):
        """
        eye_feats: (B, T, 128)
        mouth_feats: (B, T, 128)
        geo_vectors: (B, T, 16)
        condition_vec: (B, 5)
        """
        B, T, _ = eye_feats.shape
        
        # Sequence-level temporal instance normalization for subject invariance
        geo_mean = geo_vectors.mean(dim=1, keepdim=True)
        geo_std = geo_vectors.std(dim=1, keepdim=True) + 1e-5
        geo_norm = (geo_vectors - geo_mean) / geo_std
        
        # 1. Encode geometric/kinematic vectors
        geo_flat = geo_norm.view(B * T, 16)
        h_feat = self.geo_encoder(geo_flat).view(B, T, self.feature_dim)
        
        # 2. FiLM Condition Modulation
        e_feat = self.film_eye(eye_feats, condition_vec)
        m_feat = self.film_mouth(mouth_feats, condition_vec)
        h_feat = self.film_geo(h_feat, condition_vec)
        
        # 3. Cross-Modal Attention with Reliability Gating
        fused_spatial, modality_gates = self.cross_modal_fusion(e_feat, m_feat, h_feat)
        
        # 4. Multi-Scale 1D Temporal Convolutions
        fused_temporal = self.ms_tcn(fused_spatial)
        
        # 5. Bidirectional Temporal Sequence Modeling with Additive Attention
        temp_context, frame_hidden_states, attn_weights = self.temporal_encoder(fused_temporal)
        
        # 5. Multi-Task Outputs
        drowsy_logits = self.main_head(temp_context).squeeze(-1)
        aux_eye_logits = self.aux_eye_head(frame_hidden_states).squeeze(-1)
        aux_mouth_logits = self.aux_mouth_head(frame_hidden_states).squeeze(-1)
        aux_head_logits = self.aux_head_head(frame_hidden_states).squeeze(-1)
        
        return {
            'drowsy_logits': drowsy_logits,
            'aux_eye_logits': aux_eye_logits,
            'aux_mouth_logits': aux_mouth_logits,
            'aux_head_logits': aux_head_logits,
            'temporal_attn_weights': attn_weights,
            'modality_gates': modality_gates
        }
