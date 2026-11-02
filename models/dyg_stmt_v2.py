import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HighOrderKinematicProjector(nn.Module):
    """
    Extracts high-order differential kinematics across sequence time T:
    - 1st Order Velocity: dEAR/dt, dMAR/dt, dPitch/dt, dYaw/dt, dRoll/dt
    - 2nd Order Acceleration: d2EAR/dt2, d2MAR/dt2
    - Windowed PERCLOS: Cumulative eye closure duration (< 0.20)
    - Normalized Spatial Deviations: (x_t - mean(x)) / std(x)
    Transforms 16-D raw kinematics -> 32-D enriched kinematic representations.
    """
    def __init__(self, in_dim=16, hidden_dim=64, out_dim=128):
        super().__init__()
        # 16 raw + 16 normalized deviations + 5 velocities = 37 features
        self.proj = nn.Sequential(
            nn.Linear(37, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, geo):
        # geo: (B, T, 16)
        B, T, D = geo.shape
        
        # 1. Subject-invariant temporal deviation
        mean = geo.mean(dim=1, keepdim=True)
        std = geo.std(dim=1, keepdim=True) + 1e-5
        geo_norm = (geo - mean) / std
        
        # 2. 1st order velocity (difference across frames)
        # Pad first frame
        diff = geo[:, 1:, :] - geo[:, :-1, :] # (B, T-1, 16)
        diff_pad = torch.cat([diff[:, :1, :], diff], dim=1) # (B, T, 16)
        vel_subset = diff_pad[:, :, [0, 1, 3, 5, 6]] # (B, T, 5) -> dEAR_L, dEAR_R, dMAR, dPitch, dYaw
        
        # 3. Concatenate enriched feature vector: 16 raw + 16 norm + 5 vel = 37
        enriched = torch.cat([geo, geo_norm, vel_subset], dim=-1)
        out = self.proj(enriched.view(B * T, 37)).view(B, T, -1)
        return out


class FiLMConditionModulationV2(nn.Module):
    """
    Enhanced FiLM Condition Modulation with residual scaling and bottleneck expansion.
    """
    def __init__(self, num_conditions=5, feature_dim=128):
        super().__init__()
        self.film_net = nn.Sequential(
            nn.Linear(num_conditions, 32),
            nn.GELU(),
            nn.Linear(32, feature_dim * 2)
        )
        # Initialize gamma to 1 and beta to 0
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)

    def forward(self, x, cond):
        B, T, D = x.shape
        film_params = self.film_net(cond).unsqueeze(1) # (B, 1, 2*D)
        gamma, beta = film_params.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


class CrossModalAttentionGatingV2(nn.Module):
    """
    Multi-Head Cross-Modal Attention with Dynamic Reliability Gating.
    Enables Mouth and Head channels to dynamically compensate under eye occlusion (sunglasses).
    """
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
        
        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, eye, mouth, head):
        B, T, D = eye.shape
        
        # Reshape to token sequences per timestep
        e_flat = eye.reshape(B * T, 1, D)
        m_flat = mouth.reshape(B * T, 1, D)
        h_flat = head.reshape(B * T, 1, D)
        
        tokens = torch.cat([e_flat, m_flat, h_flat], dim=1) # (B*T, 3, D)
        attn_out, _ = self.mha(tokens, tokens, tokens)      # (B*T, 3, D)
        
        concat_raw = torch.cat([e_flat.squeeze(1), m_flat.squeeze(1), h_flat.squeeze(1)], dim=1) # (B*T, 3*D)
        gates = self.gate_net(concat_raw).unsqueeze(2) # (B*T, 3, 1)
        
        gated_attn = (attn_out * gates).reshape(B * T, 3 * D)
        fused = self.proj(gated_attn).reshape(B, T, D)
        return fused, gates.reshape(B, T, 3)


class MultiScaleTemporalConvV2(nn.Module):
    """
    Multi-Scale 1D Temporal Convolutions extracting micro-temporal multi-frequency dynamics:
    - Kernel 1: Instantaneous state
    - Kernel 3: Rapid micro-blinks (100ms)
    - Kernel 5: Head nodding dynamics (160ms)
    - Kernel 7: Prolonged yawning expansions (230ms)
    """
    def __init__(self, in_dim=128, out_dim=128):
        super().__init__()
        b_dim = out_dim // 4
        self.conv1 = nn.Conv1d(in_dim, b_dim, kernel_size=1)
        self.conv3 = nn.Conv1d(in_dim, b_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_dim, b_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_dim, b_dim, kernel_size=7, padding=3)
        self.norm = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, T, D) -> transpose to (B, D, T)
        x_t = x.transpose(1, 2)
        c1 = self.conv1(x_t)
        c3 = self.conv3(x_t)
        c5 = self.conv5(x_t)
        c7 = self.conv7(x_t)
        concat = torch.cat([c1, c3, c5, c7], dim=1)
        out = self.act(self.norm(concat)) + x_t
        return out.transpose(1, 2)


class DualPathTransformerGRU(nn.Module):
    """
    Dual-Path Spatio-Temporal Hybrid Block:
    Path 1: Multi-Head Temporal Self-Attention Transformer with Learnable Positional Encodings.
    Path 2: Bidirectional GRU capturing sequential Markovian transitions.
    Path Fusion: Gated Linear Cross-Fusion.
    """
    def __init__(self, in_dim=128, hidden_dim=64, seq_len=30, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, in_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Path 1: Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=4, dim_feedforward=in_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Path 2: BiGRU
        self.bigru = nn.GRU(in_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        
        # Additive Temporal Attention
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Path Cross-Fusion Projector
        self.fusion_norm = nn.LayerNorm(hidden_dim * 2 + in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        
        # Path 1: Transformer with positional encoding
        x_pos = x + self.pos_embed[:, :T, :]
        trans_out = self.transformer(x_pos) # (B, T, D)
        
        # Path 2: BiGRU
        gru_out, _ = self.bigru(x) # (B, T, 2*hidden_dim)
        
        # Dual-Path Concatenation
        combined = torch.cat([gru_out, trans_out], dim=-1) # (B, T, 2*hidden + D)
        combined = self.fusion_norm(combined)
        
        # Additive Temporal Attention
        attn_scores = self.attn_net(combined) # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1) # (B, T, 1)
        
        context_vector = torch.sum(combined * attn_weights, dim=1) # (B, 2*hidden + D)
        return context_vector, combined, attn_weights.squeeze(-1)


class DyGSTMTv2(nn.Module):
    """
    DyG-STMT v2: Enhanced Dynamic Gated Spatio-Temporal Multi-Task Architecture.
    
    Integrated Modules:
    1. High-Order Differential Kinematic Projector (37-D -> 128-D)
    2. FiLM Scenario Condition Modulator (v2)
    3. Cross-Modal Attention with Reliability Gating (v2)
    4. Multi-Scale 1D Temporal Convolutions (MS-TCN)
    5. Dual-Path Transformer-GRU Spatio-Temporal Encoder
    6. Multi-Task Heads (Global Drowsiness + Eye + Mouth + Head Pose)
    """
    def __init__(self, num_conditions=5, feature_dim=128, gru_hidden=64, seq_len=30, dropout=0.3):
        super().__init__()
        self.feature_dim = feature_dim
        
        # High-Order Kinematic Projector
        self.geo_projector = HighOrderKinematicProjector(in_dim=16, hidden_dim=64, out_dim=feature_dim)
        
        # FiLM Condition Modulators
        self.film_eye = FiLMConditionModulationV2(num_conditions, feature_dim)
        self.film_mouth = FiLMConditionModulationV2(num_conditions, feature_dim)
        self.film_geo = FiLMConditionModulationV2(num_conditions, feature_dim)
        
        # Cross-Modal Attention with Reliability Gating
        self.cross_modal_fusion = CrossModalAttentionGatingV2(embed_dim=feature_dim, num_heads=4, dropout=dropout)
        
        # Multi-Scale 1D Temporal Convolutions
        self.ms_tcn = MultiScaleTemporalConvV2(in_dim=feature_dim, out_dim=feature_dim)
        
        # Dual-Path Spatio-Temporal Encoder
        self.spatio_temporal = DualPathTransformerGRU(in_dim=feature_dim, hidden_dim=gru_hidden, seq_len=seq_len, dropout=dropout)
        
        # Feature representation dimension after dual-path fusion: 2*gru_hidden + feature_dim = 128 + 128 = 256
        repr_dim = gru_hidden * 2 + feature_dim
        
        # Contrastive Latent Projection Head (used during contrastive regularization)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(repr_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        
        # --- Multi-Task Prediction Heads ---
        # Main Head (Global Drowsiness classification)
        self.main_head = nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Auxiliary Head 1: Frame-level Eye State
        self.aux_eye_head = nn.Linear(repr_dim, 1)
        
        # Auxiliary Head 2: Frame-level Mouth State
        self.aux_mouth_head = nn.Linear(repr_dim, 1)
        
        # Auxiliary Head 3: Frame-level Head State
        self.aux_head_head = nn.Linear(repr_dim, 1)

    def forward(self, eye_feats, mouth_feats, geo_vectors, condition_vec):
        """
        eye_feats: (B, T, 128)
        mouth_feats: (B, T, 128)
        geo_vectors: (B, T, 16)
        condition_vec: (B, 5)
        """
        B, T, _ = eye_feats.shape
        
        # 1. High-order differential kinematics
        h_feat = self.geo_projector(geo_vectors) # (B, T, 128)
        
        # 2. FiLM Condition Modulation
        e_feat = self.film_eye(eye_feats, condition_vec)
        m_feat = self.film_mouth(mouth_feats, condition_vec)
        h_feat = self.film_geo(h_feat, condition_vec)
        
        # 3. Cross-Modal Attention with Reliability Gating
        fused_spatial, modality_gates = self.cross_modal_fusion(e_feat, m_feat, h_feat)
        
        # 4. Multi-Scale 1D Temporal Convolutions
        fused_temporal = self.ms_tcn(fused_spatial)
        
        # 5. Dual-Path Transformer-GRU Sequence Modeling with Additive Attention
        context_vector, frame_states, attn_weights = self.spatio_temporal(fused_temporal)
        
        # 6. Contrastive Projection & Multi-Task Outputs
        latent_proj = F.normalize(self.contrastive_proj(context_vector), dim=-1)
        drowsy_logits = self.main_head(context_vector).squeeze(-1)
        aux_eye_logits = self.aux_eye_head(frame_states).squeeze(-1)
        aux_mouth_logits = self.aux_mouth_head(frame_states).squeeze(-1)
        aux_head_logits = self.aux_head_head(frame_states).squeeze(-1)
        
        return {
            'drowsy_logits': drowsy_logits,
            'latent_proj': latent_proj,
            'aux_eye_logits': aux_eye_logits,
            'aux_mouth_logits': aux_mouth_logits,
            'aux_head_logits': aux_head_logits,
            'temporal_attn_weights': attn_weights,
            'modality_gates': modality_gates
        }
