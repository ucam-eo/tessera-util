# src/models/ssl_model.py

import torch
import torch.nn as nn
from .modules import *

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_coeff=5e-3):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

    def forward(self, z1, z2):
        B = z1.size(0)
        eps = 1e-9
        z1_mean = z1.mean(dim=0)
        z2_mean = z2.mean(dim=0)
        z1_std = z1.std(dim=0).clamp_min(eps)
        z2_std = z2.std(dim=0).clamp_min(eps)
        z1_whiten = (z1 - z1_mean) / z1_std
        z2_whiten = (z2 - z2_mean) / z2_std
        c = torch.matmul(z1_whiten.T, z2_whiten) / B
        on_diag = torch.diagonal(c) - 1
        on_diag_loss = (on_diag ** 2).sum()
        off_diag = self.off_diagonal(c)
        off_diag_loss = (off_diag ** 2).sum()
        loss = on_diag_loss + self.lambda_coeff * off_diag_loss
        return loss, on_diag_loss, off_diag_loss

def compute_cross_correlation(z1, z2):
    B = z1.size(0)
    eps = 1e-9
    z1_mean = z1.mean(dim=0)
    z2_mean = z2.mean(dim=0)
    z1_std = z1.std(dim=0).clamp_min(eps)
    z2_std = z2.std(dim=0).clamp_min(eps)
    z1_w = (z1 - z1_mean) / z1_std
    z2_w = (z2 - z2_mean) / z2_std
    c = torch.matmul(z1_w.T, z2_w) / B
    return c

# class MultimodalBTModel(nn.Module):
#     def __init__(self, s2_backbone, s1_backbone, projector, fusion_method='sum', return_repr=False, latent_dim=None):
#         """
#         fusion_method: 'sum', 'concat' 或 'transformer'
#         若使用'transformer'则需要提供latent_dim
#         """
#         super().__init__()
#         self.s2_backbone = s2_backbone
#         self.s1_backbone = s1_backbone
#         self.projector = projector
#         self.fusion_method = fusion_method
#         self.return_repr = return_repr
#         if self.fusion_method == 'transformer':
#             if latent_dim is None:
#                 raise ValueError("latent_dim must be provided for transformer fusion")
#             self.fusion_transformer = FusionTransformer(input_dim=latent_dim, num_layers=1, nhead=4)
            
#         self.dim_reducer = nn.Sequential(
#             nn.Linear(1024, 128)
#             # nn.Linear(2048, 256)
#             # nn.Linear(512, 64)
#         )

#     def forward(self, s2_x, s1_x):
#         s2_repr = self.s2_backbone(s2_x)
#         s1_repr = self.s1_backbone(s1_x)
#         if self.fusion_method == 'concat':
#             fused = torch.cat([s2_repr, s1_repr], dim=-1)
#         elif self.fusion_method == 'sum':
#             fused = s2_repr + s1_repr
#         elif self.fusion_method == 'transformer':
#             tokens = torch.stack([s2_repr, s1_repr], dim=1)
#             fused = self.fusion_transformer(tokens)
#         else:
#             raise ValueError(f"Unknown fusion method: {self.fusion_method}")
#         # 降维到128
#         fused = self.dim_reducer(fused)
#         feats = self.projector(fused)
#         if self.return_repr:
#             return feats, fused
#         return feats

class MultimodalBTModel(nn.Module):
    def __init__(self, s2_backbone, s1_backbone, projector, fusion_method='concat', return_repr=False, latent_dim=128):
        """
        fusion_method: 'sum', 'concat' 或 'transformer'
        若使用'transformer'则需要提供latent_dim
        """
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.projector = projector
        self.fusion_method = fusion_method
        self.return_repr = return_repr
        
        if fusion_method == 'concat':
            in_dim = 8 * latent_dim  
        elif fusion_method == 'sum':
            in_dim = 4 * latent_dim
            
        self.dim_reducer = nn.Sequential(nn.Linear(in_dim, latent_dim))

        # self.dim_reducer = nn.Sequential(
        #     nn.Linear(in_dim, latent_dim),
        #     nn.LayerNorm(latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim),
        #     )
        
        # self.dim_reducer = nn.Sequential(
        #     nn.Linear(1024, 128)
        # )

    def forward(self, s2_x, s1_x):
        s2_repr = self.s2_backbone(s2_x)
        s1_repr = self.s1_backbone(s1_x)
        if self.fusion_method == 'concat':
            fused = torch.cat([s2_repr, s1_repr], dim=-1)
        elif self.fusion_method == 'sum':
            fused = s2_repr + s1_repr
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        # 降维到128
        fused = self.dim_reducer(fused)
        feats = self.projector(fused)
        if self.return_repr:
            return feats, fused
        return feats
    

class MultimodalBTModelDCube(nn.Module):
    def __init__(self, s2_backbone, s1_backbone, projector, fusion_method='sum', return_repr=False, latent_dim=None):
        """
        fusion_method: 'sum', 'concat' 或 'transformer'
        若使用'transformer'则需要提供latent_dim
        """
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.projector = projector
        self.fusion_method = fusion_method
        self.return_repr = return_repr
        if self.fusion_method == 'transformer':
            if latent_dim is None:
                raise ValueError("latent_dim must be provided for transformer fusion")
            self.fusion_transformer = FusionTransformer(input_dim=latent_dim, num_layers=1, nhead=4)
            
        self.dim_reducer = nn.Sequential(
            nn.Linear(256, 128),
            # nn.LayerNorm(128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
        )

    def forward(self, s2_x, s1_x):
        s2_repr = self.s2_backbone(s2_x)
        s1_repr = self.s1_backbone(s1_x)
        if self.fusion_method == 'concat':
            fused = torch.cat([s2_repr, s1_repr], dim=-1)
        elif self.fusion_method == 'sum':
            fused = s2_repr + s1_repr
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        # 降维到128
        fused = self.dim_reducer(fused)
        feats = self.projector(fused)
        if self.return_repr:
            return feats, fused
        return feats

class MultimodalBTInferenceModel(torch.nn.Module):
    """
    用于推理阶段的模型，只包含两个Transformer encoder (S2 + S1)，
    去掉了投影头。
    """
    def __init__(self, s2_backbone, s1_backbone, fusion_method, dim_reducer):
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.fusion_method = fusion_method
        self.dim_reducer = dim_reducer

    def forward(self, s2_x, s1_x):
        """
        s2_x.shape = (batch, seq_len_s2, band_num_s2)
        s1_x.shape = (batch, seq_len_s1, band_num_s1)
        输出: (batch, latent_dim) 或 (batch, 2*latent_dim) if fusion=concat
        """
        s2_repr = self.s2_backbone(s2_x)  # (batch, latent_dim)
        s1_repr = self.s1_backbone(s1_x)  # (batch, latent_dim)

        if self.fusion_method == "sum":
            fused = s2_repr + s1_repr
        elif self.fusion_method == "concat":
            fused = torch.cat([s2_repr, s1_repr], dim=-1)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        fused = self.dim_reducer(fused)
        return fused