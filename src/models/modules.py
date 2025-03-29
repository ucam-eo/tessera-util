# src/models/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# class AttentionPooling(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.query = nn.Linear(input_dim, 1)
#     def forward(self, x):
#         # x: (B, seq_len, dim)
#         w = torch.softmax(self.query(x), dim=1)  # (B, seq_len, 1)
#         return (w * x).sum(dim=1)
class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, 1)
    def forward(self, x):
        # More explicit operations
        w = torch.softmax(self.query(x), dim=1)
        weighted = w * x
        return weighted.sum(dim=1)  # More explicit sum
    
class TemporalAwarePooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, 1)
        self.temporal_context = nn.GRU(input_dim, input_dim, batch_first=True)
        
    def forward(self, x):
        # 先通过RNN捕获时序上下文
        x_context, _ = self.temporal_context(x)
        # 再计算注意力权重
        w = torch.softmax(self.query(x_context), dim=1)
        return (w * x).sum(dim=1)

class TemporalPositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
       
    def forward(self, doy):
        # doy: [B, T] tensor containing DOY values (0-365)
        position = doy.unsqueeze(-1).float()  # Ensure float type
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float) * -(math.log(10000.0) / self.d_model))
        div_term = div_term.to(doy.device)
       
        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

# class TransformerEncoder(nn.Module):
#     def __init__(self, band_num, latent_dim, nhead=8, num_encoder_layers=4,
#                  dim_feedforward=512, dropout=0.1, max_seq_len=20):
#         super().__init__()
#         self.embedding = nn.Sequential(
#             nn.Linear(band_num, latent_dim),
#             nn.ReLU(),
#             nn.Linear(latent_dim, latent_dim)
#         )
#         self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, latent_dim))
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=latent_dim,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation="relu",
#             batch_first=False
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#         self.attn_pool = AttentionPooling(latent_dim)
#         self.fc_out = nn.Linear(latent_dim, latent_dim)
#     def forward(self, x):
#         # x: (B, seq_len, band_num)
#         b, s, _ = x.shape
#         x = self.embedding(x)
#         x = x + self.pos_encoder[:, :s, :]
#         x = x.permute(1, 0, 2)  # (seq_len, B, latent_dim)
#         x = self.transformer_encoder(x)
#         x = x.permute(1, 0, 2)  # (B, seq_len, latent_dim)
#         x = self.attn_pool(x)
#         x = self.fc_out(x)
#         return x

# class TransformerEncoder(nn.Module):
#     def __init__(self, band_num, latent_dim, nhead=8, num_encoder_layers=4,
#                  dim_feedforward=512, dropout=0.1, max_seq_len=20):
#         super().__init__()
#         # 将 embedding 维度提升到 latent_dim*4
#         self.embedding = nn.Sequential(
#             nn.Linear(band_num, latent_dim * 8),
#             nn.ReLU(),
#             nn.Linear(latent_dim * 8, latent_dim * 8)
#         )
#         self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, latent_dim * 8))
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=latent_dim * 8,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation="relu",
#             batch_first=True,
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
#         # self.attn_pool = AttentionPooling(latent_dim * 8)
#         self.attn_pool = TemporalAwarePooling(latent_dim * 8)
    
#     def forward(self, x):
#         # x: (B, seq_len, band_num)
#         b, s, _ = x.shape
#         x = self.embedding(x)  # (B, seq_len, latent_dim*4)
#         x = x + self.pos_encoder[:, :s, :]
#         # x = x.permute(1, 0, 2)  # (seq_len, B, latent_dim*4)
#         x = self.transformer_encoder(x)
#         # x = x.permute(1, 0, 2)  # (B, seq_len, latent_dim*4)
#         x = self.attn_pool(x)
#         # x = self.fc_out(x)
#         return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, band_num, latent_dim, nhead=8, num_encoder_layers=4,
                dim_feedforward=512, dropout=0.1, max_seq_len=20):
        super().__init__()
        # Total input dimension: bands
        input_dim = band_num
        
        # Embedding to increase dimension
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 8),
            nn.ReLU(),
            nn.Linear(latent_dim * 8, latent_dim * 8)
        )
        
        # Temporal Encoder for DOY as position encoding
        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 8)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim * 8,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
       
        # Temporal Aware Pooling
        self.attn_pool = TemporalAwarePooling(latent_dim * 8)
   
    def forward(self, x):
        # x: (B, seq_len, 10 bands + 1 doy)
        # Split bands and doy
        bands = x[:, :, :-1]  # All columns except last one
        doy = x[:, :, -1]     # Last column is DOY
        # Embedding of bands
        bands_embedded = self.embedding(bands)  # (B, seq_len, latent_dim*8)
        temporal_encoding = self.temporal_encoder(doy)
        # Add temporal encoding to embedded bands (instead of random positional encoding)
        x = bands_embedded + temporal_encoding
        x = self.transformer_encoder(x)
        x = self.attn_pool(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)

class FusionTransformer(nn.Module):
    """
    使用Transformer融合多个模态表示：
    将各模态表示（如形状(B, 2, latent_dim)）与一个可学习的[CLS] token拼接，
    经过TransformerEncoder后取CLS token作为融合结果。
    """
    def __init__(self, input_dim, num_layers=1, nhead=4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, tokens):
        # tokens: (B, num_tokens, input_dim)
        B = tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        out = self.transformer_encoder(tokens)
        fused = out[:, 0, :]
        return fused

class SingleModalityTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim=128,
                 max_seq_len=20,
                 nhead=8,
                 num_layers=2,
                 dim_feedforward=512,
                 dropout=0.1):
        super().__init__()
        # 1) 线性embedding
        self.embedding = nn.Linear(input_dim, latent_dim)
        # 2) 可学习位置编码
        self.pos_encoder = nn.Parameter(
            torch.randn(1, max_seq_len, latent_dim)
        )
        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        self.attn_pool = AttentionPooling(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        """
        x: (B, seq_len, input_dim)
        """
        seq_len = x.shape[1]
        x = self.embedding(x)                         # (B, seq_len, d_model)
        x = x + self.pos_encoder[:, :seq_len, :]      # (B, seq_len, d_model)
        x = self.transformer_encoder(x)               # (B, seq_len, d_model)
        x = self.attn_pool(x)                         # (B, d_model)
        x = self.out_proj(x)
        return x

class SpectralTemporalTransformer(nn.Module):
    def __init__(self, 
                 data_dim, 
                 time_dim=2, 
                 latent_dim=128,
                 nhead=8,
                 num_layers=16,
                 fusion_method='concat',
                 **kwargs):
        super().__init__()
        self.fusion_method = fusion_method
        # 分开两个transformer
        self.band_transformer = SingleModalityTransformer(
            input_dim=data_dim-2,
            latent_dim=latent_dim,
            dim_feedforward=512,
            nhead=nhead,
            num_layers=num_layers,
            **kwargs
        )
        self.time_transformer = SingleModalityTransformer(
            input_dim=time_dim,
            latent_dim=latent_dim,
            dim_feedforward=256,
            nhead=4,
            num_layers=4,
            **kwargs
        )
        # 如果要 concat
        if fusion_method == 'concat':
            self.fuse_linear = nn.Linear(2*latent_dim, latent_dim)

    def forward(self, x):
        # x: (B, seq_len, data_dim + 2)
        # 1) 分别过自己的transformer
        band_x = x[..., :-2]  # (B, seq_len, data_dim)
        time_x = x[..., -2:]  # (B, seq_len, 2)
        band_feat = self.band_transformer(band_x)  # (B, latent_dim)
        time_feat = self.time_transformer(time_x)  # (B, latent_dim)

        # 2) 结果融合
        if self.fusion_method == 'sum':
            fused = band_feat + time_feat
        elif self.fusion_method == 'concat':
            fused = torch.cat([band_feat, time_feat], dim=-1)  # (B, 2*latent_dim)
            fused = self.fuse_linear(fused)                    # (B, latent_dim)
        else:
            raise ValueError("fusion_method must be 'sum' or 'concat'.")

        return fused