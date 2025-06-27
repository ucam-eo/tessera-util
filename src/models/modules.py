# src/models/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, 1)
    def forward(self, x):
        # x: (B, seq_len, dim)
        w = torch.softmax(self.query(x), dim=1)  # (B, seq_len, 1)
        return (w * x).sum(dim=1)
    

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

class ImprovedAttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.head_dim = input_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Projections for query, key, and value
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        
        # Output projection
        self.out_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # x: (B, seq_len, dim)
        batch_size, seq_len, _ = x.shape
        
        # Project to query, key, value
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Reshape for multi-head attention
        q = q.permute(0, 2, 1, 3)  # (B, num_heads, seq_len, head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, num_heads, seq_len, head_dim)
        v = v.permute(0, 2, 1, 3)  # (B, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, num_heads, seq_len, seq_len)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, num_heads, seq_len, seq_len)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)  # (B, num_heads, seq_len, head_dim)
        
        # Reshape back
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.input_dim)  # (B, seq_len, dim)
        
        # Apply output projection
        context = self.out_proj(context)  # (B, seq_len, dim)
        
        # Compute per-timestep importance for pooling
        # Use mean attention weight across all heads
        importance = attention_weights.mean(1).mean(-1, keepdim=True)  # (B, seq_len, 1)
        importance = importance.transpose(1, 2)  # (B, 1, seq_len)
        
        # Normalize importance weights
        importance = torch.softmax(importance, dim=-1)  # (B, 1, seq_len)
        
        # Apply weighted sum for pooling
        pooled = torch.bmm(importance, context).squeeze(1)  # (B, dim)
        
        return pooled

# class TransformerEncoder(nn.Module):
#     def __init__(self, band_num, latent_dim, nhead=8, num_encoder_layers=4,
#                  dim_feedforward=512, dropout=0.1, max_seq_len=20):
#         super().__init__()
#         # 将 embedding 维度提升到 latent_dim*4
#         self.embedding = nn.Sequential(
#             nn.Linear(band_num, latent_dim * 4),
#             nn.ReLU(),
#             nn.Linear(latent_dim * 4, latent_dim * 4)
#         )
#         self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, latent_dim * 4))
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=latent_dim * 4,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation="relu",
#             batch_first=False
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
#         self.attn_pool = AttentionPooling(latent_dim * 4)
#         # 降维到 latent_dim
#         self.fc_out = nn.Linear(latent_dim * 4, latent_dim)
    
#     def forward(self, x):
#         # x: (B, seq_len, band_num)
#         b, s, _ = x.shape
#         x = self.embedding(x)  # (B, seq_len, latent_dim*4)
#         x = x + self.pos_encoder[:, :s, :]
#         x = x.permute(1, 0, 2)  # (seq_len, B, latent_dim*4)
#         x = self.transformer_encoder(x)
#         x = x.permute(1, 0, 2)  # (B, seq_len, latent_dim*4)
#         x = self.attn_pool(x)
#         # x = self.fc_out(x)
#         return x

class TemporalEncoding(nn.Module):
    def __init__(self, d_model, num_freqs=64):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_model = d_model
        
        # 可学习的频率参数（比固定频率更灵活）
        self.freqs = nn.Parameter(torch.exp(torch.linspace(0, np.log(365.0), num_freqs)))
        
        # 通过线性层将傅里叶特征投影到目标维度
        self.proj = nn.Linear(2 * num_freqs, d_model)
        self.phase = nn.Parameter(torch.zeros(1, 1, d_model))  # 可学习相位偏移

    def forward(self, doy):
        # doy: (B, seq_len, 1)
        t = doy / 365.0 * 2 * np.pi  # 归一化到0-2π范围
        
        # 生成多频率正弦/余弦特征
        t_scaled = t * self.freqs.view(1, 1, -1)  # (B, seq_len, num_freqs)
        sin = torch.sin(t_scaled + self.phase[..., :self.num_freqs])
        cos = torch.cos(t_scaled + self.phase[..., self.num_freqs:2*self.num_freqs])
        
        # 拼接并投影到目标维度
        encoding = torch.cat([sin, cos], dim=-1)  # (B, seq_len, 2*num_freqs)
        return self.proj(encoding)  # (B, seq_len, d_model)
    
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
        
#         # 数据特征嵌入模块
#         self.embedding = nn.Sequential(
#             nn.Linear(band_num, latent_dim*4),  # 注意输入维度变化
#             nn.LayerNorm(latent_dim*4),
#             nn.ReLU(),
#             nn.Linear(latent_dim*4, latent_dim*4)
#         )
        
#         # 时间编码模块
#         # self.temporal_encoding = TemporalEncoding(latent_dim*4)
#         self.temporal_encoding = TemporalPositionalEncoder(latent_dim*4)
#         # 位置编码，可学习
#         # self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, latent_dim*4))
        
#         # Transformer编码器
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=latent_dim*4,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation="relu",
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
#         # 输出层
#         # self.attn_pool = AttentionPooling(latent_dim*4)
#         self.attn_pool = TemporalAwarePooling(latent_dim*4)
#         # self.attn_pool = ImprovedAttentionPooling(latent_dim*4, num_heads=8)
#         self.fc_out = nn.Sequential(
#             nn.LayerNorm(latent_dim*4),
#             nn.Linear(latent_dim*4, latent_dim)
#         )

#     def forward(self, x):
#         # x形状: (B, seq_len, band_num)
#         B, seq_len, _ = x.shape
        
#         # 分离数据和doy特征
#         x_data = x[..., :-1]  # (B, seq_len, band_num-1)
#         doy = x[..., -1]     # (B, seq_len, 1)
        
#         # 特征嵌入
#         x_emb = self.embedding(x_data)  # (B, seq_len, latent_dim*4)
        
#         # 时间编码
#         t_emb = self.temporal_encoding(doy)  # (B, seq_len, latent_dim*4)
        
#         x = x_emb + t_emb
        
#         x = self.transformer_encoder(x)
        
#         # 输出处理
#         x = self.attn_pool(x)
#         # x = self.fc_out(x)
#         return x

# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, output_dim)
#         )
#     def forward(self, x):
#         return self.net(x)

class TransformerEncoder(nn.Module):
    def __init__(self, band_num, latent_dim, nhead=8, num_encoder_layers=4,
                dim_feedforward=512, dropout=0.1, max_seq_len=20):
        super().__init__()
        # Total input dimension: bands
        input_dim = band_num
        
        # Embedding to increase dimension
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, latent_dim*4),
            nn.ReLU(),
            nn.Linear(latent_dim*4, latent_dim*4)
        )
        
        # Temporal Encoder for DOY as position encoding
        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim*4)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim*4,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
       
        # Temporal Aware Pooling
        self.attn_pool = TemporalAwarePooling(latent_dim*4)
   
    def forward(self, x):
        # x: (B, seq_len, 10 bands + 1 doy)
        # Split bands and doy
        bands = x[:, :, :-1]  # All columns except last one
        doy = x[:, :, -1]     # Last column is DOY
        # Embedding of bands
        bands_embedded = self.embedding(bands)  # (B, seq_len, latent_dim*4)
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
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim//4, hidden_dim//8),
            nn.BatchNorm1d(hidden_dim//8),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim//8, hidden_dim//4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim//2, hidden_dim),
            
            
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(inplace=False),
            # nn.Linear(hidden_dim, output_dim),
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
    

class SpatioTemporalCNNEncoder(nn.Module):
    def __init__(self, input_channels, representation_dim=128):
        """
        Spatio-Temporal CNN Encoder for remote sensing data
        
        Args:
            input_channels (int): Number of spectral bands (excluding DOY)
            representation_dim (int): Output representation dimension, default 128
        """
        super(SpatioTemporalCNNEncoder, self).__init__()
        
        # 3D CNN for processing (Batch, C, T, H, W) input
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # Spatial downsampling
            
            # Second conv block
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # Spatial downsampling
            
            # Third conv block
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling for feature aggregation
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Final projection to representation space
        self.fc = nn.Linear(128, representation_dim)
        
    def forward(self, x):
        """
        Forward pass through the encoder
        
        Args:
            x (torch.Tensor): Input tensor with shape (Batch, H, W, T, C)
                              where the last channel of C is DOY
        
        Returns:
            torch.Tensor: Representation with shape (Batch, representation_dim)
        """
        batch_size = x.shape[0]
        
        # 1. Remove DOY channel (last channel)
        x = x[..., :-1]  # Now shape: (Batch, H, W, T, C-1)
        
        # 2. Rearrange dimensions to (Batch, C-1, T, H, W) for 3D CNN
        x = x.permute(0, 4, 3, 1, 2)
        
        # 3. Apply CNN encoder
        x = self.encoder(x)  # Output shape: (Batch, 128, T', H', W')
        
        # 4. Global pooling
        x = self.global_pool(x)  # Shape: (Batch, 128, 1, 1, 1)
        x = x.view(batch_size, -1)  # Shape: (Batch, 128)
        
        # 5. Project to representation dimension
        x = self.fc(x)  # Shape: (Batch, representation_dim)
        
        return x