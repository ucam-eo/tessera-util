# src/models/modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

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
        
    # def forward(self, x):
    #     # 先通过RNN捕获时序上下文
    #     x_context, _ = self.temporal_context(x)
    #     # 再计算注意力权重
    #     w = torch.softmax(self.query(x_context), dim=1)
    #     return (w * x).sum(dim=1)
    def forward(self, x):
        # Add safety check for sequence length
        B, T, D = x.shape
        if T == 0:
            # Handle empty sequence case - return zeros
            return torch.zeros(B, D, device=x.device)
        elif T == 1:
            # Handle single element sequence - skip GRU
            return x.squeeze(1)
        else:
            # Normal case with multiple timesteps
            try:
                # Process through RNN to capture temporal context
                x_context, _ = self.temporal_context(x)
                # Calculate attention weights
                w = torch.softmax(self.query(x_context), dim=1)
                return (w * x).sum(dim=1)
            except Exception as e:
                # Fallback to mean pooling if RNN fails
                print(f"RNN failed with error: {e}. Falling back to mean pooling, current x shape is {x.shape}, values are {x}")
                return torch.mean(x, dim=1)

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
        # self.net = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=False),
        #     nn.Linear(hidden_dim, output_dim),
        # )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            
            # nn.Linear(hidden_dim, hidden_dim//2),
            # nn.BatchNorm1d(hidden_dim//2),
            # nn.ReLU(inplace=False),
            # nn.Linear(hidden_dim//2, hidden_dim//4),
            # nn.BatchNorm1d(hidden_dim//4),
            # nn.ReLU(inplace=False),
            # nn.Linear(hidden_dim//4, hidden_dim//8),
            # nn.BatchNorm1d(hidden_dim//8),
            # nn.ReLU(inplace=False),
            # nn.Linear(hidden_dim//8, hidden_dim//4),
            # nn.BatchNorm1d(hidden_dim//4),
            # nn.ReLU(inplace=False),
            # nn.Linear(hidden_dim//4, hidden_dim//2),
            # nn.BatchNorm1d(hidden_dim//2),
            # nn.ReLU(inplace=False),
            # nn.Linear(hidden_dim//2, hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
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
    


class SpatioTemporalCNNEncoder(nn.Module):
    def __init__(self, input_channels, representation_dim=128):
        """
        改进的时空特征提取器，针对低分辨率(60m)的遥感时序数据优化
        
        Args:
            input_channels (int): 波段数量（不包括DOY）
            representation_dim (int): 输出表示维度，默认为128
        """
        super(SpatioTemporalCNNEncoder, self).__init__()
        
        # 空间特征提取
        self.spatial_conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 残差块，保留更多空间信息
        self.res_block1 = ResBlock(32, 32)
        self.res_block2 = ResBlock(32, 64)
        
        # 时间特征提取
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(64*9, 128, kernel_size=3, padding=1),  # 空间特征后有降采样
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        
        # 自注意力机制，加强对关键时间点的关注
        self.temporal_attention = TemporalAttention(128)
        
        # 全局池化后的特征投影
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, representation_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入tensor, shape = (Batch, H, W, T, C)
                             其中最后一个通道为DOY
        
        Returns:
            torch.Tensor: 输出表示, shape = (Batch, representation_dim)
        """
        batch_size = x.shape[0]
        time_steps = x.shape[3]
        
        # 1. 移除DOY通道
        x = x[..., :-1]  # shape: (Batch, H, W, T, C-1)
        
        # 2. 时空特征提取
        spatial_features = []
        
        # 针对每个时间点提取空间特征
        for t in range(time_steps):
            # 获取当前时间点的数据
            curr_x = x[:, :, :, t, :]  # (Batch, H, W, C-1)
            curr_x = curr_x.permute(0, 3, 1, 2)  # 调整为(Batch, C-1, H, W)
            
            # 空间特征提取
            feat = self.spatial_conv1(curr_x)
            feat = self.res_block1(feat)
            feat = self.res_block2(feat)  # (Batch, 64, H/2, W/2)
            
            # 空间池化得到每个时间点的特征
            feat = F.adaptive_avg_pool2d(feat, (3, 3))  # 获得3x3的空间网格
            spatial_features.append(feat)
        
        # 3. 将所有时间点的空间特征连接并处理时间维度
        spatial_features = torch.stack(spatial_features, dim=2)  # (Batch, 64, T, 3, 3)
        spatial_features = spatial_features.flatten(3)  # (Batch, 64, T, 9)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # (Batch, 9, 64, T)
        spatial_features = spatial_features.reshape(batch_size, 9*64, time_steps)  # (Batch, 9*64, T)
        
        # 4. 时间特征提取
        temporal_features = self.temporal_conv(spatial_features)  # (Batch, 128, T)
        
        # 5. 应用时间注意力
        temporal_features = self.temporal_attention(temporal_features)  # (Batch, 128, T)
        
        # 6. 全局池化和投影
        pooled_features = F.adaptive_max_pool1d(temporal_features, 1).squeeze(-1)  # (Batch, 128)
        representation = self.fc(pooled_features)  # (Batch, representation_dim)
        
        return representation


class ResBlock(nn.Module):
    """残差块，保留更多空间信息特征"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += self.shortcut(residual)
        out = self.relu(out)
        if out.shape[2] > 12:  # 适当的空间下采样
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        return out


class TemporalAttention(nn.Module):
    """时间注意力机制，关注重要的时间点"""
    def __init__(self, channels):
        super(TemporalAttention, self).__init__()
        self.query = nn.Conv1d(channels, channels//8, kernel_size=1)
        self.key = nn.Conv1d(channels, channels//8, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的权重参数
    
    def forward(self, x):
        # x: (Batch, Channel, Time)
        batch_size, C, T = x.size()
        
        # 计算注意力
        proj_query = self.query(x).permute(0, 2, 1)  # B x T x C/8
        proj_key = self.key(x)  # B x C/8 x T
        energy = torch.bmm(proj_query, proj_key)  # B x T x T
        attention = F.softmax(energy, dim=-1)  # B x T x T
        
        # 应用注意力
        proj_value = self.value(x)  # B x C x T
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x T
        
        # 残差连接
        out = self.gamma * out + x
        
        return out
    
    

class TemporalAwarePoolingWithMask(nn.Module):
    """
    Temporal-aware pooling with attention mechanism that respects the mask.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, 1)
        self.temporal_context = nn.GRU(input_dim, input_dim, batch_first=True)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Boolean mask of shape (batch_size, seq_len), where True indicates valid timesteps
        """
        # Apply temporal context via RNN
        x_context, _ = self.temporal_context(x)
        
        # Calculate attention weights
        attn_scores = self.query(x_context)  # (B, seq_len, 1)
        
        # Apply mask by setting scores of invalid timesteps to a smaller negative value
        # Using -1e4 instead of -1e9 to avoid overflow with half-precision (when using AMP)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, seq_len, 1)
            attn_scores = attn_scores.masked_fill(~mask_expanded, float('-inf'))
        
        # Apply softmax to get attention weights
        w = torch.softmax(attn_scores, dim=1)
        
        # Apply attention to input and sum
        return (w * x).sum(dim=1)  # (B, input_dim)


class TransformerEncoder_64_Fixed(nn.Module):
    """
    Enhanced transformer encoder that handles variable-length sequences with attention masks.
    Designed for 64 timesteps data structure.
    """
    def __init__(self, band_num, latent_dim, nhead=8, num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.1, max_seq_len=64):
        super().__init__()
        
        # Data feature embedding module
        self.embedding = nn.Sequential(
            nn.Linear(band_num, latent_dim*4),
            nn.LayerNorm(latent_dim*4),
            nn.ReLU(),
            nn.Linear(latent_dim*4, latent_dim*4)
        )
        
        # Temporal encoding module
        self.temporal_encoding = TemporalPositionalEncoder(latent_dim*4)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim*4,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Output layers
        self.attn_pool = TemporalAwarePoolingWithMask(latent_dim*4)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, band_num)
            attention_mask: Boolean mask of shape (batch_size, seq_len), where True indicates valid timesteps
        """
        # x shape: (B, seq_len, band_num)
        B, seq_len, _ = x.shape
        
        # Split data and DOY features
        x_data = x[..., :-1]  # (B, seq_len, band_num-1)
        doy = x[..., -1]     # (B, seq_len)
        
        # Feature embedding
        x_emb = self.embedding(x_data)  # (B, seq_len, latent_dim*4)
        
        # Temporal encoding
        t_emb = self.temporal_encoding(doy)  # (B, seq_len, latent_dim*4)
        
        # Combine embeddings
        x_t_emb = x_emb + t_emb
        
        # Create attention mask for transformer if mask is provided
        # In transformer, we need to convert to key_padding_mask which is True for positions to ignore
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # Invert since transformer wants 'True' for positions to mask
            x = self.transformer_encoder(x_t_emb, src_key_padding_mask=key_padding_mask)
        else:
            x = self.transformer_encoder(x_t_emb)
        
        # Output processing with attention-aware pooling
        x = self.attn_pool(x, attention_mask)  # (B, latent_dim*4)
        
        return x    


