import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 常量定义
NUM_SAMPLES = 1280
BATCH_SIZE = 64
NUM_DAYS = 365
NUM_BANDS = 10
EMBED_DIM = 512
OUTPUT_DIM = 128

# 生成随机数据
def generate_data(num_samples):
    """
    生成随机像素数据和掩码
    
    参数:
        num_samples: 要生成的样本数量
        
    返回:
        data: 形状为(num_samples, NUM_DAYS, NUM_BANDS)的张量
        mask: 形状为(num_samples, NUM_DAYS)的张量，值为0, 1, 2
    """
    # 初始化数据和掩码张量
    data = torch.zeros(num_samples, NUM_DAYS, NUM_BANDS)
    mask = torch.randint(0, 3, (num_samples, NUM_DAYS))  # 0, 1, 2大致均匀分布
    
    # 在mask不为0的位置填充随机值
    for i in range(num_samples):
        for j in range(NUM_DAYS):
            if mask[i, j] != 0:
                data[i, j] = torch.rand(NUM_BANDS)
    
    return data, mask

# 创建数据增强
def create_augmentations(data, mask):
    """
    通过修改掩码创建两个增强
    
    参数:
        data: 形状为(batch_size, NUM_DAYS, NUM_BANDS)的张量
        mask: 形状为(batch_size, NUM_DAYS)的张量，值为0, 1, 2
        
    返回:
        data: 不变的数据张量
        mask1, mask2: 两个增强掩码，其中70%的云遮挡(值=1)时间步被随机更改为0
    """
    batch_size = data.shape[0]
    
    # 创建两个新掩码
    mask1 = mask.clone()
    mask2 = mask.clone()
    
    # 对批次中的每个样本
    for i in range(batch_size):
        # 找出mask为1的索引(云遮挡数据)
        cloudy_indices = torch.where(mask[i] == 1)[0]
        
        if len(cloudy_indices) > 0:  # 只有当有云遮挡索引时才继续
            # 随机选择70%的索引更改为0
            num_to_change = max(1, int(0.7 * len(cloudy_indices)))  # 确保至少更改1个
            
            # 对mask1
            indices_to_change = cloudy_indices[torch.randperm(len(cloudy_indices))[:num_to_change]]
            mask1[i, indices_to_change] = 0
            
            # 对mask2(不同的随机选择)
            indices_to_change = cloudy_indices[torch.randperm(len(cloudy_indices))[:num_to_change]]
            mask2[i, indices_to_change] = 0
    
    return data, mask1, mask2

# 自定义位置编码
class PositionalEncoding(nn.Module):
    """
    Transformer模型的位置编码
    """
    def __init__(self, d_model, max_len=NUM_DAYS):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册缓冲区(不是参数但应保存在state_dict中)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        将位置编码添加到输入嵌入中
        
        参数:
            x: 形状为(batch_size, seq_len, d_model)的输入嵌入
            
        返回:
            x + 位置编码
        """
        return x + self.pe[:, :x.size(1), :]

# 时间注意力池化，用于减少序列维度
class TemporalAttentionPool(nn.Module):
    """
    基于注意力的池化，用于减少时间维度
    """
    def __init__(self, embed_dim):
        super(TemporalAttentionPool, self).__init__()
        self.attention = nn.Linear(embed_dim, 1)
        
    def forward(self, x, mask):
        """
        在时间维度上应用注意力池化
        
        参数:
            x: 形状为(batch_size, seq_len, embed_dim)的输入张量
            mask: 形状为(batch_size, seq_len)的二进制掩码
                  其中0表示要忽略的位置
                  
        返回:
            pooled: 形状为(batch_size, embed_dim)的张量
        """
        # 获取注意力分数
        scores = self.attention(x).squeeze(-1)  # (batch_size, seq_len)
        
        # 将mask为0的位置的分数设置为-inf(以忽略这些位置)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # 应用注意力权重到嵌入
        pooled = torch.sum(x * attn_weights, dim=1)  # (batch_size, embed_dim)
        
        return pooled

# 时间序列Transformer编码器
class TimeSeriesTransformer(nn.Module):
    """
    基于Transformer的时间序列数据编码器
    """
    def __init__(self, input_dim, embed_dim, output_dim, nhead=8, num_layers=4, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # 光谱波段的嵌入
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # 云标志矩阵 - 用于标记云遮挡的时间步
        self.cloud_flag_matrix = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 时间池化和最终投影
        self.temporal_pool = TemporalAttentionPool(embed_dim)
        self.final_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim)
        )
        
    def forward(self, x, mask):
        """
        Transformer模型的前向传递
        
        参数:
            x: 形状为(batch_size, seq_len, input_dim)的输入张量
            mask: 形状为(batch_size, seq_len)的掩码张量，值为0, 1, 2
                 0: 无数据, 1: 云遮挡数据, 2: 有效数据
                 
        返回:
            output: 形状为(batch_size, output_dim)的张量
        """
        batch_size, seq_len, _ = x.shape
        
        # 将输入投影到嵌入维度
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # 创建云掩码(mask=1的位置)
        cloud_mask = (mask == 1).unsqueeze(-1).expand(-1, -1, self.embed_dim) # (batch_size, seq_len, embed_dim)
        
        # 将云标志变换应用于所有云遮挡时间步
        # 需要重塑进行批量矩阵乘法
        reshaped_embedded = embedded.reshape(-1, self.embed_dim)  # (batch_size*seq_len, embed_dim)
        transformed = torch.mm(reshaped_embedded, self.cloud_flag_matrix)  # (batch_size*seq_len, embed_dim)
        transformed = transformed.reshape(batch_size, seq_len, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        
        # 只在mask == 1(云遮挡)的位置应用变换
        embedded = torch.where(cloud_mask, transformed, embedded)
        
        # 添加位置编码
        embedded = self.pos_encoder(embedded)
        
        # 为transformer创建关键填充掩码(原始掩码为0的位置为True)
        # 这告诉transformer哪些位置要忽略
        key_padding_mask = (mask == 0)
        
        # 应用Transformer编码器
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=key_padding_mask)
        # 输出形状: (batch_size, seq_len, embed_dim)
        
        # 应用时间注意力池化
        pooled = self.temporal_pool(encoded, mask)
        # 输出形状: (batch_size, embed_dim)
        
        # 投影到输出维度
        output = self.final_projection(pooled)
        # 输出形状: (batch_size, output_dim)
        
        return output

# 主执行函数
def main():
    # 生成随机数据
    print("生成随机数据...")
    data, mask = generate_data(NUM_SAMPLES)
    print(f"数据形状: {data.shape}, 掩码形状: {mask.shape}")
    
    # 打印掩码统计信息
    mask_values, counts = torch.unique(mask, return_counts=True)
    print("掩码值分布:")
    for value, count in zip(mask_values.tolist(), counts.tolist()):
        print(f"  值 {value}: {count} 个 ({100 * count / mask.numel():.2f}%)")
    
    # 创建模型
    print("\n初始化模型...")
    model = TimeSeriesTransformer(
        input_dim=NUM_BANDS,
        embed_dim=EMBED_DIM,
        output_dim=OUTPUT_DIM
    )
    
    # 分批处理
    num_batches = NUM_SAMPLES // BATCH_SIZE
    print(f"\n处理 {num_batches} 批，每批大小 {BATCH_SIZE}...")
    
    for batch_idx in range(num_batches):
        print(f"\n批次 {batch_idx + 1}/{num_batches}")
        
        # 获取批次数据
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        
        batch_data = data[start_idx:end_idx]
        batch_mask = mask[start_idx:end_idx]
        
        # 创建增强
        print("  创建增强...")
        _, batch_mask1, batch_mask2 = create_augmentations(batch_data, batch_mask)
        
        # 第一批次的掩码值统计
        if batch_idx == 0:
            print("  第一个样本增强前的掩码统计:")
            for value in range(3):
                count = torch.sum(batch_mask[0] == value).item()
                print(f"    值 {value}: {count} 个")
                
            print("  第一个样本增强1后的掩码统计:")
            for value in range(3):
                count = torch.sum(batch_mask1[0] == value).item()
                print(f"    值 {value}: {count} 个")
                
            print("  第一个样本增强2后的掩码统计:")
            for value in range(3):
                count = torch.sum(batch_mask2[0] == value).item()
                print(f"    值 {value}: {count} 个")
        
        # 两个增强的前向传递
        print("  运行前向传递...")
        repr1 = model(batch_data, batch_mask1)
        repr2 = model(batch_data, batch_mask2)
        
        # 打印形状和样本值
        print(f"  增强1表示形状: {repr1.shape}")
        print(f"  增强2表示形状: {repr2.shape}")
        
        # 打印第一个样本的前几个值
        if batch_idx == 0:
            print("\n  第一个样本的表示(前5个值):")
            print(f"  增强1: {repr1[0][:5].tolist()}")
            print(f"  增强2: {repr2[0][:5].tolist()}")

if __name__ == "__main__":
    main()