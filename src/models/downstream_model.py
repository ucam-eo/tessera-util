# src/models/downstream_model.py

import torch
import torch.nn as nn
from .modules import FusionTransformer

class ClassificationHead(nn.Module):
    """
    下游分类任务的MLP
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        return x
    
class LinearProbeHead(nn.Module):
    """
    下游线性探针任务的MLP
    """
    def __init__(self, input_dim, num_classes):
        super(LinearProbeHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)
    
class RegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RegressionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class MultimodalDownstreamModel(nn.Module):
    """
    将SSL中训练好的s2和s1骨干接上下游任务head
    """
    def __init__(self, s2_backbone, s1_backbone, head, dim_reducer, fusion_method):
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.head = head
        self.fusion_method = fusion_method
        self.dim_reducer = dim_reducer

    def forward(self, s2_x, s1_x):
        s2_repr = self.s2_backbone(s2_x)
        s1_repr = self.s1_backbone(s1_x)
        if self.fusion_method == 'concat':
            fused_repr = torch.cat([s2_repr, s1_repr], dim=-1)
        elif self.fusion_method == 'sum':
            fused_repr = s2_repr + s1_repr
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        fused_repr = self.dim_reducer(fused_repr)
        out = self.head(fused_repr)
        return out
