import torch
import torch.nn as nn
import csv

"""
ProjectionHead 参数量计算器

网络结构：
- 输入维度：固定为 128
- 中间层维度（宽度）：可变
- 输出维度：等于中间层维度（宽度）
- 深度：可变（表示有几个 Linear->BN->ReLU 块）
"""

class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim, depth):
        """
        根据用户数据推测的网络结构：
        - depth表示Linear+BN+ReLU块的数量（最少为1）
        - 最后总是有一个额外的Linear层（没有BN和ReLU）
        
        depth=0和depth=1都是：input -> hidden (with BN+ReLU) -> output
        depth=2: input -> hidden -> hidden -> output
        """
        super().__init__()
        layers = []
        
        input_dim = 128  # 固定输入维度为128
        
        # 第一层: input -> hidden
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False)
        ])
        
        # 中间层 (depth个)
        for _ in range(depth):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=False)
            ])
        
        # 输出层 (没有BN和ReLU)
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def count_parameters(model):
    """计算模型的总参数量"""
    return sum(p.numel() for p in model.parameters())

def generate_csv():
    """生成CSV文件"""
    # 深度列表（中间隐藏层的数量）
    depths = [0, 1, 2, 4, 6, 8]
    widths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    # 准备CSV数据
    csv_data = []
    
    for depth in depths:
        for width in widths:
            # 创建模型
            model = ProjectionHead(width, depth)
            
            # 计算参数量
            param_count = count_parameters(model)
            param_count_m = param_count / 1e6  # 转换为百万(M)
            
            csv_data.append([depth, width, round(param_count_m, 3)])
            print(f"深度={depth}, 宽度={width}, 参数量={param_count_m:.3f}M")
    
    # 写入CSV文件
    filename = 'projection_head_params.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        writer.writerow(['depth', 'width', 'param(M)'])
        
        # 写入数据
        writer.writerows(csv_data)
    
    print(f"CSV文件已生成: {filename}")
    
    # 显示部分结果以供确认
    print("\n生成的数据预览：")
    print("depth  width    param(M)")
    print("-" * 30)
    for i, row in enumerate(csv_data):
        if i < 5 or i >= len(csv_data) - 5:  # 显示前5行和后5行
            print(f"{row[0]:5d}  {row[1]:5d}  {row[2]:10.3f}")
        elif i == 5:
            print("...")
    
    return csv_data

def verify_params():
    """验证参数计算的正确性"""
    print("\n参数量计算验证：")
    print("=" * 60)
    
    # 示例1：深度=0，宽度=128
    model = ProjectionHead(128, 0)
    
    print(f"示例模型结构（深度=0, 宽度=128）：")
    print(f"- Linear: 128 -> 128  参数: {128 * 128 + 128} = {128 * 128 + 128}")
    print(f"- BatchNorm1d(128)   参数: {128 * 2} = {128 * 2}")
    print(f"- Linear: 128 -> 128  参数: {128 * 128 + 128} = {128 * 128 + 128}")
    
    total_expected = (128 * 128 + 128) + (128 * 2) + (128 * 128 + 128)
    total_actual = count_parameters(model)
    
    print(f"\n预期总参数: {total_expected:,}")
    print(f"实际总参数: {total_actual:,}")
    print(f"参数量(M): {total_actual / 1e6:.3f}")
    
    # 示例2：深度=1，宽度=256
    print("\n" + "-" * 40)
    model2 = ProjectionHead(256, 1)
    
    print(f"\n示例模型结构（深度=1, 宽度=256）：")
    print("层结构：input(128) -> hidden(256) -> hidden(256) -> output(256)")
    print(f"- Linear: 128 -> 256  参数: {128 * 256 + 256}")
    print(f"- BatchNorm1d(256)   参数: {256 * 2}")
    print(f"- Linear: 256 -> 256  参数: {256 * 256 + 256}")
    print(f"- BatchNorm1d(256)   参数: {256 * 2}")
    print(f"- Linear: 256 -> 256  参数: {256 * 256 + 256}")
    
    print(f"\n实际总参数: {count_parameters(model2):,}")
    print(f"参数量(M): {count_parameters(model2) / 1e6:.3f}")

if __name__ == "__main__":
    # 生成CSV文件
    generate_csv()
    
    # 验证参数计算
    verify_params()