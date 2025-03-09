import sys
import logging
import random
import pandas as pd
import torch
import torch.nn as nn

# 如果 models.py 在其他地方，请自行调整路径或改写import
sys.path.append('src/models')
from modules import TransformerEncoder

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_transformer_core_param_count(
    band_num: int,
    latent_dim: int,
    nhead: int,
    num_encoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    max_seq_len: int
) -> int:
    """
    实例化一个 TransformerEncoder，并返回“非embedding”部分的参数总数 (numel).
    即：total_params - embedding与pos_encoder的参数量。
    """
    encoder = TransformerEncoder(
        band_num=band_num,
        latent_dim=latent_dim,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len
    )
    total_params = sum(p.numel() for p in encoder.parameters())

    # 统计 embedding、pos_encoder 的参数量
    embedding_params = 0
    pos_encoder_params = 0

    # 1) 如果 embedding 是 nn.Sequential 或 nn.Module
    if hasattr(encoder, 'embedding'):
        embedding_params = sum(p.numel() for p in encoder.embedding.parameters())

    # 2) 如果 pos_encoder 是一个 nn.Parameter
    if hasattr(encoder, 'pos_encoder') and isinstance(encoder.pos_encoder, nn.Parameter):
        pos_encoder_params = encoder.pos_encoder.numel()
    # 如 pos_encoder 是其他形式的模块，也可类似处理

    non_embed_param_count = total_params - (embedding_params + pos_encoder_params)
    return non_embed_param_count


def pick_closest_target(value, targets):
    """
    给定 value 和一组 target 列表, 返回和 value 最接近的 target.
    """
    return min(targets, key=lambda t: abs(t - value))


def generate_param_sweep(
    # 基本配置
    s2_band_num=12,
    s1_band_num=4,
    latent_dim=128,
    dropout=0.1,
    s2_max_seq_len=20,
    s1_max_seq_len=20,
    # 目标总参数数
    total_targets=(8e6, 14e6, 28e6, 36e6, 64e6, 128e6),
    # 搜索空间配置
    n_samples=3000,     # 先随机生成多少组
    num_rows=50,        # 最终在CSV里保留多少行
    random_seed=42
):
    random.seed(random_seed)

    # -- 1) 定义更丰富的搜索空间 --
    # 要求 latent_dim=128 能被 nhead 整除，否则会报错 "embed_dim must be divisible by num_heads"
    # 128 = 2^7，因此它的正因数有1,2,4,8,16,32,64,128
    nhead_candidates = [1, 2, 4, 8, 16, 32, 64]  # 128 头可能太大，这里暂且不加

    # 层数可选
    layer_candidates = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 36, 40, 42, 46, 48, 52, 64, 72, 80, 96, 100, 104, 108, 112, 120, 128]

    # FF维度可选
    dim_ff_candidates = [
        128, 144, 168, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960,
        1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920,
        1984, 2048, 2112, 2176, 2240, 2304, 2368, 2432, 2496, 2560, 2624, 2688, 2752, 2816, 2880,
        2944, 3008, 3072, 3136, 3200, 3264
    ]

    combos = []

    # -- 2) 随机采样 n_samples 组超参 --
    for i in range(n_samples):
        s2_nhead = random.choice(nhead_candidates)
        s2_layer = random.choice(layer_candidates)
        s2_dim_ff = random.choice(dim_ff_candidates)

        s1_nhead = random.choice(nhead_candidates)
        s1_layer = random.choice(layer_candidates)
        s1_dim_ff = random.choice(dim_ff_candidates)

        # 计算 s2, s1 的核心参数量
        s2_params = get_transformer_core_param_count(
            band_num=s2_band_num,
            latent_dim=latent_dim,
            nhead=s2_nhead,
            num_encoder_layers=s2_layer,
            dim_feedforward=s2_dim_ff,
            dropout=dropout,
            max_seq_len=s2_max_seq_len
        )
        s1_params = get_transformer_core_param_count(
            band_num=s1_band_num,
            latent_dim=latent_dim,
            nhead=s1_nhead,
            num_encoder_layers=s1_layer,
            dim_feedforward=s1_dim_ff,
            dropout=dropout,
            max_seq_len=s1_max_seq_len
        )

        # 保证 s2_params > s1_params
        if s2_params == 0 or s1_params == 0:
            continue
        if s1_params >= s2_params:
            # 若想要 ratio < 1, 则跳过这组
            continue

        total_params = s2_params + s1_params
        ratio_s1_s2 = round(s1_params / s2_params, 4)
        best_target = pick_closest_target(total_params, total_targets)

        # 可在 debug 级别打印，以免刷屏
        logger.debug(
            f"[{i+1}/{n_samples}] "
            f"S2=({s2_nhead},{s2_layer},{s2_dim_ff})=>{s2_params} | "
            f"S1=({s1_nhead},{s1_layer},{s1_dim_ff})=>{s1_params} | "
            f"total={total_params} ~ {best_target}"
        )

        combos.append({
            's2nhead': s2_nhead,
            's2layer': s2_layer,
            's2dim_feedforward': s2_dim_ff,
            's2non_embedding': s2_params,

            's1nhead': s1_nhead,
            's1layer': s1_layer,
            's1dim_feedforward': s1_dim_ff,
            's1non_embedding': s1_params,

            'ratio_s1/s2': ratio_s1_s2,
            'alltotal': best_target,
            'actual_total': total_params
        })
        
        if i%20 == 0:
            logger.info(f"Generated {i+1} random combos...")

    # -- 3) 将结果转为 DataFrame，并计算与目标值的差值 --
    df = pd.DataFrame(combos)
    logger.info(f"Total valid combos (s2>s1) = {len(df)} from {n_samples} random trials.")
    df['diff_to_target'] = abs(df['actual_total'] - df['alltotal'])

    # -- 4) 按 alltotal 分组，并尽可能均衡地选取 num_rows 行 --
    # 将每个组按照 diff_to_target 排序
    groups = {}
    for target, group in df.groupby('alltotal'):
        groups[target] = group.sort_values('diff_to_target')

    unique_targets = list(groups.keys())
    n_groups = len(unique_targets)
    # 初始配额：每组至少分得 base_quota 行
    base_quota = num_rows // n_groups
    extra = num_rows % n_groups

    quotas = {target: base_quota for target in unique_targets}
    # 将余数均匀分配给各组（按照 target 顺序）
    for target in sorted(unique_targets):
        if extra > 0:
            quotas[target] += 1
            extra -= 1

    # 从每个组中按排序结果选取 quota 数量的行（如果不足则全部选取）
    selected_dfs = {}
    total_selected = 0
    for target in unique_targets:
        group = groups[target]
        num_select = min(len(group), quotas[target])
        selected_dfs[target] = group.iloc[:num_select]
        total_selected += num_select

    # 如果总数未达到 num_rows，则在各组中进行轮询补足
    remaining = num_rows - total_selected
    pointers = {target: len(selected_dfs[target]) for target in unique_targets}
    while remaining > 0:
        added_this_round = False
        for target in sorted(unique_targets):
            if remaining <= 0:
                break
            group = groups[target]
            pointer = pointers[target]
            if pointer < len(group):
                selected_dfs[target] = pd.concat([selected_dfs[target], group.iloc[[pointer]]])
                pointers[target] += 1
                remaining -= 1
                added_this_round = True
        if not added_this_round:
            # 如果所有组都没有更多候选，则跳出循环
            break

    final_df = pd.concat(selected_dfs.values())
    final_df.sort_values('diff_to_target', inplace=True)

    # -- 5) 计算各 target 在最终 CSV 中所占比例，并生成 summary DataFrame --
    proportions = final_df['alltotal'].value_counts(normalize=True).sort_index()
    summary_df = pd.DataFrame({
        'alltotal': proportions.index,
        'proportion': proportions.values
    })

    # -- 6) 写 CSV --
    csv_filename = 'param_sweep.csv'
    final_df.drop(columns=['diff_to_target'], inplace=True)
    final_df.to_csv(csv_filename, index=False)
    logger.info(f"Saved top {num_rows} param combos to {csv_filename}.")

    summary_csv = 'param_sweep_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved parameter distribution summary to {summary_csv}.")

    logger.info("Parameter distribution proportions in final CSV:")
    for _, row in summary_df.iterrows():
        logger.info(f"Target {row['alltotal']}: {row['proportion']*100:.2f}%")

    return final_df


if __name__ == "__main__":
    df_result = generate_param_sweep(
        s2_band_num=12,
        s1_band_num=4,
        latent_dim=128,
        dropout=0.1,
        s2_max_seq_len=20,
        s1_max_seq_len=20,
        total_targets=(4e6, 8e6, 16e6, 24e6, 32e6, 40e6, 48e6, 64e6, 96e6, 128e6), 
        n_samples=5000,
        num_rows=200,
        random_seed=42
    )
    print(df_result)
