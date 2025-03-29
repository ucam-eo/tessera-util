# src/utils/misc.py

import os
import logging
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch

# generates and returns an image of the cross-correlation matrix
# for the provided z0 and z1 arrays
# z is (batch_size, latent_dim) array
def plot_cross_corr(z0, z1):
    z0 = z0.detach().cpu().numpy()
    z1 = z1.detach().cpu().numpy()
    z0 = (z0 - z0.mean(axis=0)) / z0.std(axis=0)
    z1 = (z1 - z1.mean(axis=0)) / z1.std(axis=0)
    C = np.matmul(z0.T, z1) / z0.shape[0]
    C = np.abs(C)
    fig, ax = plt.subplots()
    im = ax.imshow(C, cmap='binary', interpolation='nearest')
    ax.set_title("Embeddings cross-correlation")
    plt.colorbar(im, ax=ax)
    return fig

def remove_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        logging.info(f"Removed directory: {dir_path}")

def save_checkpoint(model, optimizer, epoch, step, val_acc, ckpt_path):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc
    }, ckpt_path)
    logging.info(f"Saved checkpoint at {ckpt_path}")


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

def plot_4_augs_in_one(mask_s2_aug1, mask_s2_aug2, mask_s1_aug1, mask_s1_aug2, title="All Mask Visualization"):
    """
    将 4 个 mask (形状 [365,], 值 in {0,1,2}) 合并到一张图里：
      - 上面一行: S2_Aug1, S2_Aug2，中间用 1-pixel thin white line 分隔
      - 中间: 2-pixel thick white line
      - 下面一行: S1_Aug1, S1_Aug2，之间也用 1-pixel thin white line

    每个mask只取前364天 => reshape (4,91)，再转置 => (91,4)，得到小块 shape=(91,4)。
    整张大图 shape = ( (91) + (2) + (91),  (4) + (1) + (4) ) = (184, 9).
    其中:
      - thin line = 1 列
      - thick line = 2 行
    数值用 3 代表白线.
    """

    # --- 先把四个mask各取前364天（也可选不截断，由你决定）---
    m_s2_a1 = mask_s2_aug1[:364]
    m_s2_a2 = mask_s2_aug2[:364]
    m_s1_a1 = mask_s1_aug1[:364]
    m_s1_a2 = mask_s1_aug2[:364]

    # --- reshape 到 (4,91) -> transpose -> (91,4) ---
    def reshape_4x91(m):
        g = m.reshape(4, 91)
        return g.T  # => (91,4)

    block_s2_a1 = reshape_4x91(m_s2_a1)
    block_s2_a2 = reshape_4x91(m_s2_a2)
    block_s1_a1 = reshape_4x91(m_s1_a1)
    block_s1_a2 = reshape_4x91(m_s1_a2)

    # --- 准备大矩阵 (H=91+2+91=184, W=4+1+4=9) ---
    H = 91 + 2 + 91
    W = 4 + 1 + 4
    bigmat = np.full((H, W), 3, dtype=np.int32)  # 3=white

    # 约定: 
    #   top-left  => S2_Aug1 => rows [0:91], cols [0:4]
    #   top-right => S2_Aug2 => rows [0:91], cols [5:9]  # col4是thin line
    #   thick line => rows [91:93], all cols => 3
    #   bottom-left => S1_Aug1 => rows [93:93+91], cols [0:4]
    #   bottom-right => S1_Aug2 => rows [93:93+91], cols [5:9]

    # 填充 top-left
    bigmat[0:91, 0:4] = block_s2_a1
    # 填充 top-right
    bigmat[0:91, 5:9] = block_s2_a2
    # 中间 [91:93, :] 已经是 3 (white) => thick line
    # 填充 bottom-left
    bigmat[93:93+91, 0:4] = block_s1_a1
    # 填充 bottom-right
    bigmat[93:93+91, 5:9] = block_s1_a2

    # --- colormap: 0->blue, 1->yellow, 2->green, 3->white
    cmap = mcolors.ListedColormap(["blue", "yellow", "green", "white"])
    
    fig, ax = plt.subplots(figsize=(6,10))
    im = ax.imshow(bigmat, cmap=cmap, vmin=0, vmax=3, aspect='auto')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    # 自定义图例
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='s', color='w', label='0=missing', markerfacecolor='blue', markersize=10),
        Line2D([0],[0], marker='s', color='w', label='1=invalid', markerfacecolor='yellow', markersize=10),
        Line2D([0],[0], marker='s', color='w', label='2=valid', markerfacecolor='green', markersize=10),
        Line2D([0],[0], marker='s', color='w', label='3=white line', markerfacecolor='white', markersize=10),
    ]
    ax.legend(handles=legend_elems, loc='best')
    fig.tight_layout()
    return fig
