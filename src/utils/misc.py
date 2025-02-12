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
