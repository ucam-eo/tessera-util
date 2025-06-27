import os
import time
import argparse
import logging
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
import gc

from models.modules import TransformerEncoder, ProjectionHead
from models.ssl_model import MultimodalBTModel, MultimodalBTInferenceModel
from datasets.downstream_dataset import EthiopiaInferenceDataset, ethiopia_crop_infer_fn
import importlib.util
import sys

# Add project root to sys.path (assuming src and configs are in the same directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def load_config_module(config_file_path):
    spec = importlib.util.spec_from_file_location("my_dynamic_config", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["my_dynamic_config"] = config_module
    spec.loader.exec_module(config_module)
    return config_module

def parse_args():
    parser = argparse.ArgumentParser(description="Ethiopia Crop Classification Inference")
    parser.add_argument('--config', type=str, default="configs/ethiopia_infer_config.py", help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    config_module = load_config_module(args.config)
    config = config_module.config
    
    # Setup logging & device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print config
    logging.info("Configurations:")
    for k, v in config.items():
        logging.info(f"  {k}: {v}")
    
    # Create output directory
    os.makedirs(os.path.dirname(config['output_npz']), exist_ok=True)
    
    # Create dataset
    dataset = EthiopiaInferenceDataset(
        polygons_locations_csv=config['polygons_locations_csv'],
        data_processed_dir=config['data_processed_dir'],
        data_raw_dir=config['data_raw_dir'],
        class_names=config['class_names'],
        standardize=True,
        max_seq_len_s2=config['max_seq_len_s2'],
        max_seq_len_s1=config['max_seq_len_s1'],
    )
    
    # Create DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'], 
        collate_fn=ethiopia_crop_infer_fn, 
        pin_memory=True
    )
    
    logging.info(f"Dataset size: {len(dataset)}")
    
    # Build model components
    latent_dim = config['latent_dim']
    
    s2_backbone = TransformerEncoder(
        band_num=10,
        latent_dim=latent_dim,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_len=config['max_seq_len_s2']
    )
    
    s1_backbone = TransformerEncoder(
        band_num=2,
        latent_dim=latent_dim,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_len=config['max_seq_len_s1']
    )
    
    input_dim_for_projector = latent_dim
    projector = ProjectionHead(
        input_dim_for_projector, 
        config['projector_hidden_dim'], 
        config['projector_out_dim']
    )
    
    # Build SSL model
    ssl_model = MultimodalBTModel(
        s2_backbone, 
        s1_backbone, 
        projector,
        fusion_method=config['fusion_method'], 
        return_repr=True
    ).to(device)
    
    # Load checkpoint
    logging.info(f"Loading checkpoint from {config['checkpoint_path']}")
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
    
    # Handle FSDP state dict
    state_dict = checkpoint[state_key]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]  # Remove prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load state dict
    ssl_model.load_state_dict(new_state_dict, strict=True)
    
    # Build inference model - without the projector head
    infer_model = MultimodalBTInferenceModel(
        s2_backbone=ssl_model.s2_backbone,
        s1_backbone=ssl_model.s1_backbone,
        fusion_method=config['fusion_method'],
        dim_reducer=ssl_model.dim_reducer,
    ).to(device)
    
    # Set model to evaluation mode
    infer_model.eval()
    
    # Inference loop
    all_representations = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (s2_batch, s1_batch, labels) in enumerate(tqdm.tqdm(loader, desc="Inference")):
            s2_batch = s2_batch.to(device)
            s1_batch = s1_batch.to(device)
            
            # Initialize sum for averaging multiple runs
            sum_repr = None
            
            # Run multiple forward passes for robustness (like your original code)
            for r in range(config['repeat_times']):
                # Forward pass
                representations = infer_model(s2_batch, s1_batch)
                
                if sum_repr is None:
                    sum_repr = representations
                else:
                    sum_repr += representations
            
            # Average the representations
            avg_repr = sum_repr / config['repeat_times']
            
            # Store results
            all_representations.append(avg_repr.cpu().numpy())
            all_labels.append(labels.numpy())
            
            if batch_idx % 10 == 0:
                logging.info(f"Processed {batch_idx+1}/{len(loader)} batches")
    
    # Concatenate results
    all_representations = np.concatenate(all_representations, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    logging.info(f"Final representation shape: {all_representations.shape}")
    logging.info(f"Final labels shape: {all_labels.shape}")
    
    # Save results
    np.savez(
        config['output_npz'],
        representations=all_representations,
        labels=all_labels
    )
    
    logging.info(f"Saved results to {config['output_npz']}")

if __name__ == "__main__":
    main()