#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import logging
import numpy as np
import json
import traceback

import torch
from torch.utils.data import DataLoader

# Setup early logging to capture import errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

try:
    # Add project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    logging.info(f"Added project root to path: {project_root}")

    # Import required modules
    logging.info("Importing required modules...")
    from models.ssl_model import MultimodalBTInferenceModel
    from models.builder import build_ssl_model
    from datasets.ssl_dataset import SingleTileInferenceDataset
    import importlib.util
    logging.info("All modules imported successfully")
except Exception as e:
    logging.error(f"Error during import: {str(e)}")
    logging.error(traceback.format_exc())
    sys.exit(1)

def load_config_module(config_file_path):
    try:
        logging.info(f"Loading config from: {config_file_path}")
        if not os.path.exists(config_file_path):
            logging.error(f"Config file does not exist: {config_file_path}")
            sys.exit(1)
            
        spec = importlib.util.spec_from_file_location("my_dynamic_config", config_file_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["my_dynamic_config"] = config_module
        spec.loader.exec_module(config_module)
        logging.info("Config loaded successfully")
        return config_module
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-tile Inference")
    parser.add_argument('--config', type=str, default="configs/multi_tile_infer_config_gpu.py", 
                        help="Path to config file")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use")
    parser.add_argument('--tile_list', type=str, required=True, 
                        help="JSON file containing list of tiles to process")
    parser.add_argument('--process_id', type=int, default=0, 
                        help="Process ID for logging purposes")
    args = parser.parse_args()
    logging.info(f"Args: {args}")
    return args

def sample_s2_batch(s2_bands_batch, s2_masks_batch, s2_doys_batch,
                band_mean, band_std, sample_size_s2, standardize=True):
    """
    Process S2 batch data with random sampling.
      s2_bands_batch.shape = (B, T_s2, 10)
      s2_masks_batch.shape = (B, T_s2)
      s2_doys_batch.shape  = (B, T_s2)
    Returns: np.array, shape=(B, sample_size_s2, 11), dtype float32
    """
    B = s2_bands_batch.shape[0]
    out_list = []
    for b in range(B):
        valid_idx = np.nonzero(s2_masks_batch[b])[0]
        
        if len(valid_idx) == 0:
            # If all timesteps are 0, use all indices
            valid_idx = np.arange(s2_bands_batch.shape[1])
        
        if len(valid_idx) < sample_size_s2:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=True)
        else:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s2, replace=False)
        idx_chosen = np.sort(idx_chosen)

        sub_bands = s2_bands_batch[b, idx_chosen, :]  # (sample_size_s2, 10)
        sub_doys  = s2_doys_batch[b, idx_chosen]      # (sample_size_s2,)
        if standardize:
            sub_bands = (sub_bands - band_mean) / (band_std + 1e-9)

        # Directly append doy
        out_arr = np.hstack([sub_bands, sub_doys.reshape(-1, 1)])  # (sample_size_s2, 11)
        
        out_list.append(out_arr.astype(np.float32))

    return np.stack(out_list, axis=0).astype(np.float32)  # (B, sample_size_s2, 11)

def sample_s1_batch(s1_asc_bands_batch, s1_asc_doys_batch,
                s1_desc_bands_batch, s1_desc_doys_batch,
                band_mean, band_std, sample_size_s1, standardize=True):
    """
    Process S1 batch data with random sampling.
      s1_asc_bands_batch.shape = (B, t_s1a, 2)
      s1_asc_doys_batch.shape  = (B, t_s1a)
      s1_desc_bands_batch.shape= (B, t_s1d, 2)
      s1_desc_doys_batch.shape = (B, t_s1d)
    Returns: np.array, shape=(B, sample_size_s1, 3), dtype float32
    """
    B = s1_asc_bands_batch.shape[0]
    out_list = []
    for b in range(B):
        s1_bands_all = np.concatenate([s1_asc_bands_batch[b], s1_desc_bands_batch[b]], axis=0)  # shape (t_s1a+t_s1d, 2)
        s1_doys_all  = np.concatenate([s1_asc_doys_batch[b], s1_desc_doys_batch[b]], axis=0)

        valid_mask = np.any(s1_bands_all != 0, axis=-1)
        valid_idx = np.nonzero(valid_mask)[0]
        if len(valid_idx) == 0:
            # If all timesteps are 0, use all indices
            valid_idx = np.arange(s1_bands_all.shape[0])
        if len(valid_idx) < sample_size_s1:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=True)
        else:
            idx_chosen = np.random.choice(valid_idx, size=sample_size_s1, replace=False)
        idx_chosen = np.sort(idx_chosen)

        sub_bands = s1_bands_all[idx_chosen, :]  # (sample_size_s1, 2)
        sub_doys  = s1_doys_all[idx_chosen]

        if standardize:
            sub_bands = (sub_bands - band_mean) / (band_std + 1e-9)

        # Directly append doy
        out_arr = np.hstack([sub_bands, sub_doys.reshape(-1, 1)])  # (sample_size_s1, 3)
        
        out_list.append(out_arr.astype(np.float32))

    return np.stack(out_list, axis=0).astype(np.float32)  # (B, sample_size_s1, 3)

def process_tile(tile_path, output_path, model, config, device, process_id):
    """Process a single tile with the loaded model"""
    logging.info(f"[Process {process_id}] Processing tile: {tile_path}")
    
    try:
        # Construct dataset
        logging.info(f"[Process {process_id}] Creating dataset for {os.path.basename(tile_path)}")
        dataset = SingleTileInferenceDataset(
            tile_path=tile_path,
            min_valid_timesteps=config["min_valid_timesteps"],
            standardize=False  # Standardize during sampling
        )
        
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=False
        )
        
        # Inference loop
        local_results = []
        start_time = time.time()
        
        logging.info(f"[Process {process_id}] {os.path.basename(tile_path)}: Dataset loaded, " 
                     f"shape={dataset.H}x{dataset.W}, total batches: {len(loader)}")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                global_idxs = batch_data["global_idx"]  # shape=(B,)
                
                # Get numpy data from batch
                s2_bands_batch = batch_data["s2_bands"].numpy()  # (B, t_s2, 10)
                s2_masks_batch = batch_data["s2_masks"].numpy()  # (B, t_s2)
                s2_doys_batch  = batch_data["s2_doys"].numpy()   # (B, t_s2)
                
                s1_asc_bands_batch = batch_data["s1_asc_bands"].numpy()   # (B, t_s1a, 2)
                s1_asc_doys_batch  = batch_data["s1_asc_doys"].numpy()    # (B, t_s1a)
                s1_desc_bands_batch= batch_data["s1_desc_bands"].numpy()   # (B, t_s1d, 2)
                s1_desc_doys_batch = batch_data["s1_desc_doys"].numpy()    # (B, t_s1d)
                
                B = s2_bands_batch.shape[0]
                sum_repr = None
                
                for r in range(config["repeat_times"]):
                    # S2
                    s2_input_np = sample_s2_batch(
                        s2_bands_batch, s2_masks_batch, s2_doys_batch,
                        band_mean=dataset.s2_band_mean,
                        band_std=dataset.s2_band_std,
                        sample_size_s2=config["sample_size_s2"],
                        standardize=True
                    )  # (B, sample_size_s2, 11) float32
                    
                    # S1
                    s1_input_np = sample_s1_batch(
                        s1_asc_bands_batch, s1_asc_doys_batch,
                        s1_desc_bands_batch, s1_desc_doys_batch,
                        band_mean=dataset.s1_band_mean,
                        band_std=dataset.s1_band_std,
                        sample_size_s1=config["sample_size_s1"],
                        standardize=True
                    )  # (B, sample_size_s1, 3) float32
                    
                    # Convert to Tensor 
                    s2_input = torch.tensor(s2_input_np, dtype=torch.float32, device=device)
                    s1_input = torch.tensor(s1_input_np, dtype=torch.float32, device=device)
                    
                    # Forward pass
                    z = model(s2_input, s1_input)
                    
                    if sum_repr is None:
                        sum_repr = z
                    else:
                        sum_repr += z
                
                avg_repr = sum_repr / float(config["repeat_times"])
                
                # Save to local_results
                avg_repr_np = avg_repr.cpu().numpy()  # (B, latent_dim)
                global_idxs_list = global_idxs.tolist()
                
                for b in range(B):
                    gidx = global_idxs_list[b]
                    local_results.append((gidx, avg_repr_np[b]))
                
                if batch_idx % 5 == 0:
                    logging.info(f"[Process {process_id}] {os.path.basename(tile_path)}: " 
                                 f"Processed batch {batch_idx}/{len(loader)} - "
                                 f"{len(local_results)} samples")
        
        # Save results
        final_gidx_np = np.array([item[0] for item in local_results], dtype=np.int64)
        final_vecs_np = np.array([item[1] for item in local_results], dtype=np.float32)
        
        H, W = dataset.H, dataset.W
        latent_dim = final_vecs_np.shape[1]
        
        out_array = np.full((H * W, latent_dim), 0, dtype=np.float32)
        out_array[final_gidx_np] = final_vecs_np
        out_array = out_array.reshape(H, W, latent_dim)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, out_array)
        logging.info(f"[Process {process_id}] {os.path.basename(tile_path)}: "
                     f"Saved representation to {output_path}, shape={out_array.shape}, "
                     f"time={time.time()-start_time:.2f}s")
        
        return output_path
    except Exception as e:
        logging.error(f"[Process {process_id}] Error processing tile {os.path.basename(tile_path)}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging with file handler
    log_file = f"infer_process_{args.process_id}.log"
    logging.info(f"Setting up logging to file: {log_file}")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    try:
        # Load configuration
        config_module = load_config_module(args.config)
        config = config_module.config
        
        # Log GPU info before loading anything
        logging.info(f"[Process {args.process_id}] CUDA available: {torch.cuda.is_available()}")
        logging.info(f"[Process {args.process_id}] CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logging.info(f"[Process {args.process_id}] GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Load tile list from JSON
        logging.info(f"[Process {args.process_id}] Loading tile list from: {args.tile_list}")
        if not os.path.exists(args.tile_list):
            logging.error(f"[Process {args.process_id}] Tile list file does not exist: {args.tile_list}")
            sys.exit(1)
            
        with open(args.tile_list, 'r') as f:
            tile_list_content = f.read()
            logging.info(f"[Process {args.process_id}] Tile list content (first 500 chars): {tile_list_content[:500]}")
            
            try:
                tile_list = json.loads(tile_list_content)
                logging.info(f"[Process {args.process_id}] Loaded {len(tile_list)} tiles from JSON")
            except json.JSONDecodeError as e:
                logging.error(f"[Process {args.process_id}] Error parsing JSON: {str(e)}")
                logging.error(f"[Process {args.process_id}] Content: {tile_list_content}")
                sys.exit(1)
        
        # Setup device
        device_id = args.gpu_id
        logging.info(f"[Process {args.process_id}] Setting up device: cuda:{device_id}")
        
        if not torch.cuda.is_available():
            logging.error(f"[Process {args.process_id}] CUDA is not available but required")
            sys.exit(1)
            
        if device_id >= torch.cuda.device_count():
            logging.error(f"[Process {args.process_id}] GPU {device_id} is not available, "
                          f"max is {torch.cuda.device_count()-1}")
            sys.exit(1)
            
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
        
        # Print configuration
        logging.info(f"[Process {args.process_id}] Configurations:")
        for k, v in config.items():
            logging.info(f"  {k}: {v}")
        
        logging.info(f"[Process {args.process_id}] Using GPU: {device_id} ({torch.cuda.get_device_name(device_id)})")
        logging.info(f"[Process {args.process_id}] Will process {len(tile_list)} tiles")
        
        # Check if checkpoint file exists
        if not os.path.exists(config["checkpoint_path"]):
            logging.error(f"[Process {args.process_id}] Checkpoint file does not exist: {config['checkpoint_path']}")
            sys.exit(1)
        
        # Build and load SSL model (only once)
        logging.info(f"[Process {args.process_id}] Building SSL model...")
        ssl_model = build_ssl_model(config, device)
        
        logging.info(f"[Process {args.process_id}] Loading checkpoint from {config['checkpoint_path']}")
        try:
            checkpoint = torch.load(config["checkpoint_path"], map_location=device)
            logging.info(f"[Process {args.process_id}] Checkpoint loaded, keys: {checkpoint.keys()}")
            
            state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
            logging.info(f"[Process {args.process_id}] Using state key: {state_key}")
            
            if state_key not in checkpoint:
                logging.error(f"[Process {args.process_id}] State key '{state_key}' not found in checkpoint")
                logging.error(f"[Process {args.process_id}] Available keys: {list(checkpoint.keys())}")
                sys.exit(1)
                
            # Handle FSDP prefixes
            state_dict = checkpoint[state_key]
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]  # Remove prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            # Load state dict
            logging.info(f"[Process {args.process_id}] Loading state dict into model")
            ssl_model.load_state_dict(new_state_dict, strict=True)
            logging.info(f"[Process {args.process_id}] Model loaded successfully")
        except Exception as e:
            logging.error(f"[Process {args.process_id}] Error loading checkpoint: {str(e)}")
            logging.error(traceback.format_exc())
            sys.exit(1)
        
        # Freeze SSL backbone parameters
        logging.info(f"[Process {args.process_id}] Freezing SSL backbone parameters")
        for param in ssl_model.s2_backbone.parameters():
            param.requires_grad = False
        for param in ssl_model.s1_backbone.parameters():
            param.requires_grad = False
        for param in ssl_model.dim_reducer.parameters():
            param.requires_grad = False
        
        # Build inference model
        logging.info(f"[Process {args.process_id}] Building inference model")
        model = MultimodalBTInferenceModel(
            s2_backbone=ssl_model.s2_backbone,
            s1_backbone=ssl_model.s1_backbone,
            fusion_method=config["fusion_method"],
            dim_reducer=ssl_model.dim_reducer,
        ).to(device)
        model.eval()
        
        logging.info(f"[Process {args.process_id}] Model initialized and ready for inference")
        
        # Process each tile
        output_dir = config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"[Process {args.process_id}] Output directory: {output_dir}")
        
        for i, tile_path in enumerate(tile_list):
            tile_name = os.path.basename(tile_path)
            output_path = os.path.join(output_dir, f"{tile_name}.npy")
            
            logging.info(f"[Process {args.process_id}] Processing tile {i+1}/{len(tile_list)}: {tile_name}")
            try:
                process_tile(tile_path, output_path, model, config, device, args.process_id)
                logging.info(f"[Process {args.process_id}] Completed tile {i+1}/{len(tile_list)}: {tile_name}")
            except Exception as e:
                logging.error(f"[Process {args.process_id}] Error processing tile {tile_name}: {str(e)}")
                logging.error(traceback.format_exc())
                # Continue with next tile instead of exiting
                continue
        
        logging.info(f"[Process {args.process_id}] All tiles processed successfully")
        
    except Exception as e:
        logging.error(f"[Process {args.process_id}] Unhandled error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()