# src/utils/metrics.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from sklearn.ensemble import RandomForestClassifier

import csv
from sklearn.ensemble import RandomForestClassifier

from contextlib import nullcontext # For amp_ctx
import gc
import time
import logging

# Import quantization helper functions
# Adjust the path based on where you placed quantization.py
# Assuming models/quantization.py relative to src/
try:
    from models.quantization import quantize_tensor_symmetric, dequantize_tensor_symmetric
except ImportError:
    # Fallback for direct execution or different structure, you might need to adjust this
    logging.warning("Could not import quantization functions from models.quantization. Ensure it's in PYTHONPATH.")
    # Define them locally if import fails (not recommended for modularity)
    def quantize_tensor_symmetric(x_f32, bits=8):
        if bits == 8: qmin, qmax, dtype = -128, 127, torch.int8
        else: qmin, qmax, dtype = -2**(bits-1), 2**(bits-1)-1, torch.int32 # Simplified
        abs_max = torch.max(torch.abs(x_f32.detach())); scale = torch.clamp(abs_max / qmax, min=1e-8)
        return torch.clamp(torch.round(x_f32 / scale), qmin, qmax).to(dtype), scale
    def dequantize_tensor_symmetric(x_q, scale): return x_q.float() * scale


def rankme(z, eps=1e-7):
    # Convert to float32 for SVD
    # if z.dtype == torch.float16:
    z = z.to(torch.float32)
    # Perform SVD
    s = z.svd(compute_uv=False)[1]
    # Calculate rank metrics
    p = s / (s.sum() + eps)
    entropy = -(p * torch.log(p + eps)).sum()
    rankme_score = entropy / torch.log(torch.tensor(float(len(s))))
    return rankme_score


# def linear_probe_evaluate(model, val_loader, field_ids=None, field_data_path=None, 
#                          training_ratio=0.3, val_test_split_ratio=0.5, 
#                          classifier_type='lr', num_inference=1, device='cuda',
#                          apply_amp=False):
#     """
#     Using validation set to compute the performance of a classifier trained on model embeddings,
#     returns accuracy, weighted F1 score and confusion matrix.
    
#     Parameters:
#     - model: Trained model (can be loaded from checkpoint)
#     - val_loader: Validation data loader
#     - field_ids: Optional, for field-based splitting
#     - field_data_path: Optional, CSV file path for field data
#     - training_ratio: Field area ratio for training, default 0.3
#     - val_test_split_ratio: Validation/test split ratio, default 0.5
#     - classifier_type: Classifier type, 'lr' or 'rf', default 'lr'
#     - num_inference: Number of inferences to average, default 1
#     - device: Computation device, default 'cuda'
#     - apply_amp: Whether to use automatic mixed precision, default False
    
#     Returns:
#     - accuracy, f1_score, confusion matrix, classification report
#     """
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
#     import csv
#     import os
#     from contextlib import nullcontext
#     import gc
#     import time
#     import logging
    
#     np.random.seed(42)
#     torch.manual_seed(42)
    
#     logging.info(f"Starting linear probe evaluation with apply_amp={apply_amp}")
    
#     # Make sure model is in eval mode
#     model.eval()
    
#     # 打印embedding的值
#     logging.info(f"Embedding weight during evaluation: {model.s2_backbone.embedding[0].weight}")
#     logging.info(f"Embedding bias during evaluation: {model.s2_backbone.embedding[0].bias}")
    
#     # Initialize containers
#     embeddings_dict = {}
#     labels_dict = {}
#     field_id_dict = {}
    
#     # Define autocast context
#     amp_ctx = torch.amp.autocast("cuda") if apply_amp else nullcontext()
    
#     # Process embeddings in smaller batches to prevent OOM
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     # Set a batch processing timeout
#     batch_timeout = 120  # seconds
    
#     # Collect embeddings with proper error handling
#     with torch.no_grad():
#         for batch_idx, batch_data in enumerate(val_loader):
#             try:
#                 # Extract data from batch with appropriate error handling
#                 if isinstance(batch_data, dict):
#                     # Handle dict-type batch
#                     batch_s2 = batch_data.get('s2', None)
#                     batch_s1 = batch_data.get('s1', None)
#                     batch_label = batch_data.get('label', None)
#                     batch_field_id = batch_data.get('field_id', None)
#                     batch_pos = batch_data.get('pos', None)
#                 elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 3:
#                     # Handle tuple-type batch
#                     s2_sample, s1_sample, label = batch_data[:3]
#                     batch_s2 = s2_sample
#                     batch_s1 = s1_sample
#                     batch_label = label
                    
#                     field_id = batch_data[3] if len(batch_data) > 3 else None
#                     batch_field_id = field_id
                    
#                     pos = batch_data[4] if len(batch_data) > 4 else None
#                     batch_pos = pos
#                 else:
#                     logging.error(f"Unexpected batch format: {type(batch_data)}")
#                     continue
                
#                 # Skip if we don't have the necessary data
#                 if batch_s2 is None or batch_s1 is None or batch_label is None or batch_pos is None:
#                     logging.warning(f"Skipping batch {batch_idx} - missing required data")
#                     continue
                
#                 # Transfer data to device
#                 batch_s2 = batch_s2.to(device, non_blocking=True)
#                 batch_s1 = batch_s1.to(device, non_blocking=True)
#                 batch_label = batch_label.cpu().numpy()
#                 if isinstance(batch_field_id, torch.Tensor):
#                     batch_field_id = batch_field_id.cpu().numpy()
#                 batch_pos = batch_pos.cpu().numpy()
                
#                 # Set a timer for this batch
#                 start_time = time.time()
                
#                 # Multiple inference with averaging
#                 embeddings_sum = None
                
#                 for inference_idx in range(num_inference):
#                     # Check if timeout occurred
#                     if time.time() - start_time > batch_timeout:
#                         logging.warning(f"Timeout processing batch {batch_idx}, inference {inference_idx}")
#                         break
                        
#                     # Use consistent precision context
#                     with amp_ctx:
#                         try:
#                             out = model(batch_s2, batch_s1)
#                             if isinstance(out, tuple):
#                                 out = out[1]  # Get representation not projection
                            
#                             # Transfer to CPU and convert to numpy
#                             out_cpu = out.cpu()
                            
#                             if embeddings_sum is None:
#                                 embeddings_sum = out_cpu.numpy()
#                             else:
#                                 embeddings_sum += out_cpu.numpy()
#                         except RuntimeError as e:
#                             logging.error(f"Error in inference {inference_idx}: {e}")
#                             # Try clearing cache and retry with smaller batch if possible
#                             torch.cuda.empty_cache()
#                             if "out of memory" in str(e) and batch_s2.shape[0] > 1:
#                                 logging.info("OOM error, trying with half batch size...")
#                                 # Split into smaller batches
#                                 half_point = batch_s2.shape[0] // 2
                                
#                                 # Process first half
#                                 with amp_ctx:
#                                     out1 = model(batch_s2[:half_point], batch_s1[:half_point])
#                                     if isinstance(out1, tuple):
#                                         out1 = out1[1]
                                
#                                 # Process second half
#                                 with amp_ctx:
#                                     out2 = model(batch_s2[half_point:], batch_s1[half_point:])
#                                     if isinstance(out2, tuple):
#                                         out2 = out2[1]
                                
#                                 # Combine results
#                                 out = torch.cat([out1, out2], dim=0)
#                                 out_cpu = out.cpu()
                                
#                                 if embeddings_sum is None:
#                                     embeddings_sum = out_cpu.numpy()
#                                 else:
#                                     embeddings_sum += out_cpu.numpy()
#                             else:
#                                 # If not OOM or can't split, just skip this batch
#                                 raise
                
#                 # Average embeddings
#                 if embeddings_sum is not None:
#                     emb = embeddings_sum / num_inference
                    
#                     # Store results
#                     for i in range(len(batch_pos)):
#                         position = tuple(batch_pos[i])
#                         embeddings_dict[position] = emb[i]
#                         labels_dict[position] = batch_label[i]
#                         if isinstance(batch_field_id, np.ndarray):
#                             field_id_dict[position] = batch_field_id[i]
                
#                 # Periodically clean GPU memory
#                 if batch_idx % 5 == 0:
#                     torch.cuda.empty_cache()
                    
#                 # Log progress
#                 if batch_idx % 10 == 0:
#                     logging.info(f"Processed {batch_idx}/{len(val_loader)} validation batches")
                
#             except Exception as e:
#                 logging.error(f"Error in batch {batch_idx}: {e}")
#                 import traceback
#                 logging.error(traceback.format_exc())
#                 torch.cuda.empty_cache()
#                 continue
    
#     logging.info(f"Collected embeddings for {len(embeddings_dict)} positions")
    
#     # Field-based splitting
#     field_split_success = False
    
#     if field_data_path is not None and os.path.exists(field_data_path) and field_ids is not None:
#         try:
#             # Load field data
#             field_data = []
#             with open(field_data_path, 'r') as f:
#                 csv_reader = csv.reader(f)
#                 header = next(csv_reader)
                
#                 # Find necessary columns
#                 snar_code_idx = header.index('SNAR_CODE') if 'SNAR_CODE' in header else None
#                 fid_1_idx = header.index('fid_1') if 'fid_1' in header else None
#                 area_m2_idx = header.index('area_m2') if 'area_m2' in header else None
                
#                 if snar_code_idx is None or fid_1_idx is None or area_m2_idx is None:
#                     logging.error("Error: Required columns missing in CSV. Found:", header)
#                     raise ValueError("CSV missing required columns")
                
#                 for row in csv_reader:
#                     if len(row) <= max(snar_code_idx, fid_1_idx, area_m2_idx):
#                         continue
#                     snar_code = row[snar_code_idx]
#                     try:
#                         fid_1 = int(float(row[fid_1_idx]))
#                         area_m2 = float(row[area_m2_idx])
#                         field_data.append({
#                             'SNAR_CODE': snar_code,
#                             'fid_1': fid_1,
#                             'area_m2': area_m2
#                         })
#                     except (ValueError, IndexError) as e:
#                         logging.warning(f"Warning: skip row: {row}, Error: {e}")
#                         continue
            
#             # Group by SNAR_CODE and calculate total area
#             snar_code_areas = {}
#             for entry in field_data:
#                 sn_code = entry['SNAR_CODE']
#                 area = entry['area_m2']
#                 snar_code_areas[sn_code] = snar_code_areas.get(sn_code, 0) + area
            
#             area_summary = [
#                 {'SNAR_CODE': sn_code, 'total_area': total_area}
#                 for sn_code, total_area in snar_code_areas.items()
#             ]
            
#             # For each crop type, select a portion of field_ids for training based on area
#             train_fids = []
#             for area_entry in area_summary:
#                 sn_code = area_entry['SNAR_CODE']
#                 total_area = area_entry['total_area']
#                 target_area = total_area * training_ratio
                
#                 # Find all fields with the current SNAR_CODE and sort by area
#                 rows_sncode = [x for x in field_data if x['SNAR_CODE'] == sn_code]
#                 rows_sncode.sort(key=lambda x: x['area_m2'])
                
#                 selected_fids = []
#                 selected_area_sum = 0
#                 for row in rows_sncode:
#                     if selected_area_sum < target_area:
#                         selected_fids.append(row['fid_1'])
#                         selected_area_sum += row['area_m2']
#                     else:
#                         break
#                 train_fids.extend(selected_fids)
            
#             train_fids = set(train_fids)
            
#             # Split remaining field IDs into validation and test sets
#             all_fields = set(entry['fid_1'] for entry in field_data)
#             remaining = list(all_fields - train_fids)
#             np.random.shuffle(remaining)
#             val_count = int(len(remaining) * val_test_split_ratio)
#             val_fids = set(remaining[:val_count])
#             test_fids = set(remaining[val_count:])
            
#             # Allocate data based on field ID
#             train_positions = []
#             val_positions = []
#             test_positions = []
            
#             for pos, fid in field_id_dict.items():
#                 if fid in train_fids:
#                     train_positions.append(pos)
#                 elif fid in val_fids:
#                     val_positions.append(pos)
#                 elif fid in test_fids:
#                     test_positions.append(pos)
            
#             # Check if there's enough for training/validation
#             if len(train_positions) > 0 and len(test_positions) > 0:
#                 # Create training and test datasets
#                 X_train = np.array([embeddings_dict[pos] for pos in train_positions])
#                 y_train = np.array([labels_dict[pos] for pos in train_positions])
#                 X_test = np.array([embeddings_dict[pos] for pos in test_positions])
#                 y_test = np.array([labels_dict[pos] for pos in test_positions])
                
#                 logging.info(f"Field-based split: Train: {len(train_positions)}, Val: {len(val_positions)}, Test: {len(test_positions)}")
#                 field_split_success = True
#             else:
#                 logging.warning("Warning: Field-based splitting yielded empty training or test set.")
#                 field_split_success = False
                
#         except Exception as e:
#             logging.error(f"Error during field-based splitting: {e}")
#             import traceback
#             logging.error(traceback.format_exc())
#             field_split_success = False
    
#     # If field splitting failed, use random splitting
#     if not field_split_success:
#         logging.info("Using random splitting as fallback...")
#         positions = list(embeddings_dict.keys())
#         np.random.shuffle(positions)
        
#         embeddings = np.array([embeddings_dict[pos] for pos in positions])
#         labels_arr = np.array([labels_dict[pos] for pos in positions])
        
#         # Randomly select training_ratio% of data for training
#         N = len(embeddings)
#         split = int(training_ratio * N)
#         X_train = embeddings[:split]
#         y_train = labels_arr[:split]
#         X_test = embeddings[split:]
#         y_test = labels_arr[split:]
        
#         logging.info(f"Random split: Train: {len(X_train)}, Test: {len(X_test)}")
    
#     # Check data validity
#     if len(X_train) == 0 or len(y_train) == 0 or len(X_test) == 0 or len(y_test) == 0:
#         logging.error("Error: Empty training or test set after splitting")
#         return 0.0, 0.0, np.zeros((1, 1)), "Error: Empty dataset"

#     if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
#         logging.warning("Warning: Not enough classes in training or test set")
#         logging.warning(f"Classes in train: {np.unique(y_train)}")
#         logging.warning(f"Classes in test: {np.unique(y_test)}")
#         return 0.0, 0.0, np.zeros((1, 1)), "Error: Not enough classes"
    
#     # Choose classifier
#     if classifier_type.lower() == 'rf':
#         clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4)  # Limit jobs to prevent hang
#     else:  # Default to LR
#         clf = LogisticRegression(max_iter=100000, n_jobs=4)  # Limit jobs to prevent hang
    
#     # Train classifier
#     logging.info(f"Training {classifier_type} classifier with {len(X_train)} samples...")
#     clf.fit(X_train, y_train)
#     pred = clf.predict(X_test)
    
#     # Calculate evaluation metrics
#     acc = accuracy_score(y_test, pred)
#     f1 = f1_score(y_test, pred, average='weighted')
#     cm = confusion_matrix(y_test, pred)
#     cr = classification_report(y_test, pred, digits=4)
    
#     logging.info(f"Evaluation complete: acc={acc:.4f}, f1={f1:.4f}")
    
#     return acc, f1, cm, cr


def linear_probe_evaluate(model, val_loader, field_ids=None,
                          field_data_path=None, 
                          training_ratio=0.3, val_test_split_ratio=0.5, # val_test_split_ratio was 0.5, user has 1/7
                          classifier_type='lr', num_inference=1, device='cuda',
                          apply_amp=False,
                          # QAT related parameters for evaluation logic
                          apply_qat_representation_for_eval=False,
                          qat_representation_bits_for_eval=8):
    # ... (Initial setup: imports, random seeds, model.eval()) ...
    # (Copied relevant parts of your existing function)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    import csv
    import os
    from contextlib import nullcontext
    import gc
    import time
    import logging
    
    np.random.seed(42) # You had 3407 in main, but 42 here. Sticking to 42 for local consistency.
    torch.manual_seed(42)
    
    logging.info(f"Starting linear probe evaluation. Apply AMP: {apply_amp}, Eval QAT Repr: {apply_qat_representation_for_eval}")
    
    model.eval() # Ensure model is in evaluation mode

    # Logging embedding weights from eval_model if needed (your existing code)
    # logging.info(f"Embedding weight during evaluation: {model.s2_backbone.embedding[0].weight}")
    # logging.info(f"Embedding bias during evaluation: {model.s2_backbone.embedding[0].bias}")
    
    embeddings_dict = {}
    labels_dict = {}
    field_id_dict = {} # If field_ids are used
    
    amp_ctx = torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", enabled=apply_amp) if apply_amp else nullcontext()
    
    torch.cuda.empty_cache()
    gc.collect()
    batch_timeout = 120

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            try:
                # ... (Your existing batch data extraction logic) ...
                if isinstance(batch_data, dict):
                    batch_s2 = batch_data.get('s2', None); batch_s1 = batch_data.get('s1', None)
                    batch_label = batch_data.get('label', None); batch_field_id = batch_data.get('field_id', None)
                    batch_pos = batch_data.get('pos', None)
                elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 3:
                    s2_sample, s1_sample, label = batch_data[:3]
                    batch_s2, batch_s1, batch_label = s2_sample, s1_sample, label
                    batch_field_id = batch_data[3] if len(batch_data) > 3 else None
                    batch_pos = batch_data[4] if len(batch_data) > 4 else None # Assuming pos is 5th if field_id is 4th
                else:
                    logging.error(f"Unexpected batch format: {type(batch_data)}"); continue
                
                if batch_s2 is None or batch_s1 is None or batch_label is None or batch_pos is None:
                    logging.warning(f"Skipping batch {batch_idx} - missing required data (s2, s1, label, or pos)"); continue

                batch_s2 = batch_s2.to(device, non_blocking=True)
                batch_s1 = batch_s1.to(device, non_blocking=True)
                # batch_label is already cpu().numpy() if from your dataset, or needs conversion
                if isinstance(batch_label, torch.Tensor): batch_label = batch_label.cpu().numpy()
                if isinstance(batch_field_id, torch.Tensor): batch_field_id = batch_field_id.cpu().numpy()
                if isinstance(batch_pos, torch.Tensor): batch_pos = batch_pos.cpu().numpy()

                # --- Get f32 representation ---
                # The eval_model is created with return_repr=True.
                # It returns (projected_feats, fused_representation_f32)
                # We need fused_representation_f32 for linear probing.
                
                current_repr_f32_batch_sum = None
                for _ in range(num_inference): # Your num_inference loop
                    with amp_ctx:
                         # model here is the eval_model loaded from checkpoint
                        _, repr_f32_single_inference = model(batch_s2, batch_s1) # Get the 128-dim f32 output
                    
                    if current_repr_f32_batch_sum is None:
                        current_repr_f32_batch_sum = repr_f32_single_inference.cpu() # Move to CPU after inference
                    else:
                        current_repr_f32_batch_sum += repr_f32_single_inference.cpu()
                
                avg_repr_f32_batch = current_repr_f32_batch_sum / num_inference

                # --- Actual Quantization for Evaluation ---
                # The classifier will be trained on these (potentially dequantized) representations.
                # The key is that the information content reflects int8.
                if apply_qat_representation_for_eval:
                    # Quantize the f32 batch representations to int8, then dequantize for the sklearn classifier.
                    # quantize_tensor_symmetric returns (quantized_int_tensor, scale_tensor)
                    # Note: scale is per-tensor (i.e., per-batch here).
                    repr_int_batch, scale_batch = quantize_tensor_symmetric(avg_repr_f32_batch, bits=qat_representation_bits_for_eval)
                    # Dequantize for the classifier (sklearn typically expects float inputs)
                    repr_for_classifier_batch = dequantize_tensor_symmetric(repr_int_batch, scale_batch)
                    repr_for_classifier_batch_np = repr_for_classifier_batch.numpy()
                else:
                    # If not evaluating QAT representations, use the original f32 representations
                    repr_for_classifier_batch_np = avg_repr_f32_batch.numpy()


                # Store results
                for i in range(batch_s2.size(0)): # Iterate through samples in the batch
                    position = tuple(batch_pos[i]) # Assuming batch_pos is (N, D_pos)
                    embeddings_dict[position] = repr_for_classifier_batch_np[i]
                    labels_dict[position] = batch_label[i]
                    if batch_field_id is not None and field_ids is not None : # Ensure field_id exists for this sample
                         # Check if batch_field_id is an array/list and has entry for i
                        if isinstance(batch_field_id, (np.ndarray, list)) and i < len(batch_field_id):
                            field_id_dict[position] = batch_field_id[i]
                        elif isinstance(batch_field_id, (int, np.integer)): # If it's a scalar, but that'd be odd for a batch
                            field_id_dict[position] = batch_field_id 


                if batch_idx % 5 == 0: torch.cuda.empty_cache()
                if batch_idx % 10 == 0: logging.info(f"Processed {batch_idx+1}/{len(val_loader)} validation batches for embeddings.")

            except Exception as e:
                logging.error(f"Error processing validation batch {batch_idx}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                torch.cuda.empty_cache()
                continue
    
    logging.info(f"Collected embeddings for {len(embeddings_dict)} positions for linear probing.")
    # ... (Rest of your linear_probe_evaluate function: field-based splitting, classifier training, metrics)
    # Ensure that X_train, y_train, X_test, y_test are correctly populated from embeddings_dict and labels_dict.

    # Field-based splitting logic (from your existing code)
    field_split_success = False
    if field_data_path and os.path.exists(field_data_path) and field_ids is not None and field_id_dict:
        # ... (Your field splitting logic, ensure it uses embeddings_dict, labels_dict correctly) ...
        # This part seems complex and specific to your data, I'm assuming it's mostly correct
        # but ensure that `train_positions`, `val_positions`, `test_positions` correctly index `embeddings_dict`
        # and `labels_dict`.
        try:
            # Load field data
            field_data_rows = []
            with open(field_data_path, 'r') as f:
                csv_reader = csv.reader(f); header = next(csv_reader)
                snar_code_idx = header.index('SNAR_CODE'); fid_1_idx = header.index('fid_1'); area_m2_idx = header.index('area_m2')
                for row in csv_reader:
                    if len(row) <= max(snar_code_idx, fid_1_idx, area_m2_idx): continue
                    try: field_data_rows.append({'SNAR_CODE': row[snar_code_idx], 'fid_1': int(float(row[fid_1_idx])), 'area_m2': float(row[area_m2_idx])})
                    except (ValueError, IndexError) as e: logging.warning(f"Skipping CSV row: {row}, Error: {e}"); continue
            
            snar_code_areas = {}; [snar_code_areas.update({r['SNAR_CODE']: snar_code_areas.get(r['SNAR_CODE'], 0) + r['area_m2']}) for r in field_data_rows]
            area_summary = [{'SNAR_CODE': sc, 'total_area': ta} for sc, ta in snar_code_areas.items()]
            
            train_fids_set = set()
            for area_entry in area_summary:
                sc = area_entry['SNAR_CODE']; target_area = area_entry['total_area'] * training_ratio
                rows_sncode = sorted([r for r in field_data_rows if r['SNAR_CODE'] == sc], key=lambda x: x['area_m2'])
                current_selected_area = 0
                for r_sc in rows_sncode:
                    if current_selected_area < target_area: train_fids_set.add(r_sc['fid_1']); current_selected_area += r_sc['area_m2']
                    else: break
            
            all_csv_fids = set(r['fid_1'] for r in field_data_rows)
            remaining_fids = list(all_csv_fids - train_fids_set)
            np.random.shuffle(remaining_fids)
            
            # val_test_split_ratio here is ratio of (val_set / (val_set + test_set))
            # If your val_test_split_ratio means val_set = ratio * remaining, and test is (1-ratio)*remaining
            # Original code had: val_count = int(len(remaining) * val_test_split_ratio) -> this implies val_test_split_ratio is val / total_remaining
            # User prompt: "val_test_split_ratio=1/7" - is this val size relative to test size, or val size relative to (val+test)
            # Assuming it means val is 1 part, test is 6 parts, so val is 1/7 of remaining_for_val_test
            # If val_test_split_ratio means val_set_size / test_set_size e.g. 1/6, then val_count = len(remaining) / 7
            # If it means val_set_size / (val_set_size + test_set_size) e.g. 1/7, then val_count = len(remaining) * (1/7)
            # Your code had `val_test_split_ratio=0.5` previously.
            # Let's assume val_test_split_ratio means "proportion of remaining that goes to validation"
            val_count = int(len(remaining_fids) * val_test_split_ratio)


            val_fids_set = set(remaining_fids[:val_count])
            test_fids_set = set(remaining_fids[val_count:])

            train_positions, val_positions, test_positions = [], [], []
            # Filter positions based on whether their field_id is in the respective set AND in embeddings_dict
            for pos_key, fid_val in field_id_dict.items():
                if pos_key not in embeddings_dict: continue # Ensure embedding exists for this position
                if fid_val in train_fids_set: train_positions.append(pos_key)
                elif fid_val in val_fids_set: val_positions.append(pos_key) # You might not use val_positions if test_positions is your test set
                elif fid_val in test_fids_set: test_positions.append(pos_key)
            
            if train_positions and test_positions:
                X_train = np.array([embeddings_dict[pos] for pos in train_positions])
                y_train = np.array([labels_dict[pos] for pos in train_positions])
                X_test = np.array([embeddings_dict[pos] for pos in test_positions])
                y_test = np.array([labels_dict[pos] for pos in test_positions])
                logging.info(f"Field-based split: Train: {len(X_train)}, Test: {len(X_test)} (Val not used by classifier: {len(val_positions)})")
                field_split_success = True
            else:
                logging.warning("Field-based splitting yielded empty train or test set from available positions.")
                field_split_success = False
        except Exception as e:
            logging.error(f"Error during field-based splitting: {e}", exc_info=True); field_split_success = False

    if not field_split_success:
        logging.info("Using random splitting as fallback for linear probe...")
        if not embeddings_dict:
             logging.error("No embeddings collected, cannot perform random split.")
             return 0.0, 0.0, np.array([[]]), "Error: No embeddings for linear probe."

        all_positions = list(embeddings_dict.keys())
        np.random.shuffle(all_positions)
        
        # training_ratio here refers to the split of the *available evaluation data*
        # not the global field area ratio.
        num_total_eval_samples = len(all_positions)
        split_idx = int(training_ratio * num_total_eval_samples) 
        
        train_positions_rand = all_positions[:split_idx]
        test_positions_rand = all_positions[split_idx:]

        if not train_positions_rand or not test_positions_rand:
            logging.error("Random splitting yielded empty train or test set.")
            return 0.0, 0.0, np.array([[]]), "Error: Random split empty."

        X_train = np.array([embeddings_dict[pos] for pos in train_positions_rand])
        y_train = np.array([labels_dict[pos] for pos in train_positions_rand])
        X_test = np.array([embeddings_dict[pos] for pos in test_positions_rand])
        y_test = np.array([labels_dict[pos] for pos in test_positions_rand])
        logging.info(f"Random split: Train: {len(X_train)}, Test: {len(X_test)}")

    if len(X_train) == 0 or len(X_test) == 0:
        logging.error("Training or test set is empty before fitting classifier.")
        return 0.0, 0.0, np.array([[]]), "Error: Empty train/test set."
    if len(np.unique(y_train)) < 2 or (len(y_test) > 0 and len(np.unique(y_test)) < 2 and len(np.unique(y_train)) != len(np.unique(y_test)) ): # allow test to have fewer if some classes not sampled
        logging.warning(f"Not enough classes in training ({np.unique(y_train)}) or test ({np.unique(y_test)}). Min 2 classes needed for most classifiers.")
        # If y_test is empty or has only one class, classification_report might fail.
        # Return default bad values.
        if len(y_test) == 0 or len(np.unique(y_test)) < 1 : # if test is empty
             return 0.0,0.0, np.zeros((len(np.unique(y_train)), len(np.unique(y_train)))), "Test set empty or no classes"
        # If y_test has one class, but train has more, some metrics might be ill-defined.
        # For now, let it proceed, sklearn might handle some cases.

    clf_jobs = max(1, os.cpu_count() // 8 if os.cpu_count() else 1) # Be conservative with jobs for LR/RF
    if classifier_type.lower() == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=clf_jobs)
    else: # Default to Logistic Regression
        clf = LogisticRegression(max_iter=100000, solver='saga', n_jobs=clf_jobs, C=0.1) # Added solver and C for potential convergence
    
    logging.info(f"Training {classifier_type} classifier with {len(X_train)} samples...")
    try:
        clf.fit(X_train, y_train)
        if len(X_test) > 0:
            pred = clf.predict(X_test)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
            # Ensure all classes in y_train are potentially in y_test for cm labels
            unique_labels_train = np.unique(y_train)
            unique_labels_test = np.unique(y_test)
            all_unique_labels = np.unique(np.concatenate((unique_labels_train, unique_labels_test)))
            
            cm = confusion_matrix(y_test, pred, labels=all_unique_labels)
            cr = classification_report(y_test, pred, digits=4, zero_division=0, labels=all_unique_labels, target_names=[str(l) for l in all_unique_labels])
        else: # No test samples
            acc, f1 = 0.0, 0.0
            num_classes = len(np.unique(y_train))
            cm = np.zeros((num_classes, num_classes), dtype=int)
            cr = "No test samples to evaluate."
            logging.warning("Test set for linear probe is empty.")

    except Exception as e:
        logging.error(f"Error during classifier training or prediction: {e}", exc_info=True)
        num_classes_train = len(np.unique(y_train)) if len(y_train) > 0 else 1
        return 0.0, 0.0, np.zeros((num_classes_train, num_classes_train)), f"Error in classifier: {e}"

    logging.info(f"Linear probe evaluation complete: Accuracy={acc:.4f}, F1 Score={f1:.4f}")
    return acc, f1, cm, cr


def linear_probe_evaluate_64_fixed(
    model,
    val_loader,
    field_ids=None,
    field_data_path=None,
    training_ratio=0.3,
    val_test_split_ratio=0.5,
    classifier_type='lr',
    num_inference=1,
    device='cuda'
):
    """
    Evaluate model using linear probe with attention masks for 64 timesteps data structure.

    Args:
        model: Trained (possibly compiled/FSDP/DDP) model
        val_loader: Validation DataLoader
        field_ids: Optional, for field-based splitting
        field_data_path: Optional, CSV file path for field data
        training_ratio: Proportion of field area for training, default 0.3
        val_test_split_ratio: Validation/test split ratio, default 0.5
        classifier_type: Classifier type, 'lr' or 'rf', default 'lr'
        num_inference: Number of inference passes to average, default 1
        device: Computation device, default 'cuda'

    Returns:
        (acc, f1, cm, cr):
            - accuracy (float)
            - F1 score (float)
            - confusion matrix (numpy array)
            - classification report (string)
    """

    # ---- 1) 提取原始模型以避免使用编译 (torch.compile) 版本 ----
    # 如果你用的是 DDP，可以先获取 DDP 的内部模型:
    if isinstance(model, DDP):
        eval_model = model.module
    else:
        eval_model = model

    # 如果模型被 torch.compile(...) 包装过，则会有 _orig_mod
    if hasattr(eval_model, '_orig_mod'):
        eval_model = eval_model._orig_mod

    # 如果你使用了 FSDP，也可以类似地判断并获取 eval_model.module
    # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    # if isinstance(eval_model, FSDP):
    #     eval_model = eval_model.module
    #
    # 同理如果还嵌套了 compile，需要再次检查 _orig_mod，可视情况处理

    eval_model.eval()

    # ---- 2) 收集验证集样本的 embedding ----
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize containers to store embeddings and labels
    embeddings_dict = {}  # {position: embedding}
    labels_dict = {}      # {position: label}
    field_id_dict = {}    # {position: field_id}

    with torch.no_grad():
        for batch_data in val_loader:
            # Unpack data (including masks)
            s2_sample, s1_sample, label, field_id, pos, s2_mask, s1_mask = batch_data

            batch_s2 = s2_sample.to(device)
            batch_s1 = s1_sample.to(device)
            batch_label = label.cpu().numpy()
            batch_s2_mask = s2_mask.to(device)
            batch_s1_mask = s1_mask.to(device)

            if isinstance(field_id, torch.Tensor):
                batch_field_id = field_id.cpu().numpy()

            batch_pos = pos.cpu().numpy()

            # 多次 forward，取平均 embedding
            embeddings_sum = None
            for _ in range(num_inference):
                out = eval_model(batch_s2, batch_s1, batch_s2_mask, batch_s1_mask)
                # 若模型 forward 返回 (z, repr) 这种 tuple，就取第二项 repr
                if isinstance(out, tuple):
                    out = out[1]  # representation

                emb_this = out.cpu().numpy()
                if embeddings_sum is None:
                    embeddings_sum = emb_this
                else:
                    embeddings_sum += emb_this
            
            emb = embeddings_sum / num_inference

            # 存储到字典
            for i in range(len(batch_pos)):
                position = tuple(batch_pos[i])
                embeddings_dict[position] = emb[i]
                labels_dict[position] = batch_label[i]
                if isinstance(field_id, torch.Tensor):
                    field_id_dict[position] = batch_field_id[i]
        print(f"Embedding shape: {emb_this.shape}")

    print(f"Collected embeddings for {len(embeddings_dict)} positions")
    if field_id_dict:
        print(f"Number of unique field IDs: {len(set(field_id_dict.values()))}")
    else:
        print("No field IDs found (field_id_dict is empty)")

    # ---- 3) 如果指定了 field_data_path，尝试基于地块做拆分，否则随机拆分 ----
    field_split_success = False

    if field_data_path is not None and os.path.exists(field_data_path) and field_ids is not None:
        try:
            # Load CSV field data
            field_data = []
            with open(field_data_path, 'r') as f:
                csv_reader = csv.reader(f)
                header = next(csv_reader)
                print(f"CSV headers: {header}")

                # 找到必要列 (SNAR_CODE, fid_1, area_m2)
                snar_code_idx = header.index('SNAR_CODE') if 'SNAR_CODE' in header else None
                fid_1_idx = header.index('fid_1') if 'fid_1' in header else None
                area_m2_idx = header.index('area_m2') if 'area_m2' in header else None

                if snar_code_idx is None or fid_1_idx is None or area_m2_idx is None:
                    print("Error: Required columns missing in CSV. Found:", header)
                    raise ValueError("CSV missing required columns")

                for row in csv_reader:
                    if len(row) <= max(snar_code_idx, fid_1_idx, area_m2_idx):
                        continue
                    snar_code = row[snar_code_idx]
                    try:
                        fid_1 = int(float(row[fid_1_idx]))
                        area_m2 = float(row[area_m2_idx])
                        field_data.append({
                            'SNAR_CODE': snar_code,
                            'fid_1': fid_1,
                            'area_m2': area_m2
                        })
                    except (ValueError, IndexError) as e:
                        print(f"Warning: skip row: {row}, Error: {e}")
                        continue

            print(f"Loaded {len(field_data)} entries from field data CSV")

            csv_field_ids = set(entry['fid_1'] for entry in field_data)
            dataset_field_ids = set(field_id_dict.values()) if field_id_dict else set()
            common_field_ids = csv_field_ids.intersection(dataset_field_ids)

            print(f"Field IDs in CSV: {len(csv_field_ids)}")
            print(f"Field IDs in dataset: {len(dataset_field_ids)}")
            print(f"Common field IDs: {len(common_field_ids)}")

            if len(common_field_ids) == 0:
                raise ValueError("No common field IDs between CSV and dataset")

            # 统计每种 SNAR_CODE 的总面积
            snar_code_areas = {}
            for entry in field_data:
                sn_code = entry['SNAR_CODE']
                area = entry['area_m2']
                snar_code_areas[sn_code] = snar_code_areas.get(sn_code, 0) + area

            area_summary = [
                {'SNAR_CODE': sn_code, 'total_area': total_area}
                for sn_code, total_area in snar_code_areas.items()
            ]
            print(f"Found {len(area_summary)} unique crop types (SNAR_CODE)")

            # 逐个作物类型，按面积选择一部分 field_id 进入训练集
            train_fids = []
            for area_entry in area_summary:
                sn_code = area_entry['SNAR_CODE']
                total_area = area_entry['total_area']
                target_area = total_area * training_ratio

                rows_sncode = [x for x in field_data if x['SNAR_CODE'] == sn_code]
                rows_sncode.sort(key=lambda x: x['area_m2'])

                selected_fids = []
                selected_area_sum = 0
                for row in rows_sncode:
                    if selected_area_sum < target_area:
                        selected_fids.append(row['fid_1'])
                        selected_area_sum += row['area_m2']
                    else:
                        break
                train_fids.extend(selected_fids)

            train_fids = set(train_fids)
            print(f"Selected {len(train_fids)} field IDs for training")

            # 剩余 ID 再分成 val/test
            all_fields = set(x['fid_1'] for x in field_data)
            remaining = list(all_fields - train_fids)
            np.random.shuffle(remaining)
            val_count = int(len(remaining) * val_test_split_ratio)
            val_fids = set(remaining[:val_count])
            test_fids = set(remaining[val_count:])

            # 根据 fid 分配 position
            train_positions = []
            val_positions = []
            test_positions = []

            for pos, fid in field_id_dict.items():
                if fid in train_fids:
                    train_positions.append(pos)
                elif fid in val_fids:
                    val_positions.append(pos)
                elif fid in test_fids:
                    test_positions.append(pos)

            print(f"Field-based split: Train={len(train_positions)}, Val={len(val_positions)}, Test={len(test_positions)}")

            # 判断是否足够做训练/验证
            if len(train_positions) > 0 and (len(val_positions) > 0 or len(test_positions) > 0):
                X_train = np.array([embeddings_dict[pos] for pos in train_positions])
                y_train = np.array([labels_dict[pos] for pos in train_positions])
                # 验证集直接抛弃即可
                X_test = np.array([embeddings_dict[pos] for pos in test_positions])
                y_test = np.array([labels_dict[pos] for pos in test_positions])

                field_split_success = True
            else:
                print("Warning: Field-based splitting yielded empty training or test set.")
                field_split_success = False

        except Exception as e:
            print(f"Error during field-based splitting: {e}")
            field_split_success = False

    # ---- 4) 如果字段拆分不成功，就随机拆分 ----
    if not field_split_success:
        print("Using random splitting as fallback...")
        positions = list(embeddings_dict.keys())
        np.random.shuffle(positions)
        if len(positions) == 0:
            raise ValueError("No valid positions found in dataset")

        embeddings = np.array([embeddings_dict[pos] for pos in positions])
        labels_arr = np.array([labels_dict[pos] for pos in positions])

        N = len(embeddings)
        split = int(training_ratio * N)

        X_train = embeddings[:split]
        y_train = labels_arr[:split]
        X_test = embeddings[split:]
        y_test = labels_arr[split:]

        print(f"Random split: Train={len(X_train)}, Test={len(X_test)}")

    # ---- 5) 检查数据合法性 ----
    if len(X_train) == 0 or len(y_train) == 0 or len(X_test) == 0 or len(y_test) == 0:
        print("Error: Empty training or test set after splitting")
        return 0.0, 0.0, np.zeros((1, 1)), "Error: Empty dataset"

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("Warning: Not enough classes in training or test set")
        print(f"Classes in train: {np.unique(y_train)}")
        print(f"Classes in test: {np.unique(y_test)}")
        return 0.0, 0.0, np.zeros((1, 1)), "Error: Not enough classes"

    # ---- 6) 训练线性分类器 (RF 或 LR) ----
    if classifier_type.lower() == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=10000, n_jobs=-1)

    print(f"Training {classifier_type} classifier with {len(X_train)} samples...")
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred, digits=4)

    return acc, f1, cm, cr



