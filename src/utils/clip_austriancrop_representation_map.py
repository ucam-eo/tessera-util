#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import random

# Define required labels globally for clarity
REQUIRED_LABELS = set(range(1, 18)) # Labels 1 to 17

def save_patches_to_disk(patches_list, base_dir, patch_type_name, label_to_color, patch_size_arg, C_dim):
    """
    Saves representation and label patches to disk, including PCA visualization.
    """
    os.makedirs(base_dir, exist_ok=True)
    print(f"Saving {len(patches_list)} patches for {patch_type_name} set...")
    for idx, (repr_patch, label_patch, _unique_labels) in enumerate(patches_list):
        patch_index = idx + 1

        # Save npy files
        repr_npy_path = os.path.join(base_dir, f"patch_{patch_index}.npy")
        label_npy_path = os.path.join(base_dir, f"label_{patch_index}.npy")
        np.save(repr_npy_path, repr_patch)
        np.save(label_npy_path, label_patch)

        # --- PCA and visualization for representation patch ---
        reshaped_repr = repr_patch.reshape(-1, C_dim)
        pca = PCA(n_components=3)
        
        # Ensure there are enough samples for PCA, and C_dim is appropriate
        if reshaped_repr.shape[0] >= 3 and C_dim >=3 : # PCA needs n_samples >= n_components
            try:
                repr_pca = pca.fit_transform(reshaped_repr)
                repr_pca = repr_pca.reshape(patch_size_arg, patch_size_arg, 3)
                min_val, max_val = repr_pca.min(), repr_pca.max()
                if max_val > min_val:
                    repr_norm = (repr_pca - min_val) / (max_val - min_val)
                else:
                    repr_norm = np.zeros_like(repr_pca) # All values are the same
            except ValueError as e: # Handle cases where PCA might fail
                print(f"PCA failed for {patch_type_name} patch {patch_index}: {e}. Saving as zeros.")
                repr_norm = np.zeros((patch_size_arg, patch_size_arg, 3))
        else: 
             repr_norm = np.zeros((patch_size_arg, patch_size_arg, 3))
             if C_dim < 3:
                print(f"Warning: Representation dimension C_dim={C_dim} < 3. PCA output will be limited/zeros for patch {patch_index} in {patch_type_name}.")
             elif reshaped_repr.shape[0] < 3:
                print(f"Warning: Not enough samples ({reshaped_repr.shape[0]}) for PCA (n_components=3) for patch {patch_index} in {patch_type_name}. Saving as zeros.")
             else: # Other case if any
                print(f"Warning: PCA not performed for patch {patch_index} in {patch_type_name} due to data shape. Saving as zeros.")


        repr_png_path = os.path.join(base_dir, f"patch_{patch_index}.png")
        plt.imsave(repr_png_path, repr_norm)

        # --- Color visualization for label patch ---
        label_vis = np.zeros((patch_size_arg, patch_size_arg, 3), dtype=np.uint8)
        for label_val, color in label_to_color.items():
            mask = (label_patch == label_val)
            label_vis[mask] = color
        
        label_png_path = os.path.join(base_dir, f"label_{patch_index}.png")
        plt.imsave(label_png_path, label_vis)
        
        if (patch_index % 200 == 0) or (patch_index == len(patches_list)): # Print progress less frequently
             print(f"  Saved {patch_type_name} patch {patch_index}/{len(patches_list)}")


def split_train_test_with_completeness(all_patches_info, train_ratio, required_labels_set, seed):
    """
    Splits patches into training and testing sets, ensuring both sets contain all required labels.
    all_patches_info: list of (repr_patch, label_patch, unique_labels_in_patch)
    """
    rng = random.Random(seed) # Use standard library random for list operations
    
    if not all_patches_info:
        raise ValueError("Cannot split patches: The list of all patches is empty.")

    # Indices refer to positions in the original all_patches_info list
    original_indices = list(range(len(all_patches_info)))
    rng.shuffle(original_indices) # Shuffle indices to process patches in random order initially

    train_set_selected_indices = []
    test_set_selected_indices = []
    
    assigned_patch_mask = [False] * len(all_patches_info)

    # --- Helper: Get unassigned patch original indices (those not yet assigned to train or test) ---
    def _get_unassigned_original_indices():
        return [i for i in original_indices if not assigned_patch_mask[i]]

    # --- Phase 1: Ensure completeness for TRAIN set ---
    print("Phase 1: Ensuring TRAIN set completeness...")
    current_train_labels = set()
    
    # Loop to ensure all required labels are covered in the training set
    temp_needed_for_train = required_labels_set.copy()
    while temp_needed_for_train - current_train_labels: # While there are still labels needed for train
        labels_to_acquire = temp_needed_for_train - current_train_labels
        label_to_find = rng.choice(list(labels_to_acquire)) 

        best_patch_original_idx = -1
        max_newly_covered_count = -1
        
        unassigned_indices_pool = _get_unassigned_original_indices()
        if not unassigned_indices_pool:
            raise ValueError(f"Cannot complete TRAIN set. Ran out of patches while needing labels: {labels_to_acquire}. Current train labels: {current_train_labels}")

        candidate_indices_for_label = [
            idx for idx in unassigned_indices_pool if label_to_find in all_patches_info[idx][2]
        ]
        
        if not candidate_indices_for_label:
            raise ValueError(f"Cannot complete TRAIN set. No unassigned patch found for label {label_to_find} (needed: {labels_to_acquire}). Insufficient patch diversity.")

        # Greedy selection: pick patch that covers the most currently needed labels for train
        for patch_idx in candidate_indices_for_label:
            patch_labels = all_patches_info[patch_idx][2]
            newly_covered_by_this_patch = len(labels_to_acquire.intersection(patch_labels))
            if newly_covered_by_this_patch > max_newly_covered_count:
                max_newly_covered_count = newly_covered_by_this_patch
                best_patch_original_idx = patch_idx
            elif newly_covered_by_this_patch == max_newly_covered_count and best_patch_original_idx != -1:
                # Tie-breaking: prefer patch with more total unique labels (non-zero)
                if len(patch_labels) > len(all_patches_info[best_patch_original_idx][2]):
                    best_patch_original_idx = patch_idx
        
        if best_patch_original_idx == -1: 
            raise ValueError(f"Cannot complete TRAIN set. Logic error finding best patch for label {label_to_find}.")

        train_set_selected_indices.append(best_patch_original_idx)
        assigned_patch_mask[best_patch_original_idx] = True
        current_train_labels.update(all_patches_info[best_patch_original_idx][2])
        print(f"  Added patch (orig_idx {best_patch_original_idx}) to TRAIN. Train labels now: {len(current_train_labels & required_labels_set)}/{len(required_labels_set)}")

    print(f"TRAIN set completeness achieved with {len(train_set_selected_indices)} patches.")

    # --- Phase 2: Ensure completeness for TEST set (from remaining unassigned) ---
    print("Phase 2: Ensuring TEST set completeness...")
    current_test_labels = set()
    temp_needed_for_test = required_labels_set.copy()
    while temp_needed_for_test - current_test_labels:
        labels_to_acquire = temp_needed_for_test - current_test_labels
        label_to_find = rng.choice(list(labels_to_acquire))

        best_patch_original_idx = -1
        max_newly_covered_count = -1
        unassigned_indices_pool = _get_unassigned_original_indices()
        if not unassigned_indices_pool:
            raise ValueError(f"Cannot complete TEST set. Ran out of patches while needing labels: {labels_to_acquire}. Current test labels: {current_test_labels}")

        candidate_indices_for_label = [
            idx for idx in unassigned_indices_pool if label_to_find in all_patches_info[idx][2]
        ]
        
        if not candidate_indices_for_label:
            raise ValueError(f"Cannot complete TEST set. No unassigned patch found for label {label_to_find} (needed: {labels_to_acquire}). Remaining patches lack diversity.")

        for patch_idx in candidate_indices_for_label:
            patch_labels = all_patches_info[patch_idx][2]
            newly_covered_by_this_patch = len(labels_to_acquire.intersection(patch_labels))
            if newly_covered_by_this_patch > max_newly_covered_count:
                max_newly_covered_count = newly_covered_by_this_patch
                best_patch_original_idx = patch_idx
            elif newly_covered_by_this_patch == max_newly_covered_count and best_patch_original_idx != -1:
                 if len(patch_labels) > len(all_patches_info[best_patch_original_idx][2]):
                    best_patch_original_idx = patch_idx
        
        if best_patch_original_idx == -1:
            raise ValueError(f"Cannot complete TEST set. Logic error finding best patch for label {label_to_find}.")

        test_set_selected_indices.append(best_patch_original_idx)
        assigned_patch_mask[best_patch_original_idx] = True
        current_test_labels.update(all_patches_info[best_patch_original_idx][2])
        print(f"  Added patch (orig_idx {best_patch_original_idx}) to TEST. Test labels now: {len(current_test_labels & required_labels_set)}/{len(required_labels_set)}")
    
    print(f"TEST set completeness achieved with {len(test_set_selected_indices)} patches.")

    # --- Phase 3: Distribute remaining patches based on ratio ---
    print("Phase 3: Distributing remaining patches according to ratio...")
    num_total_patches = len(all_patches_info)
    target_num_train = int(num_total_patches * train_ratio)
    
    unassigned_indices_pool = _get_unassigned_original_indices()
    rng.shuffle(unassigned_indices_pool) # Shuffle the order of remaining patches for random distribution

    # Add to train set until its target size is met, or run out of unassigned
    while len(train_set_selected_indices) < target_num_train and unassigned_indices_pool:
        patch_original_idx_to_add = unassigned_indices_pool.pop(0)
        train_set_selected_indices.append(patch_original_idx_to_add)
        assigned_patch_mask[patch_original_idx_to_add] = True 
    
    # Add all other remaining unassigned patches to the test set
    # Re-fetch unassigned_indices (which might have been modified if pop was used on a view)
    unassigned_indices_pool = _get_unassigned_original_indices() 
    for patch_original_idx in unassigned_indices_pool: # All truly remaining ones
        test_set_selected_indices.append(patch_original_idx)
        assigned_patch_mask[patch_original_idx] = True

    # Construct final sets from indices
    train_patches_final = [all_patches_info[i] for i in train_set_selected_indices]
    test_patches_final = [all_patches_info[i] for i in test_set_selected_indices]

    # --- Final Verification ---
    if not train_patches_final:
        # This case should ideally be caught by earlier checks if all_patches_info is empty
        raise ValueError("Train set is empty after splitting. Check data and training_ratio.")
    if not test_patches_final:
        raise ValueError("Test set is empty after splitting. Check data and training_ratio (e.g., not too close to 0 or 1).")

    train_labels_final_check = set.union(*(p[2] for p in train_patches_final))
    test_labels_final_check = set.union(*(p[2] for p in test_patches_final))

    if not required_labels_set.issubset(train_labels_final_check):
        missing = required_labels_set - train_labels_final_check
        raise ValueError(f"CRITICAL FAILURE: Train set IS NOT complete after distribution. Missing: {missing}. Train count: {len(train_patches_final)}")
    if not required_labels_set.issubset(test_labels_final_check):
        missing = required_labels_set - test_labels_final_check
        raise ValueError(f"CRITICAL FAILURE: Test set IS NOT complete after distribution. Missing: {missing}. Test count: {len(test_patches_final)}")

    print(f"Splitting complete. Train patches: {len(train_patches_final)}, Test patches: {len(test_patches_final)}")
    return train_patches_final, test_patches_final


def main(args):
    # Input file paths (ensure these exist or adjust as needed)
    # Using example paths from the original script.
    # For a real run, these might need to be configurable or checked.
    base_data_path = "data/downstream/austrian_crop/"
    repr_path = "/mnt/e/Codes/btfm4rs/data/representation/austrian_crop_mpc_pipeline_fsdp_20250604_100313.npy"
    label_path = os.path.join(base_data_path, "fieldtype_17classes.npy")

    if not (os.path.exists(repr_path) and os.path.exists(label_path)):
        print(f"Warning: Input data files not found at {repr_path} or {label_path}.")
        print("Please ensure the paths are correct or the data exists.")
        print("Attempting to proceed, but loading will likely fail.")
        # Or, raise FileNotFoundError here if preferred.
    
    # Load data
    print("Loading representation data...")
    representation = np.load(repr_path)  # shape: (H, W, C)
    print("Loading fieldtype data...")
    fieldtype = np.load(label_path)      # shape: (H, W)
    
    H, W, C_dim = representation.shape
    if fieldtype.shape[0] != H or fieldtype.shape[1] != W:
        raise ValueError("Representation and fieldtype dimensions do not match!")
    
    patch_size = args.patch_size
    
    # Unique labels and color map (globally for consistency)
    unique_labels_overall = np.unique(fieldtype)
    unique_labels_overall = np.sort(unique_labels_overall) # Ensure consistent color mapping
    print("All unique labels found in the entire dataset:", unique_labels_overall)
    
    # Create color map using enough colors for all unique labels found
    # Using tab20, which has 20 distinct colors. If more, consider a different cmap or cycling.
    num_distinct_colors_needed = len(unique_labels_overall)
    if num_distinct_colors_needed > 20:
        print(f"Warning: Number of unique labels ({num_distinct_colors_needed}) exceeds tab20 colormap size (20). Colors may repeat.")
    
    cmap = plt.get_cmap("tab20", max(20, num_distinct_colors_needed)) # Ensure cmap has enough colors
    label_to_color = {}
    for idx, label_val in enumerate(unique_labels_overall):
        # Normalize index for cmap correctly, especially if num_distinct_colors_needed is 1
        norm_idx = idx / (num_distinct_colors_needed - 1) if num_distinct_colors_needed > 1 else 0.0
        rgba = cmap(norm_idx) 
        rgb = tuple(int(255 * x) for x in rgba[:3])
        label_to_color[label_val] = rgb
    
    print("Label to color mapping established.")
        
    all_patches_info = [] # List to store (repr_patch, label_patch, unique_labels_in_patch)

    # --- Generate all candidate patches ---
    print("Generating candidate patches...")
    patch_generation_count = 0
    if args.max_overlap_ratio >= 0: # Sequential sampling
        print(f"Using sequential sampling mode, max_overlap_ratio = {args.max_overlap_ratio}")
        overlap_pixels = int(patch_size * args.max_overlap_ratio)
        step = patch_size - overlap_pixels
        if step < 1: step = 1
        print(f"patch_size={patch_size}, overlap_pixels={overlap_pixels}, step={step}")
        
        if H < patch_size or W < patch_size:
            print(f"Warning: Image dimensions ({H}x{W}) are smaller than patch_size ({patch_size}). Cannot extract patches in sequential mode.")
        else:
            for i in range(0, H - patch_size + 1, step):
                for j in range(0, W - patch_size + 1, step):
                    repr_patch = representation[i:i + patch_size, j:j + patch_size, :]
                    label_patch = fieldtype[i:i + patch_size, j:j + patch_size]
                    
                    if np.all(label_patch == 0): # Skip if label_patch is all zeros
                        continue
                    
                    unique_labels_in_patch = set(np.unique(label_patch)) - {0} # Exclude 0
                    if not unique_labels_in_patch: # Also skip if only label 0 was present
                        continue
                    all_patches_info.append((repr_patch, label_patch, unique_labels_in_patch))
                    patch_generation_count +=1
        print(f"Generated {patch_generation_count} valid patches via sequential sampling.")

    else: # Random sampling
        print(f"Using random sampling mode, num_patches_target = {args.num_patches}")
        rng_sampling = np.random.default_rng(args.seed)
        attempts = 0
        # Ensure enough attempts, especially if many patches are all zeros
        max_attempts = args.num_patches * 100 if args.num_patches > 0 else 1000 

        if H < patch_size or W < patch_size:
            print(f"Warning: Image dimensions ({H}x{W}) are smaller than patch_size ({patch_size}). Cannot extract patches in random mode.")
        else:
            while len(all_patches_info) < args.num_patches and attempts < max_attempts:
                attempts += 1
                i = rng_sampling.integers(0, H - patch_size + 1)
                j = rng_sampling.integers(0, W - patch_size + 1)
                
                repr_patch = representation[i:i + patch_size, j:j + patch_size, :]
                label_patch = fieldtype[i:i + patch_size, j:j + patch_size]

                if np.all(label_patch == 0):
                    continue
                
                unique_labels_in_patch = set(np.unique(label_patch)) - {0}
                if not unique_labels_in_patch: # Also skip if only label 0 was present
                    continue
                all_patches_info.append((repr_patch, label_patch, unique_labels_in_patch))
                patch_generation_count +=1
                if patch_generation_count % 200 == 0 and patch_generation_count > 0:
                     print(f"  Generated {patch_generation_count}/{args.num_patches} random valid patches...")
            
            if attempts >= max_attempts and len(all_patches_info) < args.num_patches:
                print(f"Warning: Max attempts ({max_attempts}) reached in random sampling. Generated {len(all_patches_info)}/{args.num_patches} patches.")
        print(f"Generated {len(all_patches_info)} valid patches via random sampling.")

    if not all_patches_info:
        raise ValueError("No valid (non-empty, non-zero label) patches were generated. Check input data, patch parameters, and label distribution.")

    # --- Check overall label availability in all generated patches ---
    overall_present_labels = set.union(*(p[2] for p in all_patches_info))
    if not REQUIRED_LABELS.issubset(overall_present_labels):
        missing = REQUIRED_LABELS - overall_present_labels
        raise ValueError(f"The entire dataset of {len(all_patches_info)} generated patches does not contain all required labels (1-17). Missing: {missing}")
    print(f"All required labels (1-17) are present in the pool of {len(all_patches_info)} generated patches.")

    # --- Split into training and test sets ---
    if not (0 < args.training_ratio < 1):
        raise ValueError("training_ratio must be between 0 and 1 (exclusive for meaningful split).")
        
    train_patches, test_patches = split_train_test_with_completeness(
        all_patches_info, 
        args.training_ratio, 
        REQUIRED_LABELS, 
        args.seed
    )
    
    # --- Save patches to disk ---
    # Ensure base output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果里面有内容先清除
    if os.path.exists(args.output_dir):
        for file in os.listdir(args.output_dir):
            file_path = os.path.join(args.output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
    print(f"Output directory {args.output_dir} is ready.")
    
    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, "test")

    # The save_patches_to_disk function will create train_dir and test_dir if they don't exist.
    save_patches_to_disk(train_patches, train_dir, "train", label_to_color, args.patch_size, C_dim)
    save_patches_to_disk(test_patches, test_dir, "test", label_to_color, args.patch_size, C_dim)

    print(f"\n✅ Processing complete.")
    print(f"Total valid patches generated: {len(all_patches_info)}")
    print(f"Training patches: {len(train_patches)}")
    print(f"Test patches: {len(test_patches)}")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crops representation and fieldtype patches, splits into train/test sets ensuring label completeness, and generates visualizations."
    )
    parser.add_argument("--num_patches", type=int, default=3000, 
                        help="Target number of patches to generate (only in random sampling mode).")
    parser.add_argument("--patch_size", type=int, default=32, 
                        help="Side length of the square patch.")
    parser.add_argument("--output_dir", type=str, default="data/downstream/austrian_crop_patch_split", 
                        help="Output directory for saved files (train/test subfolders will be created).")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility of sampling and splitting.")
    parser.add_argument("--max_overlap_ratio", type=float, default=0, 
                        help="Maximum overlap ratio (0 to <1) for sequential sampling. If >= 0, enables sequential mode, otherwise random.")
    parser.add_argument("--training_ratio", type=float, default=0.01, 
                        help="Proportion of patches for the training set (e.g., 0.7 for 70%% train). Must be >0 and <1.")
    
    args = parser.parse_args()
    main(args)