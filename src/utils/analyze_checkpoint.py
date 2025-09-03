#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from collections import defaultdict, OrderedDict
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns

def format_bytes(bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def format_number(num):
    """Format large numbers with commas"""
    return f"{num:,}"

def clean_key(key):
    """Remove FSDP prefixes from parameter keys"""
    if key.startswith('_fsdp_wrapped_module.'):
        key = key[len('_fsdp_wrapped_module.'):]
    elif key.startswith('_orig_mod.'):
        key = key[len('_orig_mod.'):]
    return key

def get_tensor_stats(tensor):
    """Get statistics for a tensor"""
    tensor_np = tensor.detach().cpu().numpy().flatten()
    stats = {
        'mean': float(np.mean(tensor_np)),
        'std': float(np.std(tensor_np)),
        'min': float(np.min(tensor_np)),
        'max': float(np.max(tensor_np)),
        'abs_mean': float(np.mean(np.abs(tensor_np))),
        'sparsity': float(np.sum(tensor_np == 0) / len(tensor_np))
    }
    return stats

def plot_parameter_distribution(state_dict, output_dir):
    """Plot parameter distributions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all parameters
    all_params = []
    param_groups = defaultdict(list)
    
    for key, tensor in state_dict.items():
        clean_k = clean_key(key)
        param_type = clean_k.split('.')[-1]
        
        values = tensor.detach().cpu().numpy().flatten()
        all_params.extend(values)
        param_groups[param_type].extend(values)
    
    # Plot overall distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_params, bins=100, alpha=0.7, edgecolor='black')
    plt.xlabel('Parameter Value')
    plt.ylabel('Count')
    plt.title('Overall Parameter Distribution')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    for param_type, values in param_groups.items():
        if len(values) > 1000:  # Only plot if enough samples
            plt.hist(values, bins=50, alpha=0.5, label=param_type, density=True)
    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
    plt.title('Parameter Distribution by Type')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_distributions.png'))
    plt.close()
    
    # Plot parameter norms by layer
    layer_norms = defaultdict(list)
    for key, tensor in state_dict.items():
        clean_k = clean_key(key)
        layer_name = '.'.join(clean_k.split('.')[:-1])
        if layer_name:
            norm = torch.norm(tensor).item()
            layer_norms[layer_name].append((clean_k.split('.')[-1], norm))
    
    # Create norm plot
    if layer_norms:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        layer_names = []
        weight_norms = []
        bias_norms = []
        
        for layer, norms in sorted(layer_norms.items()):
            for param_type, norm in norms:
                if 'weight' in param_type:
                    layer_names.append(layer)
                    weight_norms.append(norm)
                    bias_norms.append(None)
                elif 'bias' in param_type:
                    # Find matching weight entry
                    if layer_names and layer_names[-1] == layer:
                        bias_norms[-1] = norm
                    else:
                        layer_names.append(layer)
                        weight_norms.append(None)
                        bias_norms.append(norm)
        
        x = np.arange(len(layer_names))
        width = 0.35
        
        # Filter out None values for plotting
        weight_x = [i for i, v in enumerate(weight_norms) if v is not None]
        weight_y = [v for v in weight_norms if v is not None]
        bias_x = [i for i, v in enumerate(bias_norms) if v is not None]
        bias_y = [v for v in bias_norms if v is not None]
        
        ax.bar([x - width/2 for x in weight_x], weight_y, width, label='Weight norms')
        ax.bar([x + width/2 for x in bias_x], bias_y, width, label='Bias norms')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('L2 Norm')
        ax.set_title('Parameter Norms by Layer')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=90, ha='right')
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_norms.png'))
        plt.close()

def analyze_checkpoint_advanced(checkpoint_path, args):
    """Advanced analysis of checkpoint file"""
    print(f"\n{'='*80}")
    print(f"Advanced Checkpoint Analysis")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Basic info
    print("1. CHECKPOINT METADATA")
    print("-" * 40)
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"  {key}: {len(checkpoint[key])} tensors")
        elif key == 'config':
            print(f"  {key}: {type(checkpoint[key]).__name__}")
            if args.show_config and isinstance(checkpoint[key], dict):
                print("    Config entries:")
                for k, v in sorted(checkpoint[key].items())[:20]:
                    print(f"      - {k}: {v}")
                if len(checkpoint[key]) > 20:
                    print(f"      ... and {len(checkpoint[key]) - 20} more entries")
        else:
            print(f"  {key}: {checkpoint[key]}")
    print()
    
    if 'model_state_dict' not in checkpoint:
        print("ERROR: No 'model_state_dict' found!")
        return
    
    state_dict = checkpoint['model_state_dict']
    
    # Clean keys
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        cleaned_state_dict[clean_key(k)] = v
    
    # 2. Model Architecture Summary
    print("2. MODEL ARCHITECTURE SUMMARY")
    print("-" * 40)
    
    # Group by top-level module
    modules = defaultdict(lambda: {'params': 0, 'size': 0, 'tensors': 0})
    for key, tensor in cleaned_state_dict.items():
        top_module = key.split('.')[0]
        modules[top_module]['params'] += tensor.numel()
        modules[top_module]['size'] += tensor.numel() * tensor.element_size()
        modules[top_module]['tensors'] += 1
    
    total_params = sum(m['params'] for m in modules.values())
    total_size = sum(m['size'] for m in modules.values())
    
    print(f"  Total Parameters: {format_number(total_params)}")
    print(f"  Total Size: {format_bytes(total_size)}")
    print(f"  Total Tensors: {len(state_dict)}")
    print()
    
    print("  Top-level Modules:")
    for name, info in sorted(modules.items(), key=lambda x: x[1]['params'], reverse=True):
        pct = (info['params'] / total_params * 100) if total_params > 0 else 0
        print(f"    {name}:")
        print(f"      Parameters: {format_number(info['params'])} ({pct:.1f}%)")
        print(f"      Size: {format_bytes(info['size'])}")
        print(f"      Tensors: {info['tensors']}")
    print()
    
    # 3. Parameter Statistics
    print("3. PARAMETER STATISTICS")
    print("-" * 40)
    
    param_stats = defaultdict(lambda: {
        'count': 0, 'total_params': 0, 'mean_abs': [], 'sparsity': []
    })
    
    for key, tensor in cleaned_state_dict.items():
        param_type = key.split('.')[-1]
        stats = get_tensor_stats(tensor)
        
        param_stats[param_type]['count'] += 1
        param_stats[param_type]['total_params'] += tensor.numel()
        param_stats[param_type]['mean_abs'].append(stats['abs_mean'])
        param_stats[param_type]['sparsity'].append(stats['sparsity'])
    
    for ptype, stats in sorted(param_stats.items()):
        print(f"  {ptype}:")
        print(f"    Count: {stats['count']}")
        print(f"    Total params: {format_number(stats['total_params'])}")
        print(f"    Avg absolute value: {np.mean(stats['mean_abs']):.6f}")
        print(f"    Avg sparsity: {np.mean(stats['sparsity']):.2%}")
    print()
    
    # 4. Layer-wise Analysis
    print("4. LAYER-WISE ANALYSIS")
    print("-" * 40)
    
    # Group by layer
    layers = defaultdict(list)
    for key, tensor in cleaned_state_dict.items():
        parts = key.split('.')
        if len(parts) > 1:
            layer_name = '.'.join(parts[:-1])
            layers[layer_name].append((parts[-1], tensor))
    
    # Analyze each layer
    layer_info = []
    for layer_name, params in sorted(layers.items()):
        layer_params = sum(t.numel() for _, t in params)
        layer_size = sum(t.numel() * t.element_size() for _, t in params)
        
        layer_info.append({
            'name': layer_name,
            'params': layer_params,
            'size': layer_size,
            'tensors': len(params)
        })
    
    # Sort by parameter count
    layer_info.sort(key=lambda x: x['params'], reverse=True)
    
    # Show top 15 layers
    print("  Top 15 layers by parameter count:")
    for i, info in enumerate(layer_info[:15]):
        print(f"    {i+1}. {info['name']}:")
        print(f"       Parameters: {format_number(info['params'])}")
        print(f"       Size: {format_bytes(info['size'])}")
        print(f"       Tensors: {info['tensors']}")
    
    if len(layer_info) > 15:
        print(f"    ... and {len(layer_info) - 15} more layers")
    print()
    
    # 5. Shape Analysis
    print("5. SHAPE ANALYSIS")
    print("-" * 40)
    
    shape_categories = {
        '1D (bias/norm)': [],
        '2D (linear/embedding)': [],
        '3D': [],
        '4D (conv)': [],
        '5D+': []
    }
    
    for key, tensor in cleaned_state_dict.items():
        ndim = len(tensor.shape)
        if ndim == 1:
            shape_categories['1D (bias/norm)'].append((key, tensor.shape))
        elif ndim == 2:
            shape_categories['2D (linear/embedding)'].append((key, tensor.shape))
        elif ndim == 3:
            shape_categories['3D'].append((key, tensor.shape))
        elif ndim == 4:
            shape_categories['4D (conv)'].append((key, tensor.shape))
        else:
            shape_categories['5D+'].append((key, tensor.shape))
    
    for category, tensors in shape_categories.items():
        if tensors:
            print(f"  {category}: {len(tensors)} tensors")
            # Show first 3 examples
            for i, (name, shape) in enumerate(tensors[:3]):
                print(f"    - {name}: {list(shape)}")
            if len(tensors) > 3:
                print(f"    ... and {len(tensors) - 3} more")
    print()
    
    # Generate plots if requested
    if args.plot:
        print("6. GENERATING VISUALIZATIONS")
        print("-" * 40)
        output_dir = os.path.dirname(checkpoint_path)
        plot_dir = os.path.join(output_dir, 'checkpoint_analysis')
        plot_parameter_distribution(state_dict, plot_dir)
        print(f"  Plots saved to: {plot_dir}")
        print()
    
    # Export detailed analysis if requested
    if args.export_detailed:
        detailed_output = {
            'checkpoint_path': checkpoint_path,
            'total_parameters': total_params,
            'total_size_bytes': total_size,
            'modules': dict(modules),
            'layers': layer_info,
            'parameter_stats': dict(param_stats),
            'shape_distribution': {k: len(v) for k, v in shape_categories.items()}
        }
        
        json_path = checkpoint_path.replace('.pt', '_detailed_analysis.json')
        with open(json_path, 'w') as f:
            json.dump(detailed_output, f, indent=2)
        print(f"Detailed analysis exported to: {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Advanced PyTorch checkpoint analysis")
    parser.add_argument('--checkpoint_path', type=str, default='/mnt/e/Codes/btfm4rs/checkpoints/ssl/checkpoint_20250608_220648.pt',
                       help="Path to the checkpoint file (.pt)")
    parser.add_argument('--show-config', action='store_true',
                       help="Show configuration details")
    parser.add_argument('--plot', action='store_true',
                       help="Generate visualization plots")
    parser.add_argument('--export-detailed', action='store_true',
                       help="Export detailed analysis to JSON")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Checkpoint file not found: {args.checkpoint_path}")
        return
    
    try:
        analyze_checkpoint_advanced(args.checkpoint_path, args)
    except Exception as e:
        print(f"ERROR: Failed to analyze checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()