import torch
import os
import numpy as np
import csv
from typing import List
from support.peconfig import PEConfig

def prune_custom_similarity(x):
    return 1 - np.sqrt((1 - x) / 2)

def angular_similarity(x):
    return 1 - np.arccos(x) / np.pi

def cosine_similarity(x):
    return (x+1)/2

def cubic_similarity(x):
    return (2 + x + x**3)/4

def exponential_similarity(x):
    return (np.exp(x) - np.exp(-1)) / (np.exp(1) - np.exp(-1))

def cos_theta(x, y, eps=1e-8):
    """Calculate cosine similarity with robust handling of extreme values."""
    x_tensor = x.clone().flatten().float()
    y_tensor = y.clone().flatten().float()
    
    # Replace any infinite values with the maximum finite value
    if torch.isinf(x_tensor).any() or torch.isinf(y_tensor).any():
        x_max_finite = x_tensor[~torch.isinf(x_tensor)].max()
        x_min_finite = x_tensor[~torch.isinf(x_tensor)].min()
        y_max_finite = y_tensor[~torch.isinf(y_tensor)].max()
        y_min_finite = y_tensor[~torch.isinf(y_tensor)].min()
        
        x_tensor = torch.nan_to_num(x_tensor, nan=0.0, posinf=x_max_finite, neginf=x_min_finite)
        y_tensor = torch.nan_to_num(y_tensor, nan=0.0, posinf=y_max_finite, neginf=y_min_finite)
    
    x_norm = torch.norm(x_tensor) + eps
    y_norm = torch.norm(y_tensor) + eps
    
    x_normalized = x_tensor / x_norm
    y_normalized = y_tensor / y_norm
    
    cos_theta = torch.dot(x_normalized, y_normalized)
    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
    
    return cos_theta

def calculate_similarities(average_layers, num_layers):
    similarities_data = []
    
    for starting_layer in range(0, num_layers - 1):
        for num_layers_to_prune in range(1, num_layers - starting_layer):
            start = average_layers[starting_layer]
            end = average_layers[starting_layer + num_layers_to_prune]
            
            print(f"\nProcessing layers {starting_layer} to {starting_layer + num_layers_to_prune}")
            print(f"Start layer stats: min={start.min()}, max={start.max()}, mean={start.mean()}")
            print(f"End layer stats: min={end.min()}, max={end.max()}, mean={end.mean()}")
            
            try:
                x = cos_theta(start, end)
                
                if torch.isnan(x):
                    print(f"WARNING: NAN detected in cosine similarity calculation")
                    continue
                
                similarities_data.extend([
                    [starting_layer, num_layers_to_prune, float(cosine_similarity(x)), "cosine"],
                    [starting_layer, num_layers_to_prune, float(angular_similarity(x)), "angular"],
                    [starting_layer, num_layers_to_prune, float(prune_custom_similarity(x)), "custom"],
                    [starting_layer, num_layers_to_prune, float(cubic_similarity(x)), "cubic"],
                    [starting_layer, num_layers_to_prune, float(exponential_similarity(x)), "exp"],
                ])
            except Exception as e:
                print(f"Error processing layers {starting_layer} to {starting_layer + num_layers_to_prune}: {e}")
    
    return similarities_data

def load_and_average_tensors(config: PEConfig):
    """Load all tensor files matching the pattern and compute layer averages with infinity checking."""
    tensor_pattern = str(config.tensor_dir / f"{config.formatted_name}_{config.task}_batch_*.pt")
    tensor_files = list(config.tensor_dir.glob(f"{config.formatted_name}_{config.task}_batch_*.pt"))
    
    if not tensor_files:
        raise FileNotFoundError(f"No tensor files found matching pattern: {tensor_pattern}")
    
    print(f"Found {len(tensor_files)} tensor files to process")
    
    layer_sums = None
    total_examples = 0
    
    for file_path in tensor_files:
        data_dict = torch.load(file_path, weights_only=True)
        if total_examples % 100 == 0:
            print(f"Processing {file_path.name}")
        
        for example_key in data_dict:
            example_data = data_dict[example_key]
            
            if layer_sums is None:
                num_layers = len(example_data)
                layer_sums = [torch.zeros_like(next(iter(example_data.values()))) for _ in range(num_layers)]
            
            for layer_key, layer_tensor in example_data.items():
                layer_idx = int(layer_key.split('_')[1])
                if torch.isinf(layer_tensor).any():
                    print(f"WARNING: Infinite values found in {file_path}, layer {layer_idx}")
                    print(f"Number of inf values: {torch.isinf(layer_tensor).sum().item()}")
                    max_finite = layer_tensor[~torch.isinf(layer_tensor)].max()
                    layer_tensor = torch.nan_to_num(layer_tensor, nan=0.0, posinf=max_finite, neginf=-max_finite)
                
                layer_sums[layer_idx] += layer_tensor
            
            total_examples += 1
    
    average_layers = []
    for layer_idx, layer_sum in enumerate(layer_sums):
        layer_avg = layer_sum / total_examples
        if torch.isinf(layer_avg).any():
            print(f"WARNING: Layer {layer_idx} has infinite values after averaging")
            max_finite = layer_avg[~torch.isinf(layer_avg)].max()
            layer_avg = torch.nan_to_num(layer_avg, nan=0.0, posinf=max_finite, neginf=-max_finite)
        average_layers.append(layer_avg)
    
    average_layers = torch.stack(average_layers)
    
    for layer_idx in range(len(average_layers)):
        layer_tensor = average_layers[layer_idx]
        print(f"\nLayer {layer_idx} statistics:")
        print(f"Min: {layer_tensor.min().item()}")
        print(f"Max: {layer_tensor.max().item()}")
        print(f"Mean: {layer_tensor.mean().item()}")
        print(f"Std: {layer_tensor.std().item()}")
    
    return average_layers, total_examples

def save_similarities_csv(similarities_data: List[List], output_path: str) -> None:
    """Save similarity metrics to a CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['starting_layer', 'layers_to_prune', 'similarity', 'metric'])
        writer.writerows(similarities_data)

def save_outputs(config: PEConfig, average_layers, similarities_data):
    """Save the averaged layers and similarities data to files."""
    config.ensure_output_dirs()
    
    # Save mean layers tensor
    mean_layers_path = config.get_tensor_path("mean_layers")
    torch.save(average_layers, mean_layers_path)
    print(f"Saved mean layers to {mean_layers_path}")
    
    # Save similarities data
    similarities_path = config.get_tensor_path("similarities").with_suffix('.csv')
    save_similarities_csv(similarities_data, str(similarities_path))
    print(f"Saved similarities data to {similarities_path}")

def process_tensors(config: PEConfig):
    """Main processing function that orchestrates the tensor analysis."""
    # Load and average tensors
    average_layers, total_examples = load_and_average_tensors(config)
    
    # Calculate similarities
    num_layers = average_layers.shape[0]
    print(f"Number of layers being processed: {num_layers}")
    print(f"Total examples processed: {total_examples}")
    
    similarities_data = calculate_similarities(average_layers, num_layers)
    
    # Save results
    save_outputs(config, average_layers, similarities_data)

def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Calculate layer similarities from tensor files')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., llama3.21b)')
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g., boolq)')
    return parser.parse_args()

def main():
    """Program which takes raw model hidden state output and means it across all examples for analysis"""
    try:
        args = parse_args()
        config = PEConfig(args)
        process_tensors(config)
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()
