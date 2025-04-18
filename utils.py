import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
import time

# --- Function to Generate Complex Data ---
def generate_complex_data(num_samples, input_dim, output_dim):
    print("Generating complex synthetic data...")
    X = torch.randn(num_samples, input_dim)
    scores = torch.zeros(num_samples, output_dim)
    features_per_group = input_dim // output_dim
    max_features_per_group = features_per_group + (input_dim % output_dim > 0) * (input_dim % output_dim)
    torch.manual_seed(42)
    group_weights = torch.randn(output_dim, max_features_per_group)
    interaction_weights = torch.randn(input_dim, output_dim) * 0.1
    torch.manual_seed(time.time_ns() % (2**32 - 1))
    for j in range(output_dim):
        start_idx = j * features_per_group
        end_idx = (j + 1) * features_per_group if j < output_dim - 1 else input_dim
        group_features = X[:, start_idx:end_idx]
        current_group_size = group_features.shape[1]
        current_group_weights = group_weights[j, :current_group_size]
        group_score = torch.tanh(torch.einsum('bi,i->b', group_features, current_group_weights))
        interaction_score = torch.sigmoid(torch.einsum('bi,io->bo', X, interaction_weights)[:, j]) * 0.5
        scores[:, j] = group_score + interaction_score
    scores += torch.randn_like(scores) * 0.2
    Y = torch.argmax(scores, dim=1)
    print("Data generation complete.")
    return X, Y
# --- End Function ---

# --- Helper Function to Calculate Parameters ---
def calculate_params(input_dim, output_dim, hidden_dim, num_layers):
    """Calculates parameters for an MLP with num_layers hidden layers."""
    params = 0
    # Input layer
    params += input_dim * hidden_dim + hidden_dim
    # Hidden layers
    params += (num_layers - 1) * (hidden_dim * hidden_dim + hidden_dim)
    # Output layer
    params += hidden_dim * output_dim + output_dim
    return params
# --- End Helper Function ---

def print_model_summary(model, model_name):
    print(f" - {model_name} Architecture ---")
    print(model)
    total_params = 0
    print("--- Parameters ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            print(f"Layer: {name}, Parameters: {num_params}")
            total_params += num_params
    print(f"Total Trainable Parameters: {total_params}")
    print("-" * (len(model_name) + 24))
    return total_params

def time_inference(model, dataset, device, duration_seconds=30):
    model.eval()
    inference_count = 0
    start_time = time.time()
    end_time = start_time + duration_seconds
    print(f"Starting inference timing for {duration_seconds} seconds...")
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    try:
        # Use a single batch for consistent timing
        sample_batch, _ = next(iter(dataset))
        sample_batch = sample_batch.to(device)
        if sample_batch.shape[0] == 0:
             print("Warning: Dataset loader returned an empty batch. Cannot perform timing.")
             return 0, 0.0
    except StopIteration:
        print("Warning: Dataset is empty. Cannot perform timing.")
        return 0, 0.0

    with torch.no_grad():
        current_time = time.time()
        while current_time < end_time:
            if device.type == 'cuda': torch.cuda.synchronize()
            _ = model(sample_batch) # Perform inference
            if device.type == 'cuda': torch.cuda.synchronize()
            inference_count += sample_batch.shape[0] # Count samples processed
            current_time = time.time() # Update current time

    actual_end_time = time.time()
    actual_duration = actual_end_time - start_time
    print(f"End Time:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(actual_end_time))}")
    print(f"Actual Duration: {actual_duration:.4f} seconds")
    print(f"Total Inferences: {inference_count}")
    if actual_duration > 0: print(f"Inferences per Second: {inference_count / actual_duration:.2f}")
    else: print("Inferences per Second: N/A")
    return inference_count, actual_duration
