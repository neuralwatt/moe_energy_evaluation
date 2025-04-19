import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
import time
import warnings
try:
    import pynvml
    pynvml_found = True
except ImportError:
    pynvml_found = False
    warnings.warn("pynvml not found. GPU power monitoring will be disabled. Run `pip install pynvml` to enable.")

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
    total_energy_joules = 0.0
    nvml_handle = None
    last_time_point = start_time

    print(f"Starting inference timing for {duration_seconds} seconds...")
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # Initialize NVML and get handle if using CUDA and pynvml is available
    if device.type == 'cuda' and pynvml_found:
        try:
            pynvml.nvmlInit()
            # Assuming device index 0, adjust if necessary
            gpu_index = torch.cuda.current_device()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            print(f"NVML initialized for GPU {gpu_index}.")
        except pynvml.NVMLError as error:
            print(f"Failed to initialize NVML or get device handle: {error}")
            nvml_handle = None # Ensure handle is None if init fails
        except Exception as e:
            print(f"An unexpected error occurred during NVML initialization: {e}")
            nvml_handle = None

    try:
        # Use a single batch for consistent timing
        sample_batch, _ = next(iter(dataset))
        sample_batch = sample_batch.to(device)
        if sample_batch.shape[0] == 0:
             print("Warning: Dataset loader returned an empty batch. Cannot perform timing.")
             # Shutdown NVML if it was initialized
             if nvml_handle is not None and pynvml_found:
                 try:
                     pynvml.nvmlShutdown()
                 except pynvml.NVMLError as error:
                     print(f"Failed to shutdown NVML: {error}")
             return 0, 0.0, 0.0
    except StopIteration:
        print("Warning: Dataset is empty. Cannot perform timing.")
        # Shutdown NVML if it was initialized
        if nvml_handle is not None and pynvml_found:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as error:
                print(f"Failed to shutdown NVML: {error}")
        return 0, 0.0, 0.0

    with torch.no_grad():
        current_time = time.time()
        while current_time < end_time:
            if device.type == 'cuda': torch.cuda.synchronize()
            iter_start_time = time.time() # Time before inference

            _ = model(sample_batch) # Perform inference

            if device.type == 'cuda': torch.cuda.synchronize()
            iter_end_time = time.time() # Time after inference

            inference_count += sample_batch.shape[0] # Count samples processed

            # Measure power and calculate energy if NVML is active
            if nvml_handle is not None:
                try:
                    power_milliwatts = pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
                    time_interval = iter_end_time - last_time_point # Use time since last measurement
                    energy_joules = (power_milliwatts / 1000.0) * time_interval # Energy = Power (W) * Time (s)
                    total_energy_joules += energy_joules
                    last_time_point = iter_end_time # Update last time point
                except pynvml.NVMLError as error:
                    # Handle error, maybe stop monitoring or just print a warning
                    print(f"Warning: Failed to get power usage: {error}. Disabling further power monitoring.")
                    nvml_handle = None # Stop trying to read power
                except Exception as e:
                    print(f"An unexpected error occurred during power measurement: {e}")
                    nvml_handle = None

            current_time = iter_end_time # Update current time based on iteration end

    actual_end_time = time.time()
    actual_duration = actual_end_time - start_time

    # Shutdown NVML if it was initialized
    if pynvml_found and 'nvmlShutdown' in dir(pynvml):
        try:
            # Check if nvml was actually initialized before shutting down
            # A bit hacky, relies on nvmlInit setting some internal state
            # A more robust way might involve a flag set during successful init
            pynvml.nvmlDeviceGetCount() # Check if NVML is still active
            pynvml.nvmlShutdown()
            print("NVML shut down.")
        except pynvml.NVMLError as error:
            if "NVML_ERROR_UNINITIALIZED" not in str(error):
                 print(f"Failed to shutdown NVML: {error}")
        except Exception as e:
             print(f"An unexpected error occurred during NVML shutdown: {e}")


    print(f"End Time:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(actual_end_time))}")
    print(f"Actual Duration: {actual_duration:.4f} seconds")
    print(f"Total Inferences: {inference_count}")
    if actual_duration > 0:
        print(f"Inferences per Second: {inference_count / actual_duration:.2f}")
        if total_energy_joules > 0:
            print(f"Total Energy Consumed: {total_energy_joules:.2f} Joules")
            print(f"Average Power: {total_energy_joules / actual_duration:.2f} Watts")
        else:
            print("Energy consumption monitoring was not active or recorded zero.")
    else:
        print("Inferences per Second: N/A")
        print("Energy Consumption: N/A")

    return inference_count, actual_duration, total_energy_joules
