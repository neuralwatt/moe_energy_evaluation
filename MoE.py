import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math
import time

# Import components from other files
from moe_optimized2 import OptimizedMoE2
from utils import (
    generate_complex_data,
    calculate_params,
    print_model_summary,
    time_inference
)
from baseline_mlp import BaselineMLP
from moe_original import MoE as MoE_Original # Rename to avoid conflict
from moe_optimized import OptimizedMoE

# --- Configuration Switch ---
# Set to True for large model configuration that favors MoE efficiency
USE_LARGE_CONFIG = True 

# --- Model Size Configurations ---
SMALL_CONFIG = {
    "NUM_EXPERTS": 8,
    "EXPERT_HIDDEN_DIM": 256,
    "EXPERT_NUM_LAYERS": 3,
    "TOP_K_EXPERTS": 2,
}

LARGE_CONFIG = {
    "NUM_EXPERTS": 16,
    "EXPERT_HIDDEN_DIM": 1024,
    "EXPERT_NUM_LAYERS": 4,
    "TOP_K_EXPERTS": 4,
}

XLARGE_CONFIG = {
    "NUM_EXPERTS": 20,
    "EXPERT_HIDDEN_DIM": 2048,
    "EXPERT_NUM_LAYERS": 5,
    "TOP_K_EXPERTS": 5,
}

# This configuration can't be loaded with the other models due to memory constraints
# XXLARGE_CONFIG = {
#     "NUM_EXPERTS": 32,
#     "EXPERT_HIDDEN_DIM": 4096,
#     "EXPERT_NUM_LAYERS": 6,
#     "TOP_K_EXPERTS": 8,
# }

# Use selected configuration
active_config = LARGE_CONFIG if USE_LARGE_CONFIG else SMALL_CONFIG


# --- Configuration ---
NUM_SAMPLES = 10000
BATCH_SIZE = 128
INPUT_DIM = 784
OUTPUT_DIM = 10
NUM_EPOCHS = 15
INFERENCE_DURATION = 30 # Seconds

# --- Apply Selected Model Architecture Config ---
NUM_EXPERTS = active_config["NUM_EXPERTS"]
EXPERT_HIDDEN_DIM = active_config["EXPERT_HIDDEN_DIM"]
EXPERT_NUM_LAYERS = active_config["EXPERT_NUM_LAYERS"]
TOP_K_EXPERTS = active_config["TOP_K_EXPERTS"]

BASELINE_NUM_LAYERS = EXPERT_NUM_LAYERS # Use same number of layers as experts
# --- End Configuration ---

print(f"Active configuration: {'LARGE' if USE_LARGE_CONFIG else 'SMALL'}")
print(f"Model parameters: Experts={NUM_EXPERTS}, Hidden Dim={EXPERT_HIDDEN_DIM}, "
      f"Layers={EXPERT_NUM_LAYERS}, Top-K={TOP_K_EXPERTS}")

# --- Synthetic Dataset Creation ---
X_synthetic, Y_synthetic = generate_complex_data(NUM_SAMPLES, INPUT_DIM, OUTPUT_DIM)
synthetic_dataset = TensorDataset(X_synthetic, Y_synthetic)
dataset = DataLoader(synthetic_dataset, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# --- End Synthetic Dataset Creation ---


# --- Model Initialization ---
model_orig = MoE_Original(input_dim=INPUT_DIM,
                         output_dim=OUTPUT_DIM,
                         num_experts=NUM_EXPERTS,
                         expert_hidden_dim=EXPERT_HIDDEN_DIM,
                         expert_num_layers=EXPERT_NUM_LAYERS,
                         top_k=TOP_K_EXPERTS).to(device)
criterion_orig = nn.CrossEntropyLoss()
optimizer_orig = optim.Adam(model_orig.parameters(), lr=0.001)
print("\n--- Original MoE Model ---")
moe_orig_total_params = print_model_summary(model_orig, f"Original MoE (Top-{TOP_K_EXPERTS})")


model_opt = OptimizedMoE2(input_dim=INPUT_DIM,
                         output_dim=OUTPUT_DIM,
                         num_experts=NUM_EXPERTS,
                         expert_hidden_dim=EXPERT_HIDDEN_DIM,
                         expert_num_layers=EXPERT_NUM_LAYERS,
                         top_k=TOP_K_EXPERTS).to(device)
criterion_opt = nn.CrossEntropyLoss()
optimizer_opt = optim.Adam(model_opt.parameters(), lr=0.001)
print("\n--- Optimized MoE Model ---")
moe_opt_total_params = print_model_summary(model_opt, f"Optimized MoE (Top-{TOP_K_EXPERTS})")

# Ensure parameter counts match (they should, init is identical)
assert moe_orig_total_params == moe_opt_total_params, "Parameter counts differ between MoE implementations!"
moe_total_params = moe_orig_total_params # Use one value for baseline calculation
# --- End Model Initialization ---


# --- Train the Models ---
# Train Original MoE
print(f"\nTraining Original MoE model for {NUM_EPOCHS} epochs...")
model_orig.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0; correct_predictions = 0; total_samples = 0
    for x, y in dataset:
        x, y = x.to(device), y.to(device)
        optimizer_orig.zero_grad()
        output = model_orig(x)
        loss = criterion_orig(output, y)
        loss.backward()
        optimizer_orig.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_samples += y.size(0)
        correct_predictions += (predicted == y).sum().item()
    accuracy = 100 * correct_predictions / total_samples
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/len(dataset):.4f}, Accuracy: {accuracy:.2f}%")

# Train Optimized MoE
print(f"\nTraining Optimized MoE model for {NUM_EPOCHS} epochs...")
model_opt.train()
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0; correct_predictions = 0; total_samples = 0
    for x, y in dataset:
        x, y = x.to(device), y.to(device)
        optimizer_opt.zero_grad()
        output = model_opt(x)
        loss = criterion_opt(output, y)
        loss.backward()
        optimizer_opt.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_samples += y.size(0)
        correct_predictions += (predicted == y).sum().item()
    accuracy = 100 * correct_predictions / total_samples
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/len(dataset):.4f}, Accuracy: {accuracy:.2f}%")
# --- End Training ---


# --- Analysis Function ---
def run_analysis(models_dict, dataset, device, moe_total_params):
    # models_dict: {'name': model_instance, ...}

    # --- Baseline Model Setup ---
    params_one_expert = calculate_params(INPUT_DIM, OUTPUT_DIM, EXPERT_HIDDEN_DIM, EXPERT_NUM_LAYERS)
    params_gating = INPUT_DIM * NUM_EXPERTS + NUM_EXPERTS
    target_baseline_params = NUM_EXPERTS * params_one_expert + params_gating
    print(f"\nTarget Baseline Params (1 Expert + Gating): {target_baseline_params}")
    print(f"Using Baseline Config: Num Layers={BASELINE_NUM_LAYERS}")
    a = BASELINE_NUM_LAYERS - 1
    b = INPUT_DIM + 1 + (BASELINE_NUM_LAYERS - 1) + OUTPUT_DIM
    c_term = OUTPUT_DIM - target_baseline_params
    if a == 0:
        baseline_hidden_dim = math.ceil(-c_term / b) if b != 0 else 1
    else:
        delta = b**2 - 4*a*c_term
        if delta >= 0:
            h1 = (-b + math.sqrt(delta)) / (2*a)
            h2 = (-b - math.sqrt(delta)) / (2*a)
            if h1 > 0: baseline_hidden_dim = math.ceil(h1)
            elif h2 > 0: baseline_hidden_dim = math.ceil(h2)
            else: baseline_hidden_dim = 1; print("Warning: Quadratic solver non-positive.")
        else: baseline_hidden_dim = 1; print("Warning: Quadratic solver no real solution.")
    print(f"Calculated Baseline Hidden Dimension (Approx): {baseline_hidden_dim}")
    baseline_model = BaselineMLP(INPUT_DIM, OUTPUT_DIM, baseline_hidden_dim, BASELINE_NUM_LAYERS).to(device)
    baseline_criterion = nn.CrossEntropyLoss()
    baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    baseline_total_params = print_model_summary(baseline_model, "Baseline Model")
    print(f"Target Baseline Params: {target_baseline_params}, Actual Baseline Params: {baseline_total_params}")

    # --- Train Baseline Model ---
    print(f"\nTraining Baseline model for {NUM_EPOCHS} epochs...")
    baseline_model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0; correct_predictions = 0; total_samples = 0
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            baseline_optimizer.zero_grad()
            output = baseline_model(x)
            loss = baseline_criterion(output, y)
            loss.backward()
            baseline_optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_samples += y.size(0)
            correct_predictions += (predicted == y).sum().item()
        accuracy = 100 * correct_predictions / total_samples
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/len(dataset):.4f}, Accuracy: {accuracy:.2f}%")

    # Add baseline model to the dictionary for timing and accuracy
    models_dict["Baseline"] = baseline_model

    # --- Inference Timing ---
    timing_results = {}
    energy_results = {}
    for name, model_instance in models_dict.items():
        print(f"\n--- {name} Inference Timing ---")
        inf_count, inf_time, total_energy = time_inference(model_instance, dataset, device, INFERENCE_DURATION)
        timing_results[name] = {'count': inf_count, 'time': inf_time}
        energy_results[name] = total_energy
        # Add total trainable parameter size
        total_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        print(f"{name} Total Trainable Parameters: {total_params}")# --- End Inference Timing ---


    # --- Final Accuracy Comparison ---
    print("\n--- Final Accuracy Comparison ---")
    accuracy_results = {}
    total = 0
    try:
        # Use full dataset for final accuracy
        full_dataset_loader = DataLoader(synthetic_dataset, batch_size=len(synthetic_dataset))
        sample_batch_x, sample_batch_y = next(iter(full_dataset_loader))
        sample_batch_x = sample_batch_x.to(device)
        sample_batch_y = sample_batch_y.to(device)
        total = sample_batch_y.size(0)

        with torch.no_grad():
            for name, model_instance in models_dict.items():
                model_instance.eval()
                outputs = model_instance(sample_batch_x)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == sample_batch_y).sum().item()
                accuracy = 100 * correct / total
                accuracy_results[name] = accuracy
                print(f"{name} Final Accuracy: {accuracy:.2f}%")

    except StopIteration:
        print("Could not load full dataset for final accuracy calculation.")
    except Exception as e:
        print(f"Error during final accuracy calculation: {e}")

    # --- End Accuracy Comparison ---

    return timing_results, accuracy_results, energy_results
# --- End Analysis Function ---


# --- Main Execution ---
if __name__ == "__main__":
    models_to_evaluate = {
        "Original MoE": model_orig,
        "Optimized MoE": model_opt
    }
    timing_results, accuracy_results, energy_results = run_analysis(models_to_evaluate, dataset, device, moe_total_params)

    print("\n--- Summary ---")
    print("Timing Results (Inferences per Second):")
    for name, results in timing_results.items():
        if results['time'] > 0:
            ips = results['count'] / results['time']
            print(f"- {name}: {ips:.2f}")
        else:
            print(f"- {name}: N/A")

    print("\nAccuracy Results:")
    for name, acc in accuracy_results.items():
        print(f"- {name}: {acc:.2f}%")

    print("\nEnergy Efficiency Results (Inferences per Joule):")
    for name, energy in energy_results.items():
        inference_count = timing_results[name]['count']
        if energy > 0 and inference_count > 0:
             inferences_per_joule = inference_count / energy
             print(f"- {name}: {inferences_per_joule:.2f} inferences/Joule")
        else:
             print(f"- {name}: N/A (Energy or Inference count is zero or unavailable)")
# --- End Main Execution ---