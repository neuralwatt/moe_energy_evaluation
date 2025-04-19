# moe_energy_evaluation

An experiment to evaluate the energy efficiency (currently measured via inference speed) of Mixture of Experts (MoE) models compared to standard Multi-Layer Perceptron (MLP) architectures using PyTorch.

## Overview

This project implements and compares different neural network architectures:

1.  **Baseline MLP (`baseline_mlp.py`):** A standard, configurable Multi-Layer Perceptron serving as a non-MoE baseline.
2.  **Original MoE (`moe_original.py`):** A basic implementation of the Mixture of Experts architecture. It uses a gating network to route inputs to a subset of "expert" networks (which are themselves MLPs defined in `expert.py`). This version iterates through all experts to check which inputs belong to them.
3.  **Optimized MoE (`moe_optimized.py`):** An optimized version of the MoE model. It aims to improve performance by:
    *   Sorting tokens based on their assigned expert to process them in contiguous chunks.
    *   Iterating only over the experts that are actually selected by the gating network in a given batch.
    *   Utilizing `torch.compile` for potential graph optimization and kernel fusion.

The goal is to measure and compare the inference speed and potentially the energy consumption of these different approaches under various configurations.

## Key Components

*   **`expert.py`:** Defines the `Expert` class, a simple MLP used as the individual expert network within the MoE models.
*   **`baseline_mlp.py`:** Defines the `BaselineMLP` class.
*   **`moe_original.py`:** Defines the `MoE` class (basic implementation).
*   **`moe_optimized.py`:** Defines the `OptimizedMoE` class.
*   **`utils.py`:** Contains helper functions for:
    *   Generating synthetic complex classification data (`generate_complex_data`).
    *   Calculating the number of parameters in an MLP (`calculate_params`).
    *   Printing a model summary (`print_model_summary`).
    *   Timing model inference over a set duration (`time_inference`).

## Requirements

*   Python 3.x
*   PyTorch (`torch`)

Install requirements using:
```bash
pip install -r requirements.txt
```

## Scaling Analysis: MoE vs Baseline Efficiency

To make the MoE architecture more computationally efficient than a baseline model with the same parameter count, the following key scaling factors are analyzed:

### Parameter Efficiency Analysis

#### Current MoE Implementation
- With top-k=2 out of n experts, theoretically only ~(2/n) of parameters are active per inference.
- The original inefficiencies (data duplication, sequential processing) negate this advantage.
- The optimized implementation removes these inefficiencies.

### Required Scaling for Efficiency

#### Expert Count Scaling:
- For a top-k=2 MoE, a minimum of n=8 experts is needed for meaningful efficiency.
- At n=8, only 25% of parameters are active per input.

#### Expert Size Scaling:
- Each expert needs to be large enough that routing overhead becomes negligible.
- Minimum expert size: ~500K parameters each.

#### Total Parameter Threshold:
- MoE advantage becomes significant when total parameters exceed 4-5 million.
- Below this threshold, routing overhead typically negates sparsity benefits.

### Efficiency Break-Even Formula

For MoE to be more efficient than a dense model with the same parameter count:

```
(k/n) + routing_overhead < 1
```

Where:
- `k` = number of active experts per input (2 in your implementation).
- `n` = total number of experts.
- `routing_overhead` ≈ 5-10% of computation for a well-implemented MoE.

### Real-World Recommendation

To make your MoE implementation more efficient than the baseline:

1. Increase to **8-16 experts** minimum.
2. Scale each expert to **500K-1M parameters**.
3. Use the optimized implementation that eliminates unnecessary data duplication.
4. Consider expert pruning techniques to improve load balancing.

At these scales, the MoE should be approximately 2-3× more computationally efficient than a baseline model with an equivalent parameter count.

