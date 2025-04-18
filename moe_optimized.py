import torch
import torch.nn as nn
import torch.nn.functional as F
from expert import Expert # Import the Expert class

@torch.compile
class OptimizedMoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, expert_hidden_dim=128, expert_num_layers=2, top_k=2):
        super(OptimizedMoE, self).__init__()
        # Identical initialization to original MoE
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.expert_num_layers = expert_num_layers
        self.output_dim = output_dim
        self.top_k = min(top_k, num_experts)
        self.experts = nn.ModuleList([Expert(input_dim, output_dim, expert_hidden_dim, expert_num_layers) for _ in range(num_experts)])
        self.gating = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        batch_size, _ = x.shape
        gating_logits = self.gating(x)
        top_k_gates, top_k_indices = torch.topk(gating_logits, self.top_k, dim=1)
        top_k_gates = F.softmax(top_k_gates, dim=1)

        # Flatten and create indices needed for dispatch and combine
        expert_idx_flat = top_k_indices.flatten() # Shape: (batch_size * top_k)
        gates_flat = top_k_gates.flatten()       # Shape: (batch_size * top_k)
        batch_idx = torch.arange(batch_size, device=x.device).repeat_interleave(self.top_k) # Shape: (batch_size * top_k)

        # Sort tokens based on the expert they are assigned to
        # This aims to group tokens for the same expert together
        sorted_expert_indices, sort_order = torch.sort(expert_idx_flat)
        # Apply the same sorting to batch index, gates, and gather the inputs accordingly
        sorted_batch_idx = batch_idx[sort_order]
        sorted_gates = gates_flat[sort_order]
        # Gather inputs based on the original batch index, then sort them
        # Avoids repeating the input tensor like in the original MoE's x_flat
        dispatched_x = x[sorted_batch_idx] # Shape: (batch_size * top_k, input_dim)

        # Find boundaries where expert assignments change in the sorted list
        expert_boundaries = torch.cat([
            torch.tensor([0], device=x.device),
            torch.where(sorted_expert_indices[1:] != sorted_expert_indices[:-1])[0] + 1,
            torch.tensor([batch_size * self.top_k], device=x.device)
        ])

        # Get the unique expert IDs present in this batch (already sorted)
        unique_expert_ids = sorted_expert_indices[expert_boundaries[:-1]]

        # Initialize output tensor for expert computations
        expert_outputs_sorted = torch.zeros(batch_size * self.top_k, self.output_dim, device=x.device)

        # Loop ONLY over the experts that were actually selected in this batch
        for i in range(len(unique_expert_ids)):
            expert_id = unique_expert_ids[i].item()
            start_idx = expert_boundaries[i].item()
            end_idx = expert_boundaries[i+1].item()

            # Select the chunk of inputs for the current expert
            expert_input_chunk = dispatched_x[start_idx:end_idx]

            # Compute expert output for this chunk
            expert_output_chunk = self.experts[expert_id](expert_input_chunk)

            # Place the output chunk into the corresponding position in the sorted output tensor
            expert_outputs_sorted[start_idx:end_idx] = expert_output_chunk

        # Unsort the expert outputs to match the original flattened order
        # Create an inverse sort order tensor
        inverse_sort_order = torch.empty_like(sort_order)
        inverse_sort_order[sort_order] = torch.arange(batch_size * self.top_k, device=x.device)
        expert_outputs_unsorted = expert_outputs_sorted[inverse_sort_order]

        # Weight the unsorted outputs by the original flattened gates
        weighted_outputs = expert_outputs_unsorted * gates_flat.unsqueeze(1)

        # Combine weighted outputs using scatter_add_ (same as before)
        final_output = torch.zeros(batch_size, self.output_dim, device=x.device)
        final_output.scatter_add_(0, batch_idx.unsqueeze(1).repeat(1, self.output_dim), weighted_outputs)

        return final_output
