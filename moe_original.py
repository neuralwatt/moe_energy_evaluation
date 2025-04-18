import torch
import torch.nn as nn
import torch.nn.functional as F
from expert import Expert # Import the Expert class

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, expert_hidden_dim=128, expert_num_layers=2, top_k=2):
        super(MoE, self).__init__()
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
        final_output = torch.zeros(batch_size, self.output_dim, device=x.device)
        batch_idx = torch.arange(batch_size, device=x.device).repeat_interleave(self.top_k)
        expert_idx_flat = top_k_indices.flatten()
        gates_flat = top_k_gates.flatten()
        x_flat = x.repeat_interleave(self.top_k, dim=0) # This repeats data, could be inefficient
        expert_outputs_flat = torch.zeros(batch_size * self.top_k, self.output_dim, device=x.device)
        for i in range(self.num_experts): # Loop over ALL experts
            mask = (expert_idx_flat == i)
            if mask.any():
                expert_input = x_flat[mask] # Gather operation
                expert_outputs_flat[mask] = self.experts[i](expert_input) # Compute on subset
        weighted_outputs_flat = expert_outputs_flat * gates_flat.unsqueeze(1)
        final_output.scatter_add_(0, batch_idx.unsqueeze(1).repeat(1, self.output_dim), weighted_outputs_flat) # Scatter operation
        return final_output
