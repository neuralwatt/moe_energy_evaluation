import torch
import torch.nn as nn
import torch.nn.functional as F
from expert import Expert

class OptimizedMoE2(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, expert_hidden_dim=128, expert_num_layers=2, top_k=2):
        super(OptimizedMoE2, self).__init__()
        self.num_experts = num_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.expert_num_layers = expert_num_layers
        self.output_dim = output_dim
        self.top_k = min(top_k, num_experts)
        self.experts = nn.ModuleList([Expert(input_dim, output_dim, expert_hidden_dim, expert_num_layers) for _ in range(num_experts)])
        self.gating = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Get gating weights and indices for top-k experts
        gating_logits = self.gating(x)
        top_k_gates, top_k_indices = torch.topk(gating_logits, self.top_k, dim=1)
        top_k_gates = F.softmax(top_k_gates, dim=1)
        
        # Initialize the output tensor
        outputs = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        # Find unique experts selected in this batch
        unique_experts = torch.unique(top_k_indices)
        
        # Process only selected experts
        for expert_id in unique_experts:
            # Find all instances where this expert is selected in any top-k position
            expert_mask = (top_k_indices == expert_id)
            
            # Get corresponding batch indices and gate positions
            batch_indices, gate_positions = torch.nonzero(expert_mask, as_tuple=True)
            
            # Get expert inputs from original input
            expert_input = x[batch_indices]
            
            # Get corresponding gates
            expert_gates = top_k_gates[batch_indices, gate_positions].unsqueeze(1)
            
            # Forward pass through this expert
            expert_output = self.experts[expert_id](expert_input)
            
            # Apply gates to expert output
            weighted_output = expert_output * expert_gates
            
            # Add weighted output to the final outputs at the correct batch positions
            outputs.index_add_(0, batch_indices, weighted_output)
        
        return outputs