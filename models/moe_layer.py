# moe_layer.py - Mixture of Experts (MoE) Layer

import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKGate(nn.Module):
    def __init__(self, input_dim, num_experts, k=2):
        super().__init__()
        self.k = k
        self.w_gating = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # x: [batch, seq_len, dim]
        gate_logits = self.w_gating(x)  # [batch, seq_len, num_experts]
        topk_vals, topk_idx = torch.topk(gate_logits, self.k, dim=-1)
        topk_softmax = F.softmax(topk_vals, dim=-1)  # [batch, seq_len, k]
        return topk_softmax, topk_idx

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        self.gate = TopKGate(input_dim, num_experts, k)

    def forward(self, x):
        # x: [batch, seq_len, dim]
        bsz, seq_len, dim = x.size()
        weights, indices = self.gate(x)  # [bsz, seq, k], [bsz, seq, k]

        expert_outputs = []
        for i in range(self.k):
            idx = indices[:, :, i]  # [bsz, seq]
            one_hot = F.one_hot(idx, num_classes=self.num_experts).float()  # [bsz, seq, num_experts]
            expert_mask = one_hot.permute(2, 0, 1)  # [num_experts, bsz, seq]

            x_flat = x.view(-1, dim)  # [bsz*seq, dim]
            expert_output = torch.zeros_like(x)

            for e in range(self.num_experts):
                selected = expert_mask[e] > 0  # [bsz, seq]
                if selected.any():
                    selected_flat = selected.view(-1)
                    x_sel = x_flat[selected_flat]
                    y_sel = self.experts[e](x_sel)
                    expert_output.view(-1, dim)[selected_flat] += weights[:, :, i].view(-1)[selected_flat].unsqueeze(1) * y_sel

            expert_outputs.append(expert_output)

        return sum(expert_outputs)

if __name__ == "__main__":
    moe = MoELayer(input_dim=512, hidden_dim=2048, num_experts=4, k=2)
    dummy_input = torch.randn(2, 16, 512)
    out = moe(dummy_input)
    print("MoE output shape:", out.shape)  # [2, 16, 512]