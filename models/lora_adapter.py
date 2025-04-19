# lora_adapter.py - Low-Rank Adaptation (LoRA) Module

import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base = self.linear(x)
        lora = self.lora_B(self.dropout(self.lora_A(x))) * self.scale
        return base + lora

if __name__ == "__main__":
    import math
    lora_layer = LoRALinear(in_features=512, out_features=512, r=8, alpha=16)
    x = torch.randn(2, 10, 512)
    y = lora_layer(x)
    print("LoRA output shape:", y.shape)  # [2, 10, 512]