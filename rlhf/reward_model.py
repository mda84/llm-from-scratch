# reward_model.py - Reward Model for RLHF

import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, depth=6, heads=12, ff_dim=2048, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=ff_dim,
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)
        ])

        self.final_ln = nn.LayerNorm(embed_dim)
        self.reward_head = nn.Linear(embed_dim, 1)

    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        x = self.token_embed(input_ids) + self.pos_embed[:, :input_ids.size(1)]
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        # Mean pooling over sequence
        x = x.mean(dim=1)
        reward = self.reward_head(x).squeeze(-1)  # [batch]
        return reward

if __name__ == "__main__":
    model = RewardModel(vocab_size=50257)
    x = torch.randint(0, 50257, (4, 128))
    r = model(x)
    print("Reward scores:", r.shape)  # [4]