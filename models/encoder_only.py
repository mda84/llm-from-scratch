# encoder_only.py - Encoder-Only Transformer (e.g., BERT)

import torch
import torch.nn as nn
import math

class BertPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.ln2(x + self.dropout(ff_out))

class BertStyleEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, depth, heads, ff_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = BertPositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, heads, ff_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.mlm_head = nn.Linear(embed_dim, vocab_size)  # For masked language modeling

    def forward(self, x, attn_mask=None):
        x = self.token_embed(x)
        x = self.position_embed(x)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.norm(x)
        return self.mlm_head(x)

if __name__ == "__main__":
    model = BertStyleEncoder(vocab_size=30522, embed_dim=768, depth=12, heads=12, ff_dim=3072)
    dummy_input = torch.randint(0, 30522, (2, 128))  # batch_size=2, seq_len=128
    output = model(dummy_input)
    print(output.shape)  # [2, 128, 30522]
