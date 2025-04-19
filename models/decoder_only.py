# decoder_only.py - Decoder-Only Transformer (e.g., GPT-style)

import torch
import torch.nn as nn
import math

class GPTPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x, sin=None, cos=None, kv_cache=None):
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=~causal_mask)
        #causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)
        #attn_out, _ = self.attn(x, x, x, attn_mask=(causal_mask == 0))
        x = self.ln(x + self.dropout(attn_out))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = CausalSelfAttention(embed_dim, heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    #def forward(self, x):
    #    x = self.self_attn(x)
    #    ff_out = self.ff(x)
    #    x = self.ln(x + self.dropout(ff_out))
    #    return x

    def forward(self, x, sin=None, cos=None, kv_cache=None):
        x = self.self_attn(x, sin=sin, cos=cos, kv_cache=kv_cache)
        ff_out = self.ff(x)
        x = self.ln(x + self.dropout(ff_out))
        return x

class GPTStyleDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, depth, heads, ff_dim, max_len=2048, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim       
        self.num_heads = heads            
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = GPTPositionalEncoding(embed_dim, max_len)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, heads, ff_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def rope_dims(self):
        return self.embed_dim // self.num_heads

    #def forward(self, x):
    #    x = self.token_embed(x)
    #    x = self.pos_embed(x)
    #    for block in self.blocks:
    #        x = block(x)
    #    x = self.norm(x)
    #    return self.lm_head(x)

    def forward(self, x, sin=None, cos=None, kv_cache=None):
        x = self.token_embed(x)
        
        # Add positional encoding (use absolute if no RoPE)
        if sin is None or cos is None:
            x = self.pos_embed(x)
        
        for block in self.blocks:
            x = block(x, sin=sin, cos=cos, kv_cache=kv_cache)
        
        x = self.norm(x)
        return self.lm_head(x)

if __name__ == "__main__":
    model = GPTStyleDecoder(vocab_size=50257, embed_dim=768, depth=12, heads=12, ff_dim=3072)
    dummy_input = torch.randint(0, 50257, (2, 128))
    output = model(dummy_input)
    print(output.shape)  # [2, 128, 50257]
