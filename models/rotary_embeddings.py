# rotary_embeddings.py - Rotary Positional Embeddings (RoPE)

import torch
import torch.nn as nn
import math

class RotaryEmbedding:
    def __init__(self, dim, base=10000):
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_frequencies(inv_freq)

    def register_frequencies(self, inv_freq):
        self.inv_freq = inv_freq

    def get_rotation_angles(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

    def apply_rotary_pos_emb(self, x, rope_cache):
        # x: [batch, seq, num_heads, head_dim]
        x1, x2 = x[..., ::2], x[..., 1::2]
        sin, cos = rope_cache
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

def build_rope_cache(seq_len, dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, device=device).type_as(inv_freq)
    freqs = torch.einsum("i , j -> i j", t, inv_freq)
    sin, cos = freqs.sin(), freqs.cos()
    sin, cos = map(lambda t: t[None, None, :, :], (sin, cos))  # [1, 1, seq_len, dim//2]
    return sin, cos

if __name__ == "__main__":
    rope = RotaryEmbedding(dim=64)
    rope_angles = rope.get_rotation_angles(seq_len=128, device=torch.device("cpu"))
    print("RoPE angles shape:", rope_angles.shape)  # [128, 64]
