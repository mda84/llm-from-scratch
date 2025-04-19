# encoder_decoder.py - Encoder-Decoder Transformer (e.g., T5/BART-style)

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
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

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.ln2(x + self.dropout(ff_out))

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        tgt_len = x.size(1)
        #causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=x.device)).unsqueeze(0)
        #self_attn_out, _ = self.self_attn(x, x, x, attn_mask=(causal_mask == 0))
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=x.device)).bool()
        self_attn_out, _ = self.self_attn(x, x, x, attn_mask=~causal_mask)
        x = self.ln1(x + self.dropout(self_attn_out))

        cross_attn_out, _ = self.cross_attn(x, enc_out, enc_out, attn_mask=memory_mask)
        x = self.ln2(x + self.dropout(cross_attn_out))

        ff_out = self.ff(x)
        return self.ln3(x + self.dropout(ff_out))

class EncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_layers, dec_layers, heads, ff_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = PositionalEncoding(embed_dim, max_len)

        self.encoder = nn.ModuleList([
            EncoderLayer(embed_dim, heads, ff_dim, dropout) for _ in range(enc_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(embed_dim, heads, ff_dim, dropout) for _ in range(dec_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def encode(self, src, src_mask=None):
        x = self.token_embed(src)
        x = self.pos_embed(x)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, tgt_mask=None, memory_mask=None):
        x = self.token_embed(tgt)
        x = self.pos_embed(x)
        for layer in self.decoder:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        return x

    def forward(self, src, tgt):
        enc_out = self.encode(src)
        dec_out = self.decode(tgt, enc_out)
        dec_out = self.norm(dec_out)
        return self.head(dec_out)

if __name__ == "__main__":
    model = EncoderDecoderModel(vocab_size=32128, embed_dim=512, enc_layers=6, dec_layers=6, heads=8, ff_dim=2048)
    src = torch.randint(0, 32128, (2, 64))  # batch x src_len
    tgt = torch.randint(0, 32128, (2, 32))  # batch x tgt_len
    output = model(src, tgt)
    print(output.shape)  # [2, 32, 32128]
