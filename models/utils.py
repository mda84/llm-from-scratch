# utils.py - Sampling, Masking, and Helper Utilities

import torch
import torch.nn.functional as F

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    if sorted_mask[:, 0].any():
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = 0
    indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
    logits[indices_to_remove] = -float('Inf')
    return logits

def sample_logits(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    if top_p is not None:
        logits = top_p_logits(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def generate_causal_mask(seq_len, device):
    return torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)

if __name__ == "__main__":
    logits = torch.randn(2, 10000)
    sampled = sample_logits(logits, temperature=1.0, top_k=50, top_p=0.9)
    print("Sampled token ids:", sampled.shape)
