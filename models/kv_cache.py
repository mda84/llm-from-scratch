# kv_cache.py - Key-Value Cache for Efficient Autoregressive Decoding

import torch

class KVCache:
    def __init__(self, max_batch_size, max_seq_len, num_heads, head_dim, device):
        self.max_seq_len = max_seq_len
        self.cache = {
            'k': torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim, device=device),
            'v': torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim, device=device)
        }
        self.device = device
        self.current_seq_len = torch.zeros(max_batch_size, dtype=torch.long, device=device)

    def update(self, keys, values, batch_idx):
        # keys/values: [batch, heads, seq_len, head_dim]
        bsz, num_heads, seq_len, head_dim = keys.shape
        for i in range(bsz):
            pos = self.current_seq_len[batch_idx[i]]
            self.cache['k'][batch_idx[i], :, pos:pos+seq_len, :] = keys[i]
            self.cache['v'][batch_idx[i], :, pos:pos+seq_len, :] = values[i]
            self.current_seq_len[batch_idx[i]] += seq_len

    def get_cache(self, batch_idx):
        # Returns sliced cache up to current seq_len for each batch
        k_list, v_list = [], []
        for i in batch_idx:
            end = self.current_seq_len[i]
            k_list.append(self.cache['k'][i:i+1, :, :end, :])
            v_list.append(self.cache['v'][i:i+1, :, :end, :])
        return torch.cat(k_list, dim=0), torch.cat(v_list, dim=0)

    def reset(self, batch_idx):
        for i in batch_idx:
            self.current_seq_len[i] = 0
            self.cache['k'][i].zero_()
            self.cache['v'][i].zero_()

if __name__ == "__main__":
    bsz, heads, seqlen, dim = 2, 4, 10, 16
    device = torch.device("cpu")
    kv = KVCache(max_batch_size=bsz, max_seq_len=128, num_heads=heads, head_dim=dim, device=device)
    k = torch.randn(bsz, heads, seqlen, dim)
    v = torch.randn(bsz, heads, seqlen, dim)
    kv.update(k, v, batch_idx=[0, 1])
    k_out, v_out = kv.get_cache(batch_idx=[0, 1])
    print("Cache shapes:", k_out.shape, v_out.shape)  # [2, heads, seqlen, dim]