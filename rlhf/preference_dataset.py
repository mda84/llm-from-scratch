# preference_dataset.py - Dataset for Reward Model Training (Preference Ranking)

import torch
from torch.utils.data import Dataset

class PreferenceDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        """
        Args:
            pairs: list of (chosen_text, rejected_text)
            tokenizer: a tokenizer with .encode or .__call__ method
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        chosen, rejected = self.pairs[idx]
        chosen_ids = self.tokenizer(chosen, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")["input_ids"].squeeze(0)
        rejected_ids = self.tokenizer(rejected, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")["input_ids"].squeeze(0)
        return {
            "chosen_input_ids": chosen_ids,
            "rejected_input_ids": rejected_ids
        }

def reward_loss(reward_chosen, reward_rejected):
    return -torch.mean(torch.log(torch.sigmoid(reward_chosen - reward_rejected)))

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    pairs = [("The sky is blue.", "The sky is green."), ("Apples are fruits.", "Apples are minerals.")]
    dataset = PreferenceDataset(pairs, tokenizer)
    item = dataset[0]
    print("Chosen shape:", item["chosen_input_ids"].shape)
    print("Rejected shape:", item["rejected_input_ids"].shape)
