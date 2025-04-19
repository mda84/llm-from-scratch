# train.py - Training Script for LLM Models

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

from models.decoder_only import GPTStyleDecoder

class TextDataset(Dataset):
    def __init__(self, tokenizer, text, block_size=128):
        tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
        self.examples = [tokens[i:i+block_size] for i in range(0, len(tokens)-block_size, block_size)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        return x[:-1], x[1:]  # input, target

def train_model(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()

    dataset = TextDataset(tokenizer, text, block_size=args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = GPTStyleDecoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        ff_dim=args.ff_dim,
        max_len=args.block_size
    ).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0
        for x, y in loop:
            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss/len(loop))

    torch.save(model.state_dict(), args.output_path)
    print(f"Model saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/tiny_shakespeare.txt")
    parser.add_argument("--output_path", type=str, default="gpt_model.pt")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_model(args)
