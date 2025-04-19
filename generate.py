# generate.py - Autoregressive Text Generation with GPT-style Decoder

import torch
from transformers import AutoTokenizer
from models.decoder_only import GPTStyleDecoder
from models.utils import sample_logits
import argparse

def generate(model, tokenizer, prompt, max_length, temperature, top_k, top_p, device):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = input_ids.clone()

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            next_token = sample_logits(next_token_logits, temperature, top_k, top_p)
            next_token = next_token.unsqueeze(1)
            generated = torch.cat((generated, next_token), dim=1)

    output = tokenizer.batch_decode(generated[:, input_ids.size(1):], skip_special_tokens=True)[0]
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="gpt_model.pt")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--block_size", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = GPTStyleDecoder(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        ff_dim=args.ff_dim,
        max_len=args.block_size
    )
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)

    output = generate(model, tokenizer, args.prompt, args.max_length, args.temperature, args.top_k, args.top_p, args.device)
    print(f"\nPrompt: {args.prompt}\nGenerated: {output}")