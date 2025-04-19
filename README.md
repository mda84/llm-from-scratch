# LLM From Scratch 🚀

This repository demonstrates how to build Large Language Models (LLMs) completely from scratch using PyTorch. It covers core architectural types, efficient training techniques, and even Reinforcement Learning from Human Feedback (RLHF).

---

## 🧠 Features

- ✅ Encoder-only (BERT-style)
- ✅ Decoder-only (GPT-style)
- ✅ Encoder-Decoder (T5/BART-style)
- ✅ Rotary Positional Embeddings (RoPE)
- ✅ Key-Value Caching
- ✅ Mixture of Experts (MoE)
- ✅ LoRA (Low-Rank Adaptation)
- ✅ Reward modeling with preferences
- ✅ PPO fine-tuning for RLHF
- ✅ Modular, extensible PyTorch code

---

## 📁 Structure

```
llm-from-scratch/
├── models/
│   ├── encoder_only.py
│   ├── decoder_only.py
│   ├── encoder_decoder.py
│   ├── rotary_embeddings.py
│   ├── kv_cache.py
│   ├── moe_layer.py
│   ├── lora_adapter.py
│   ├── utils.py
├── rlhf/
│   ├── reward_model.py
│   ├── ppo_trainer.py
│   ├── preference_dataset.py
├── notebooks/
│   ├── 01_train_decoder_only.ipynb
│   ├── 02_generate_text.ipynb
│   ├── 03_train_encoder_only.ipynb
│   ├── 04_train_encoder_decoder.ipynb
│   ├── 05_reward_model_training.ipynb
│   ├── 06_rlhf_training.ipynb
├── train.py
├── generate.py
├── requirements.txt
└── README.md
```

---

## 🧪 Notebooks
| Notebook | Purpose |
|----------|---------|
| `01_train_decoder_only.ipynb` | Train a GPT-style model from scratch |
| `02_generate_text.ipynb` | Generate text using top-k/top-p sampling |
| `03_train_encoder_only.ipynb` | Train BERT-style masked LM |
| `04_train_encoder_decoder.ipynb` | Train T5/BART-style seq2seq model |
| `05_reward_model_training.ipynb` | Train a reward model on preference pairs |
| `06_rlhf_training.ipynb` | Fine-tune model with PPO and reward feedback |

---

## 🧰 Installation
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Quickstart
```bash
python train.py --dataset_path data/tiny_shakespeare.txt
python generate.py --prompt "Once upon a time"
```

---

## 📜 License
MIT License

---

## ✨ Credits
Built by [Mohammadreza Dorkhah](https://github.com/mda84) to help developers and researchers understand and customize LLMs from scratch.

Pull requests and suggestions welcome!