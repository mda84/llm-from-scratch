# ppo_trainer.py - PPO Trainer for RLHF

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, lr=5e-5, clip_range=0.2, value_coef=0.5):
        self.policy_model = policy_model
        self.ref_model = ref_model  # frozen model
        self.reward_model = reward_model
        self.optimizer = optim.Adam(policy_model.parameters(), lr=lr)
        self.clip_range = clip_range
        self.value_coef = value_coef

    def compute_advantages(self, rewards, values):
        return rewards - values.detach()

    def compute_logprobs(self, model, input_ids):
        logits = model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=-1)  # total log prob per sample

    def train_step(self, input_ids, responses):
        # Get rewards from reward model
        with torch.no_grad():
            rewards = self.reward_model(responses)

        # Compute log probs of policy and reference model
        log_probs_policy = self.compute_logprobs(self.policy_model, responses)
        with torch.no_grad():
            log_probs_ref = self.compute_logprobs(self.ref_model, responses)

        # Compute values from current policy (can be a value head or same as reward model)
        values = rewards.detach()  # Simplified: using rewards as values

        # Compute advantages
        advantages = self.compute_advantages(rewards, values)

        # PPO loss
        ratio = torch.exp(log_probs_policy - log_probs_ref)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Optional value loss if using value model
        value_loss = F.mse_loss(values, rewards.detach())

        # Total loss
        loss = policy_loss + self.value_coef * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "rewards_mean": rewards.mean().item(),
            "advantages_mean": advantages.mean().item(),
        }

if __name__ == "__main__":
    print("This module defines a PPOTrainer class for RLHF. Instantiate with policy, ref, and reward models.")