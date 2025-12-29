# -*- coding:utf-8 -*-
import os
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from src.loss import NTXentLoss

from src.transforms.augmentation import WeakAugmenter
from src.transforms.augmentation import StrongAugmenter


class BioRLTrainer(object):
    def __init__(
            self,
            encoder: nn.Module,
            agent: nn.Module,
            config: Dict[str, Any],
            memory_loader: DataLoader,
    ) -> None:
        self.cfg = config
        self.device = torch.device(config['device'])
        self.encoder = encoder.to(self.device)
        self.agent = agent.to(self.device)

        # Optimizer
        self.opt_enc = optim.AdamW(self.encoder.parameters(), lr=float(config['lr_enc']))
        self.opt_agent = optim.AdamW(self.agent.parameters(), lr=float(config['lr_agent']))

        # Augmenters & Loss
        self.w_augmenter = WeakAugmenter()
        self.s_augmenter = StrongAugmenter()
        self.info_nce_loss = NTXentLoss(temperature=config.get('temp', 0.1))

        # RL Parameters
        self.top_k = config['top_k']
        self.num_actions = int(config.get('num_actions', 6))
        self.start_token_idx = self.num_actions
        self.ent_start = 0.02
        self.ent_end = 0.001

        # Evaluator (Vectorized Soft-KNN)
        self.knn_evaluator = Evaluator(
            encoder=self.encoder,
            memory_loader=memory_loader,
            device=self.device,
            k=20
        )

    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor, progress: float = 0.0) -> Dict[str, Any]:
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        if x_batch.dim() == 4:
            x_batch = x_batch.squeeze(2)

        b, s, t = x_batch.shape

        # ============================================================
        # 1. State Encoding
        # ============================================================
        x_flat = x_batch.view(b * s, 1, t).float()

        self.encoder.eval()
        with torch.no_grad():
            out = self.encoder(x_flat)
            h_flat = out[0] if isinstance(out, tuple) else out
            if h_flat.dim() > 2:
                h_flat = F.adaptive_avg_pool1d(h_flat, 1).squeeze(-1)

        state_seq = h_flat.view(b, s, -1)
        prev_action_seq = torch.full((b, s), self.start_token_idx, dtype=torch.long, device=self.device)
        prev_reward_seq = torch.zeros((b, s, 1), dtype=torch.float, device=self.device)

        # ============================================================
        # 2. Agent Decision
        # ============================================================
        self.agent.train()
        probs = self.agent(state_seq, prev_action_seq, prev_reward_seq)  # [Batch, Num_Actions]

        # Top K Selection
        k = self.top_k
        topk_probs, topk_indices = probs.topk(k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        dist = Categorical(topk_probs)
        sampled_relative_idx = dist.sample()
        log_prob = dist.log_prob(sampled_relative_idx)
        action = topk_indices.gather(1, sampled_relative_idx.unsqueeze(1)).squeeze(1)

        # ============================================================
        # 3. Augmentation & SSL Update
        # ============================================================
        x_curr = x_batch[:, -1, :].unsqueeze(1)
        x_view1 = x_curr.clone()  # Weak Augmentation
        x_view2 = x_curr.clone()  # Strong Augmentation

        for i in range(b):
            x_view1[i] = self.w_augmenter(x_curr[i])
            x_view2[i] = self.s_augmenter(x_curr[i], action[i].item())

        self.encoder.train()
        z1 = self.encoder(x_view1)
        z2 = self.encoder(x_view2)
        if isinstance(z1, tuple): z1 = z1[1]
        if isinstance(z2, tuple): z2 = z2[1]

        ssl_loss = self.info_nce_loss(z1, z2)
        self.opt_enc.zero_grad()
        ssl_loss.backward()
        self.opt_enc.step()

        # ============================================================
        # 4. RL Update (REINFORCE++ Style + Vectorized Soft-KNN Reward)
        # ============================================================
        soft_rewards = self.knn_evaluator.compute_batch_score(x_batch, y_batch)
        batch_mean = soft_rewards.mean()
        batch_std = soft_rewards.std() + 1e-8
        advantage = (soft_rewards - batch_mean) / batch_std

        # Entropy Decay
        entropy = dist.entropy().mean()
        current_ent_coeff = self.ent_start - (self.ent_start - self.ent_end) * progress
        current_ent_coeff = max(self.ent_end, current_ent_coeff)

        policy_loss = -(log_prob * advantage).mean() - (current_ent_coeff * entropy)

        self.opt_agent.zero_grad()
        policy_loss.backward()
        self.opt_agent.step()

        # ============================================================
        # 5. Logging Info
        # ============================================================
        avg_probs = probs.mean(dim=0)
        action_freqs = avg_probs.detach().cpu().numpy()
        if action_freqs.ndim == 0:
            action_freqs = action_freqs.reshape(-1)

        act_mean = action.float().mean().item()
        max_prob = probs.max(dim=-1).values.mean().item()

        return {
            'loss': ssl_loss.item(),
            'reward': batch_mean.item(),
            'entropy': entropy.item(),
            'max_prob': max_prob,
            'action_freqs': action_freqs,
            'act_mean': act_mean,
        }

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.agent.state_dict(), path)
        print(f"Agent saved to {path}")


class Evaluator(object):
    def __init__(
            self,
            encoder: nn.Module,
            memory_loader: DataLoader,
            device: torch.device,
            k: int = 20
    ) -> None:
        self.encoder = encoder
        self.memory_loader = memory_loader
        self.device = device
        self.k = k

    @torch.no_grad()
    def compute_batch_score(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        memory_feats = []
        memory_labels = []
        for x, y in self.memory_loader:
            x = x.to(self.device).float()
            if x.dim() == 3: x = x[:, -1, :]
            feat, _ = self.encoder(x)
            memory_feats.append(F.normalize(feat, dim=1))
            memory_labels.append(y.to(self.device))
        memory_feats = torch.cat(memory_feats, dim=0)
        memory_labels = torch.cat(memory_labels, dim=0)

        x_query = x_batch.to(self.device).float()
        if x_query.dim() == 3: x_query = x_query[:, -1, :]
        query_feats, _ = self.encoder(x_query)
        query_feats = F.normalize(query_feats, dim=1)
        query_labels = y_batch.to(self.device)

        sim_mat = torch.mm(query_feats, memory_feats.t())
        _, top_k_indices = sim_mat.topk(k=self.k, dim=1)
        top_k_labels = memory_labels[top_k_indices]

        correct_neighbors = (top_k_labels == query_labels.unsqueeze(1)).float()
        soft_scores = correct_neighbors.mean(dim=1)
        self.encoder.train()
        return soft_scores
