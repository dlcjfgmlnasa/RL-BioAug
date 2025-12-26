# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Any

from src.transforms.augmentation import WeakAugmenter
from src.transforms.augmentation import StrongAugmenter
from src.loss import NTXentLoss


class SSLRetrainer(object):
    def __init__(
            self,
            encoder: nn.Module,
            agent: nn.Module,
            config: Dict[str, Any],
    ) -> None:
        self.device = torch.device(config['device'])

        self.encoder = encoder.to(self.device)
        self.opt_enc = optim.AdamW(self.encoder.parameters(), lr=float(config['lr_enc']))

        self.agent = agent.to(self.device)
        self.agent.eval()
        for param in self.agent.parameters():
            param.requires_grad = False

        self.w_augmenter = WeakAugmenter()
        self.s_augmenter = StrongAugmenter()
        self.info_nce_loss = NTXentLoss(temperature=config.get('temp', 0.1))

        self.num_actions = int(config.get('num_actions', 6))
        self.start_token_idx = self.num_actions

        self.top_k = config['top_k']

    def train_step(self, x_batch: torch.Tensor) -> Dict[str, float]:
        x_batch = x_batch.to(self.device)
        if x_batch.dim() == 4: x_batch = x_batch.squeeze(2)
        b, s, t = x_batch.shape

        # 1. State Encoding
        x_flat = x_batch.view(b * s, 1, t).float()
        with torch.no_grad():
            out = self.encoder(x_flat)
            if isinstance(out, tuple): out = out[0]
            if out.dim() > 2: out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)
            state_seq = out.view(b, s, -1)

            prev_action = torch.full((b, s), self.start_token_idx, dtype=torch.long, device=self.device)
            prev_reward = torch.zeros((b, s, 1), dtype=torch.float, device=self.device)  # Reward는 의미 없음 (0으로 고정)

            # Agent makes action decision
            probs = self.agent(state_seq, prev_action, prev_reward)

            topk_probs, topk_indices = probs.topk(self.top_k, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

            dist = Categorical(topk_probs)
            sampled_relative_idx = dist.sample()

            action = topk_indices.gather(1, sampled_relative_idx.unsqueeze(1)).squeeze(1)

        # 2. Augmentation & SSL
        x_curr = x_batch[:, -1, :].unsqueeze(1)
        x_view1 = x_curr.clone()
        x_view2 = x_curr.clone()

        for i in range(b):
            x_view1[i] = self.w_augmenter(x_curr[i])
            x_view2[i] = self.s_augmenter(x_curr[i], action[i].item())

        self.encoder.train()
        z1 = self.encoder(x_view1)
        z2 = self.encoder(x_view2)
        if isinstance(z1, tuple): z1 = z1[1]
        if isinstance(z2, tuple): z2 = z2[1]

        # 3. InfoNCE Loss (Pure SSL)
        ssl_loss = self.info_nce_loss(z1, z2)

        self.opt_enc.zero_grad()
        ssl_loss.backward()
        self.opt_enc.step()

        return {
            'loss': ssl_loss.item(),
            'action_mean': action.float().mean().item()
        }

    def load_agent(self, path: str):
        self.agent.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Pre-trained Agent loaded from {path}")