# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextAgent(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        num_actions: int = 7,
        seq_len: int = 10,
        hidden_dim: int = 128,
        n_head: int = 4,
        num_layers: int = 2
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # 1. Feature Embeddings
        self.state_embed = nn.Linear(input_dim, hidden_dim)
        self.action_embed = nn.Embedding(num_actions + 1, hidden_dim)   # num_actions + 1 for Start Token
        self.reward_embed = nn.Linear(1, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 2. Transformer
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_head,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=num_layers
        )

        # 3. Policy Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        state_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor
    ) -> torch.Tensor:
        # 1. Embeddings
        h_s = self.state_embed(state_seq)
        h_a = self.action_embed(prev_action_seq)
        h_r = self.reward_embed(prev_reward_seq)

        # 2. Fusion
        combined = torch.cat([h_s, h_a, h_r], dim=-1)
        x = self.fusion(combined)

        # 3. Positional Encoding
        curr_len = x.shape[1]
        x = x + self.pos_embed[:, :curr_len, :]

        # 4. Transformer & Head
        out = self.transformer(x)
        last_hidden = out[:, -1, :]

        return F.softmax(self.head(last_hidden), dim=-1)