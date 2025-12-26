# -*- coding:utf-8 -*-
import argparse
import os
import random
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import get_datasets
from src.models.resnet import ResNet1D18
from src.loss import NTXentLoss
from src.transforms.augmentation import WeakAugmenter, StrongAugmenter


# ==============================================================================
# 1. Linear Evaluator
# ==============================================================================
class LinearEvaluator(object):
    def __init__(self, encoder: nn.Module, num_classes: int, device: torch.device, lr=0.001):
        self.encoder = encoder
        self.device = device
        self.hidden_num = int(encoder.resnet_out_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_num, self.hidden_num // 2),
            nn.BatchNorm1d(self.hidden_num // 2),
            nn.ELU(),
            nn.Linear(self.hidden_num // 2, num_classes),
        ).to(self.device)
        self.optimizer = optim.AdamW(self.fc.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def update(self, x, y):
        self.encoder.eval()
        with torch.no_grad():
            x = x.to(self.device).float()
            if x.dim() == 3: x = x[:, -1, :]
            features = self.encoder(x)
            if isinstance(features, tuple): features = features[0]
            features = features.detach()

        self.fc.train()
        logits = self.fc(features)
        y = y.to(self.device)
        if y.dim() > 1: y = torch.argmax(y, dim=1)  # One-hot handling

        loss = self.criterion(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, test_loader):
        self.encoder.eval()
        self.fc.eval()
        total_correct = 0
        total_num = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device).float()
                if x.dim() == 3: x = x[:, -1, :]
                y = y.to(self.device)
                if y.dim() > 1: y = torch.argmax(y, dim=1)

                features = self.encoder(x)
                if isinstance(features, tuple): features = features[0]

                logits = self.fc(features)
                pred_list = torch.argmax(logits, dim=1)

                total_correct += (pred_list == y).sum().item()
                total_num += x.size(0)

        return (total_correct / total_num) * 100


# ==============================================================================
# 2. Fixed Augmentation Trainer
# ==============================================================================
class FixedAugTrainer(object):
    def __init__(self, encoder: nn.Module, config, target_action_idx: int = -1):
        self.device = torch.device(config['device'])
        self.encoder = encoder.to(self.device)
        self.optimizer = optim.AdamW(self.encoder.parameters(), lr=float(config['lr']))

        # Augmenters
        self.w_augmenter = WeakAugmenter()
        self.s_augmenter = StrongAugmenter()
        self.loss_fn = NTXentLoss(temperature=config.get('temp', 0.1))

        # Target Action Index
        # -1: Random Selection
        # 0~4: Fixed Specific Action
        self.target_action_idx = target_action_idx
        self.num_actions = config.get('num_actions', 5)

    def train_step(self, x_batch):
        x_batch = x_batch.to(self.device)
        if x_batch.dim() == 4: x_batch = x_batch.squeeze(2)

        # x_batch shape: [B, S, T] -> We use the last time step for augmentation
        x_curr = x_batch[:, -1, :].unsqueeze(1)  # [B, 1, T]

        b = x_curr.size(0)
        x_view1 = x_curr.clone()  # Anchor (Weak)
        x_view2 = x_curr.clone()  # Positive (Strong - Fixed/Random)

        for i in range(b):
            # 1. Weak Augmentation (Anchor)
            x_view1[i] = self.w_augmenter(x_curr[i])

            # 2. Strong Augmentation (Strategy Decision)
            if self.target_action_idx == -1:
                # Random Strategy: Pick any action uniformly
                action = random.randint(0, self.num_actions - 1)
            else:
                # Single Strategy: Use fixed action
                action = self.target_action_idx

            x_view2[i] = self.s_augmenter(x_curr[i], action)

        # 3. SSL Update
        self.encoder.train()
        z1 = self.encoder(x_view1)
        z2 = self.encoder(x_view2)

        # Tuple output handling (ResNet usually returns (feat, logit) or similar)
        if isinstance(z1, tuple): z1 = z1[1]  # Projection head output
        if isinstance(z2, tuple): z2 = z2[1]

        loss = self.loss_fn(z1, z2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_checkpoint(self, path):
        torch.save(self.encoder.state_dict(), path)


# ==============================================================================
# 3. Utilities & Main
# ==============================================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Augmentation Comparison")
    parser.add_argument('--config', type=str, default='configs/sleep_edf_comparison.yaml')

    # Options: 'random', 'time_masking', 'time_permutation', 'crop_resize', 'time_flip', 'time_warp'
    parser.add_argument('--aug_type', type=str, required=True,
                        help="Choose augmentation strategy: random or specific name")

    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config['seed'])

    # Augmentation Name to Index Map
    aug_map = {
        'time_masking': 0,
        'time_permutation': 1,
        'crop_resize': 2,
        'time_flip': 3,
        'time_warp': 4,
        'random': -1  # Special flag for Random Strategy
    }

    if args.aug_type not in aug_map:
        raise ValueError(f"Invalid aug_type: {args.aug_type}. Available: {list(aug_map.keys())}")

    target_idx = aug_map[args.aug_type]
    exp_name = f"{config['experiment_name']}_Ablation_{args.aug_type}"

    print(f"=== [Comparison Experiment] Strategy: {args.aug_type} (Idx: {target_idx}) ===")

    # 1. Directories Setup
    log_dir = os.path.join("runs", exp_name)
    ckpt_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 2. Dataset Setup
    train_set, test_set = get_datasets(config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    device = torch.device(config['device'])

    # 3. Model & Trainer Setup
    encoder = ResNet1D18(resnet_out_dim=config['resnet_out_dim'], feat_dim=config['feat_dim'])

    # FixedAugTrainer
    trainer = FixedAugTrainer(encoder, config, target_action_idx=target_idx)

    # Linear Evaluator for Validating Performance
    num_classes = getattr(train_set, 'num_classes', 5)
    linear_evaluator = LinearEvaluator(encoder, num_classes, device)

    # Tensorboard
    writer = SummaryWriter(log_dir=log_dir)

    # 4. Training Loop
    epochs = config['epochs']  # Or separate epoch config for ablation
    global_step = 0
    best_acc = 0.0

    for epoch in range(epochs):
        start = time.time()
        ep_loss = 0.0
        ep_lin_loss = 0.0

        for x, y in train_loader:
            # (1) SSL Training with Fixed Strategy
            loss = trainer.train_step(x)
            ep_loss += loss

            # (2) Linear Evaluation Update (Online Eval)
            lin_loss = linear_evaluator.update(x, y)
            ep_lin_loss += lin_loss

            global_step += 1

        # (3) Evaluation
        acc = linear_evaluator.evaluate(test_loader)

        writer.add_scalar("Comparison/SSL_Loss", ep_loss / len(train_loader), epoch)
        writer.add_scalar("Comparison/Linear_Acc", acc, epoch)

        print(f"Epoch [{epoch + 1}/{epochs}] ({args.aug_type}) | "
              f"Time: {time.time() - start:.1f}s | "
              f"SSL Loss: {ep_loss / len(train_loader):.4f} | "
              f"Acc: {acc:.2f}%")

        # [Check Point Saving]
        if acc > best_acc:
            best_acc = acc
            best_path = os.path.join(ckpt_dir, "best_encoder.pth")
            trainer.save_checkpoint(best_path)
            print(f" --> Best Model Saved! (Acc: {best_acc:.2f}%)")

    # Save Final Model
    final_path = os.path.join(ckpt_dir, "final_encoder.pth")
    trainer.save_checkpoint(final_path)
    print(f"=== Experiment Finished. Final Accuracy: {acc:.2f}% | Best Accuracy: {best_acc:.2f}% ===")
    print(f"Final Model Saved to {final_path}")
    writer.close()


if __name__ == "__main__":
    main()
