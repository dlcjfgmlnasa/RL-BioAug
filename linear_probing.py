# -*- coding:utf-8 -*-
import torch
import argparse
import os
import random
import yaml
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import get_datasets
from src.models.resnet import ResNet1D18


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class LinearProbingTrainer(object):
    def __init__(self, encoder, num_classes, device, lr=0.001):
        self.encoder = encoder.to(device)
        self.device = device

        # Freeze Encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # Linear Classifier
        self.hidden_dim = int(encoder.resnet_out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ELU(),
            nn.Linear(self.hidden_dim // 2, num_classes),
        ).to(self.device)

        self.optimizer = optim.AdamW(self.classifier.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader):
        self.classifier.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(self.device).float()
            y = y.to(self.device)
            if x.dim() == 3: x = x[:, -1, :]

            with torch.no_grad():
                features = self.encoder(x)
                if isinstance(features, tuple): features = features[0]

            logits = self.classifier(features)
            if y.dim() > 1: y = torch.argmax(y, dim=1)

            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, test_loader):
        self.classifier.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device).float()
                y = y.to(self.device)
                if x.dim() == 3: x = x[:, -1, :]

                features = self.encoder(x)
                if isinstance(features, tuple): features = features[0]

                logits = self.classifier(features)
                preds = torch.argmax(logits, dim=1)

                if y.dim() > 1: y = torch.argmax(y, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y.cpu().numpy().tolist())

                correct += (preds == y).sum().item()
                total += x.size(0)

        acc = (correct / total) * 100
        return acc, all_preds, all_targets

    def save_classifier(self, path):
        torch.save(self.classifier.state_dict(), path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SSL Checkpoint via Linear Probing")
    parser.add_argument('--config', type=str, default='configs/sleep_edf_linear_probing.yaml')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    args = parser.parse_args()

    # 1. Load Config & Setup
    config = load_config(args.config)
    set_seed(config['seed'])
    device = torch.device(config['device'])

    print(f"=== Evaluation Mode: Linear Probing ===")
    print(f" > Checkpoint_path: {args.checkpoint_path}")

    # Setting Checkpoint Setting
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        save_dir = os.path.dirname(args.checkpoint_path)
        print(f" > Save Directory: {save_dir} (Same as checkpoint)")
    else:
        raise FileNotFoundError(f"[Error] Checkpoint file not found or invalid: {args.checkpoint_path}")

    # 2. Load Dataset
    train_set, test_set = get_datasets(config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    num_classes = getattr(train_set, 'num_classes', 5)

    # 3. Load Encoder
    encoder = ResNet1D18(resnet_out_dim=config['resnet_out_dim'], feat_dim=config['feat_dim'])

    # 4. Load Checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k.replace("module.", "") if "module." in k else k
        new_state_dict[name] = v

    if 'encoder' in new_state_dict:
        encoder.load_state_dict(new_state_dict['encoder'])
    else:
        encoder.load_state_dict(new_state_dict, strict=False)
    print(" > Pre-trained weights loaded successfully.")

    # 5. Training Loop
    evaluator = LinearProbingTrainer(encoder, num_classes, device, lr=config['lr'])

    print("\n>>> Start Linear Classifier Training...")
    best_acc = 0.0
    total_epochs = config['epochs']

    for epoch in range(total_epochs):
        loss = evaluator.train_epoch(train_loader)
        acc, pred_list, target_list = evaluator.evaluate(test_loader)

        if acc > best_acc:
            best_acc = acc

            # [1] Save best classifier
            best_model_name = "best_linear.pth"
            best_model_path = os.path.join(save_dir, best_model_name)
            evaluator.save_classifier(best_model_path)

            # [2] Save best predictions
            best_preds_name = "best_preds.pth"
            best_preds_path = os.path.join(save_dir, best_preds_name)

            save_dict = {
                "preds": pred_list,
                "targets": target_list,
                "accuracy": best_acc,
                "epoch": epoch + 1
            }
            torch.save(save_dict, best_preds_path)

            print(f"   --> New Best! Saved to {save_dir} (Acc: {best_acc:.2f}%)")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{total_epochs}] Loss: {loss:.4f} | Test Acc: {acc:.2f}%")

    print(f"\n=== Final Result ===")
    print(f" > Best Test Accuracy: {best_acc:.2f}%")
    print(f" > Predictions Saved at: {os.path.join(save_dir, f'best_preds.pth')}")


if __name__ == "__main__":
    main()