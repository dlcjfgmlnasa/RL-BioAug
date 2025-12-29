# -*- coding:utf-8 -*-
import argparse
import os
import random
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.main_search import BioRLTrainer
from src.main_retrain import SSLRetrainer

from src.data.dataset import get_datasets
from src.models.resnet import ResNet1D18
from src.models.policy import ContextAgent


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
            x = x[:, -1, :]
            features = self.encoder(x)[0]
            features = features.detach()
        self.fc.train()
        logits = self.fc(features)
        y = y.to(self.device)
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
                x = x[:, -1, :]
                y = y.to(self.device)
                features = self.encoder(x)[0]
                logits = self.fc(features)
                pred_list = torch.argmax(logits, dim=1)

                if y.dim() > 1: y = torch.argmax(y, dim=1)
                total_correct += (pred_list == y).sum().item()
                total_num += x.size(0)

        return (total_correct / total_num) * 100


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Integrated Bio-Signal RL Training")
    parser.add_argument('--config', type=str, default='configs/chb_mit_config_topk_1.yaml')
    parser.add_argument('--mode', type=str, default='all', choices=['search', 'retrain', 'all'])
    args = parser.parse_args()

    # 1. Init Setting
    config = load_config(args.config)
    set_seed(config['seed'])
    device = torch.device(config['device'])

    log_dir = os.path.join("runs", config['experiment_name'])
    writer = SummaryWriter(log_dir=log_dir)
    ckpt_dir = os.path.join('checkpoints', config['experiment_name'])
    os.makedirs(ckpt_dir, exist_ok=True)

    best_agent_path = os.path.join(ckpt_dir, "best_agent.pth")

    print(f"=== Experiment: {config['experiment_name']} (Mode: {args.mode}) ===")

    # 2. Dataset & DataLoader
    num_workers = config.get('num_workers', 2)
    train_set, test_set = get_datasets(config)
    full_train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    search_ratio = config.get('search_data_ratio', 1.0)
    num_search = int(len(train_set) * search_ratio)

    print(f"=== Dataset Split Info ===")
    print(f" > Total Train Samples: {len(train_set)}")
    print(f" > Search Phase Usage : {num_search} samples ({search_ratio * 100:.1f}%)")
    print(f" > Retrain Phase Usage: {len(train_set)} samples (100%)")

    search_indices = torch.randperm(len(train_set))[:num_search].tolist()
    search_subset = Subset(train_set, search_indices)

    search_loader = DataLoader(
        search_subset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Memory Loader for RL Reward
    mem_indices = torch.randperm(len(search_subset))[:min(len(search_subset), 1000)].tolist()
    memory_loader = DataLoader(
        Subset(search_subset, mem_indices),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers
    )

    # ==========================================================================
    # [Phase 1] Search Strategy (Using Search Subset & KNN Eval)
    # ==========================================================================
    if args.mode in ['search', 'all']:
        print("\n>>> [Phase 1] Start Policy Search...")

        encoder = ResNet1D18(resnet_out_dim=config['resnet_out_dim'], feat_dim=config['feat_dim'])
        agent = ContextAgent(
            input_dim=config['resnet_out_dim'], num_actions=config['num_actions'],
            seq_len=config['agent_seq_len'], hidden_dim=config['agent_hidden'],
            n_head=config['agent_num_head'], num_layers=config['agent_num_layers']
        )

        trainer = BioRLTrainer(encoder, agent, config, memory_loader)

        global_step = 0
        best_acc = 0.0

        for epoch in range(config['epochs']):
            ep_stats = {'loss': 0, 'reward': 0}
            start = time.time()

            for x, y in search_loader:
                progress = global_step / (config['epochs'] * len(search_loader))
                stats = trainer.train_step(x, y, progress=progress)

                # TensorBoard Writer
                writer.add_scalar("Search/Loss", stats['loss'], global_step)
                writer.add_scalar("Search/Reward", stats['reward'], global_step)

                action_names = ['time_masking', 'time_permutation', 'crop_resize', 'time_flip', 'time_warp']
                if 'action_freqs' in stats:
                    for i, name in enumerate(action_names):
                        if i < len(stats['action_freqs']):
                            writer.add_scalar(f"Actions/{name}", stats['action_freqs'][i], global_step)

                ep_stats['loss'] += stats['loss']
                ep_stats['reward'] += stats['reward']
                global_step += 1

            # KNN Evaluation
            acc = knn_evaluation(encoder, search_loader, test_loader, device)
            writer.add_scalar("Search/Eval_KNN", acc, epoch + 1)

            print(f"Search Epoch [{epoch + 1}/{config['epochs']}] "
                  f"Time: {time.time() - start:.1f}s | "
                  f"Reward: {ep_stats['reward'] / len(search_loader):.3f} | "
                  f"KNN: {acc:.2f}%")

            if acc > best_acc:
                best_acc = acc
                trainer.save_checkpoint(best_agent_path)
                print(f"  --> Best Agent Saved! (KNN: {best_acc:.2f}%)")

    # ==========================================================================
    # [Phase 2] Retrain Strategy (Using Full Dataset & Linear Evaluation)
    # ==========================================================================
    if args.mode in ['retrain', 'all']:
        print("\n>>> [Phase 2] Start Pure SSL Retraining with Linear Evaluation...")

        if not os.path.exists(best_agent_path):
            print(f"[Error] Best agent checkpoint not found at {best_agent_path}! Cannot retrain.")
            return

        encoder_retrain = ResNet1D18(resnet_out_dim=config['resnet_out_dim'], feat_dim=config['feat_dim'])

        # 2. Agent Load & Freeze
        agent_fixed = ContextAgent(
            input_dim=config['resnet_out_dim'], num_actions=config['num_actions'],
            seq_len=config['agent_seq_len'], hidden_dim=config['agent_hidden'],
            n_head=config['agent_num_head'], num_layers=config['agent_num_layers']
        )

        # 3. Retrainer Initialization
        retrainer = SSLRetrainer(encoder_retrain, agent_fixed, config)
        retrainer.load_agent(best_agent_path)

        num_classes = getattr(train_set, 'num_classes', 5)
        linear_evaluator = LinearEvaluator(
            encoder=encoder_retrain,
            num_classes=num_classes,
            device=device,
            lr=0.001
        )

        global_step = 0
        retrain_epochs = config.get('retrain_epochs', config['epochs'])

        for epoch in range(retrain_epochs):
            ep_ssl_loss = 0.0
            ep_lin_loss = 0.0
            start = time.time()

            for x, y in full_train_loader:
                # [A] SSL Training
                stats = retrainer.train_step(x)
                ep_ssl_loss += stats['loss']

                # [B] Linear Evaluation
                lin_loss = linear_evaluator.update(x, y)
                ep_lin_loss += lin_loss

                # TensorBoard Writer
                writer.add_scalar("Retrain/SSL_Loss", stats['loss'], global_step)
                writer.add_scalar("Retrain/Linear_Loss", lin_loss, global_step)
                global_step += 1

            # Evaluation
            acc = linear_evaluator.evaluate(test_loader)
            writer.add_scalar("Retrain/Eval_Linear_Acc", acc, epoch + 1)

            print(f"Retrain Epoch [{epoch + 1}/{retrain_epochs}] "
                  f"Time: {time.time() - start:.1f}s | "
                  f"SSL Loss: {ep_ssl_loss / len(full_train_loader):.4f} | "
                  f"Lin Loss: {ep_lin_loss / len(full_train_loader):.4f} | "
                  f"Linear Acc: {acc:.2f}%")

        torch.save(encoder_retrain.state_dict(), os.path.join(ckpt_dir, "final_encoder_retrained.pth"))
        print(f"Final Retrained Model Saved.")

    writer.close()


def knn_evaluation(encoder, train_loader, test_loader, device, k=200):
    encoder.eval()
    mem_feats, mem_labels = [], []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device).float()
            if x.dim() == 4:
                x = x[:, -1, :, :].squeeze(1)
            elif x.dim() == 3:
                x = x[:, -1, :]

            h = encoder(x)
            if isinstance(h, tuple): h = h[0]
            if h.dim() > 2: h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
            mem_feats.append(F.normalize(h, dim=1).cpu())
            mem_labels.append(y.cpu())

    mem_feats = torch.cat(mem_feats, 0).to(device)
    mem_labels = torch.cat(mem_labels, 0).to(device)

    total_acc, total_num = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            b = x.size(0)
            x = x.to(device).float()
            x = x[:, -1, :]

            h = encoder(x)[0]
            h = F.normalize(h, dim=1)

            sim = torch.mm(h, mem_feats.t())
            _, idx_list = sim.topk(k, dim=1)
            pred_list = mem_labels[idx_list]

            if pred_list.dim() > 2: pred_list = torch.argmax(pred_list, dim=2)
            pred_labels = pred_list.mode(dim=1).values

            if y.dim() > 1: y = torch.argmax(y, dim=1)
            total_acc += (pred_labels == y.to(device)).sum().item()
            total_num += b

    return (total_acc / total_num) * 100


if __name__ == "__main__":
    main()