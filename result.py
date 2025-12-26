# -*- coding:utf-8 -*-
import os
import torch
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def main():
    parser = argparse.ArgumentParser(description="Result")
    parser.add_argument('--checkpoint_path', type=str, default=os.path.join('.', 'checkpoints'))
    args = parser.parse_args()
    base_path = args.checkpoint_path

    for name in os.listdir(base_path):
        try:
            ckpt_path = os.path.join(base_path, name, 'best_preds.pth')
            ckpt = torch.load(ckpt_path)
            preds, reals = ckpt['preds'], ckpt['targets']
            acc = accuracy_score(y_true=reals, y_pred=preds)
            mf1 = f1_score(y_true=reals, y_pred=preds, average='macro')
            print(ckpt_path.split('/')[-2], end='\t')
            print(f'acc: {acc*100} \t mf1: {mf1*100}')
        except FileNotFoundError:
            continue

if __name__ == "__main__":
    main()