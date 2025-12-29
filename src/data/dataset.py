# -*- coding:utf-8 -*-
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple


class SleepEDFDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = 'train', split_ratio: float = 0.8,
                 seed: int = 42, seq_len: int = 10):
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.seq_len = seq_len
        self.num_classes = 5

        all_files = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
        total_subjects = len(all_files)

        if total_subjects == 0:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

        rng = np.random.RandomState(seed)
        indices = np.arange(total_subjects)
        rng.shuffle(indices)

        split_point = int(total_subjects * split_ratio)
        if mode == 'train':
            selected_indices = indices[:split_point]
        elif mode == 'test':
            selected_indices = indices[split_point:]
        else:
            raise ValueError(f"Invalid mode: {mode}")

        selected_files = [all_files[i] for i in selected_indices]
        print(f"[Sleep-EDF] Mode: {mode:<5} | Loading {len(selected_files)} Subjects...")

        self.data_list = []
        self.label_list = []
        self.valid_windows = []

        for file_path in selected_files:
            try:
                loaded = np.load(file_path)
                x_subj = loaded['x']  # (T, 1, 3000)
                y_subj = loaded['y']  # (T,)

                if x_subj.shape[0] < self.seq_len:
                    continue

                curr_subj_idx = len(self.data_list)
                self.data_list.append(x_subj)
                self.label_list.append(y_subj)

                # sliding window
                num_valid = x_subj.shape[0] - self.seq_len + 1
                for start_idx in range(num_valid):
                    self.valid_windows.append((curr_subj_idx, start_idx))

            except Exception as e:
                print(f"[Error] Failed to load {file_path}: {e}")

        print(f"    -> Total Valid Sequences: {len(self.valid_windows)}")

        if len(self.valid_windows) == 0:
            if mode == 'test' and split_ratio == 1.0:
                pass
            else:
                raise RuntimeError(f"No valid sequences found. Check seq_len ({self.seq_len}) vs data duration.")

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        subj_idx, start_idx = self.valid_windows[idx]

        x_data = self.data_list[subj_idx]
        y_data = self.label_list[subj_idx]
        x_seq = x_data[start_idx: start_idx + self.seq_len]

        x_seq = torch.from_numpy(x_seq).float()
        y = torch.tensor(y_data[start_idx + self.seq_len - 1]).long()
        return x_seq, y


class CHBMITDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = 'train', split_ratio: float = 0.8,
                 seed: int = 42, seq_len: int = 10, downsample_ratio: float = 1.0):
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.seq_len = seq_len
        self.mode = mode
        self.num_classes = 2

        all_files = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
        total_subjects = len(all_files)

        if total_subjects == 0:
            print(f"[Warning] No files found in {self.data_dir}.")

        # Subject Split
        rng = np.random.RandomState(seed)
        indices = np.arange(total_subjects)
        rng.shuffle(indices)

        split_point = int(total_subjects * split_ratio)
        if mode == 'train':
            selected_indices = indices[:split_point]
        elif mode == 'test':
            selected_indices = indices[split_point:]
        else:
            raise ValueError(f"Invalid mode: {mode}")

        selected_files = [all_files[i] for i in selected_indices]
        print(f"[CHB-MIT] Mode: {mode:<5} | Loading {len(selected_files)} Files...")

        self.data_list = []
        self.label_list = []
        self.valid_windows = []

        temp_window_info = []
        for file_path in selected_files:
            try:
                loaded = np.load(file_path)
                x_subj = loaded['x']
                y_subj = loaded['y']

                # Channel Dimension 확보
                if x_subj.ndim == 2:
                    x_subj = x_subj[:, np.newaxis, :]

                if x_subj.shape[0] < self.seq_len:
                    continue
                curr_subj_idx = len(self.data_list)
                self.data_list.append(x_subj)
                self.label_list.append(y_subj)

                num_valid = x_subj.shape[0] - self.seq_len + 1
                for start_idx in range(num_valid):
                    window_label = y_subj[start_idx + self.seq_len - 1]
                    temp_window_info.append((curr_subj_idx, start_idx, window_label))

            except Exception as e:
                print(f"[Error] Failed to load {file_path}: {e}")

        if mode == 'train' and downsample_ratio is not None:
            seizure_windows = [info for info in temp_window_info if info[2] == 1]
            background_windows = [info for info in temp_window_info if info[2] == 0]

            num_seizure = len(seizure_windows)
            num_background = len(background_windows)

            print(f"   -> [Before Sampling] Seizure: {num_seizure}, Background: {num_background}")

            if num_seizure > 0:
                target_bg_count = int(num_seizure * downsample_ratio)
                if num_background > target_bg_count:
                    random.seed(seed)
                    selected_bg = random.sample(background_windows, target_bg_count)
                else:
                    selected_bg = background_windows

                final_list = seizure_windows + selected_bg
                random.shuffle(final_list)

                self.valid_windows = [(s_idx, st_idx) for s_idx, st_idx, _ in final_list]
                print(
                    f"   -> [After Sampling] Total: {len(self.valid_windows)} (Seizure: {num_seizure}, BG: {len(selected_bg)})")

            else:
                print("[Warning] No seizure data found in training set. Using all background data.")
                self.valid_windows = [(s_idx, st_idx) for s_idx, st_idx, _ in temp_window_info]

        else:
            self.valid_windows = [(s_idx, st_idx) for s_idx, st_idx, _ in temp_window_info]
            num_seizure = sum(1 for info in temp_window_info if info[2] == 1)
            print(f"   -> [Test Mode] Using All Data. Total: {len(self.valid_windows)} (Seizure: {num_seizure})")

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx):
        subj_idx, start_idx = self.valid_windows[idx]

        x_data = self.data_list[subj_idx]
        y_data = self.label_list[subj_idx]

        # Get Sequence
        x_seq = x_data[start_idx: start_idx + self.seq_len]

        # Target Label (Last one in sequence)
        y = torch.tensor(y_data[start_idx + self.seq_len - 1]).long()

        x_seq = torch.from_numpy(x_seq).float().squeeze()

        # Conv1d 등을 위해 차원 유지 필요시 squeeze 조절 (N, C, L)
        if x_seq.ndim == 1:
            x_seq = x_seq.unsqueeze(0)

        return x_seq, y

def get_datasets(config: Dict) -> Tuple[Dataset, Dataset]:
    name = config['dataset_name']
    data_dir = config['data_dir']
    split_ratio = config['split_ratio']
    seed = config['seed']
    seq_len = config.get('agent_seq_len', 10)

    kwargs = {
        'data_dir': data_dir,
        'split_ratio': split_ratio,
        'seed': seed,
        'seq_len': seq_len
    }

    if name == 'sleep-edfx':
        train_set = SleepEDFDataset(mode='train', **kwargs)
        test_set = SleepEDFDataset(mode='test', **kwargs)

    elif name == 'chb-mit':
        train_set = CHBMITDataset(mode='train', **kwargs)
        test_set = CHBMITDataset(mode='test', **kwargs)

    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return train_set, test_set


if __name__ == '__main__':
    kwargs = {
        'data_dir': '../../data/chb_mit',
        'split_ratio': 0.8,
        'seed': 42,
        'seq_len': 10
    }

    train_set = CHBMITDataset(mode='train', **kwargs)
    for x, y in train_set:
        print(x.shape, end='\t')
        print(y)