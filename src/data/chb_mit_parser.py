# -*- coding:utf-8 -*-
import argparse
import glob
import os
import re
from typing import List, Tuple, Optional, Dict, Any

import mne
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default='/home/brainlab/Dataset/chb_mit/physionet.org/files/chbmit/1.0.0/',
                        help="Path to CHB-MIT dataset root directory containing chb01, chb02...")
    parser.add_argument('--output_dir',
                        type=str,
                        default='../../data/chb_mit/',
                        help="Output directory for .npz files")
    parser.add_argument('--chunk_duration',
                        type=float,
                        default=4.0,
                        help="Epoch duration in seconds (e.g., 4s for seizure detection)")
    return parser.parse_args()


class CHBMITParser(object):
    def __init__(
            self,
            data_dir: str,
            output_dir: str,
            chunk_duration: float = 2.0,
            select_channel: str = "FP1-F7",
            apply_normalization: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.chunk_duration = chunk_duration
        self.select_channel = select_channel
        self.apply_normalization = apply_normalization

        # 0: Interictal (Background), 1: Ictal (Seizure)
        self.annot_map: Dict[str, int] = {
            "Background": 0,
            "Seizure": 1,
        }

        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _normalize_subject(x: np.ndarray) -> np.ndarray:
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        x_norm = (x - mean) / std

        # Clipping to remove outliers
        x_norm = np.clip(x_norm, -10.0, 10.0)
        return x_norm

    @staticmethod
    def _parse_summary_file(summary_path: str) -> Dict[str, List[Tuple[float, float]]]:
        seizure_info = {}

        if not os.path.exists(summary_path):
            print(f"[Warning] Summary file not found: {summary_path}")
            return {}

        with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        file_blocks = content.split("File Name:")
        for block in file_blocks:
            if not block.strip(): continue

            lines = block.strip().split('\n')
            filename = lines[0].strip()

            starts = []
            ends = []

            for line in lines:
                # 1. Seizure Start Time
                if "Seizure" in line and "Start Time" in line and "seconds" in line:
                    match = re.search(r"(\d+)\s+seconds", line)
                    if match:
                        starts.append(float(match.group(1)))

                # 2. Seizure End Time
                elif "Seizure" in line and "End Time" in line and "seconds" in line:
                    match = re.search(r"(\d+)\s+seconds", line)
                    if match:
                        ends.append(float(match.group(1)))

            if len(starts) > 0 and len(starts) == len(ends):
                intervals = list(zip(starts, ends))
                seizure_info[filename] = intervals
            elif len(starts) != len(ends):
                print(
                    f"[Warning] Mismatched seizure start/end times in {filename} (Starts: {len(starts)}, Ends: {len(ends)})")
                seizure_info[filename] = []
            else:
                seizure_info[filename] = []

        return seizure_info

    def get_subject_files(self) -> List[Tuple[str, List[Tuple[float, float]]]]:
        # CHB-MIT structure: root/chb01/chb01_01.edf, root/chb01/chb01-summary.txt
        subject_dirs = sorted(glob.glob(os.path.join(self.data_dir, "chb*")))
        file_tasks = []

        for subj_dir in subject_dirs:
            if not os.path.isdir(subj_dir): continue

            subj_name = os.path.basename(subj_dir)
            summary_pattern = os.path.join(subj_dir, f"{subj_name}-summary.txt")
            summary_files = glob.glob(summary_pattern)

            if not summary_files:
                print(f"[Warning] No summary file found for {subj_name}")
                seizure_map = {}
            else:
                seizure_map = self._parse_summary_file(summary_files[0])

            # Find all EDFs in this subject folder
            edf_files = sorted(glob.glob(os.path.join(subj_dir, "*.edf")))

            for edf_path in edf_files:
                basename = os.path.basename(edf_path)
                # Get seizure intervals for this file (empty list if none)
                intervals = seizure_map.get(basename, [])
                file_tasks.append((edf_path, intervals))

        return file_tasks

    def process_subject(
            self, edf_path: str, seizure_intervals: List[Tuple[float, float]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            # 1. Read Raw Data
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='error')
            candidate_channels = [
                # Temporal
                "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
                "FP2-F8", "F8-T8", "T8-P8", "P8-O2",

                # Parasagittal
                "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
                "FP2-F4", "F4-C4", "C4-P4", "P4-O2",

                # Central
                "FZ-CZ", "CZ-PZ"
            ]

            selected_channel = None
            available_chs = raw.ch_names

            for cand in candidate_channels:
                if cand in available_chs:
                    selected_channel = cand
                    break

                elif f"{cand}." in available_chs:
                    selected_channel = f"{cand}."
                    break

                elif f"EEG {cand}" in available_chs:
                    selected_channel = f"EEG {cand}"
                    break

            if selected_channel is None:
                print(f"[Skip] No valid channel found in {os.path.basename(edf_path)}. \n"
                      f"       Available: {available_chs[:5]} ...")
                return None, None

            raw.pick_channels([selected_channel])
            raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=self.chunk_duration,
                preload=True,
                verbose=False
            )

            if len(epochs) == 0:
                return None, None

            # 4. Extract Data & Labeling
            x: np.ndarray = epochs.get_data() * 1e6  # uV scale

            sfreq = raw.info['sfreq']
            epoch_start_times = epochs.events[:, 0] / sfreq

            y = np.zeros(len(epochs), dtype=np.int64)

            for start_t, end_t in seizure_intervals:
                mid_points = epoch_start_times + (4.0 / 2)  # duration 절반
                seizure_mask = (mid_points >= start_t) & (mid_points <= end_t)
                y[seizure_mask] = 1

            # 5. Normalization
            if self.apply_normalization:
                x = self._normalize_subject(x)

            return x.astype(np.float32), y.astype(np.int64)

        except Exception as e:
            print(f"[Error] Failed to process {os.path.basename(edf_path)}: {e}")
            return None, None

    def run(self) -> None:
        tasks = self.get_subject_files()
        print(f"Found {len(tasks)} EDF files. Normalization: {self.apply_normalization}")

        for edf_path, seizure_intervals in tqdm(tasks, desc="Processing"):
            x, y = self.process_subject(edf_path, seizure_intervals)

            if x is not None and y is not None:
                basename = os.path.basename(edf_path).replace(".edf", "")
                save_path = os.path.join(self.output_dir, f"{basename}.npz")
                x, y = x.squeeze(), y.squeeze()
                np.savez_compressed(save_path, x=x, y=y)

        print(f"Done! Data saved in: {self.output_dir}")


if __name__ == "__main__":
    augments = get_args()
    DATA_DIR = augments.data_dir
    OUTPUT_DIR = augments.output_dir

    parser = CHBMITParser(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        chunk_duration=4.0,
        select_channel="FP1-F7",
        apply_normalization=True
    )

    if os.path.exists(DATA_DIR):
        parser.run()
    else:
        print(f"Directory not found: {DATA_DIR}")