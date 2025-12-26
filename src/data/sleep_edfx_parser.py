# -*- coding:utf-8 -*-
import argparse

import glob
import os
from typing import List, Tuple, Optional, Dict

import mne
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default='/home/brainlab/Dataset/sleep-edfx/1.0.0/sleep-cassette')
    parser.add_argument('--output_dir',
                        type=str,
                        default='../../data/sleep-edf/')
    return parser.parse_args()


class SleepEDFxParser(object):
    def __init__(
            self,
            data_dir: str,
            output_dir: str,
            select_channel: str = "EEG Fpz-Cz",
            apply_normalization: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.select_channel = select_channel
        self.apply_normalization = apply_normalization

        self.annot_map: Dict[str, int] = {
            "Sleep stage W": 0,
            "Sleep stage 1": 1,
            "Sleep stage 2": 2,
            "Sleep stage 3": 3,
            "Sleep stage 4": 3,  # N4 -> N3
            "Sleep stage R": 4,
        }

        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _normalize_subject(x: np.ndarray) -> np.ndarray:
        mean = np.mean(x)
        std = np.std(x) + 1e-8  # 0 나누기 방지

        x_norm = (x - mean) / std

        # 2. Clipping (remove Outlier)
        x_norm = np.clip(x_norm, -10.0, 10.0)
        return x_norm

    def get_subject_files(self) -> List[Tuple[str, str]]:
        psg_pattern = os.path.join(self.data_dir, "*PSG.edf")
        hyp_pattern = os.path.join(self.data_dir, "*Hypnogram.edf")

        psg_files: List[str] = sorted(glob.glob(psg_pattern))
        hyp_files: List[str] = sorted(glob.glob(hyp_pattern))

        pairs: List[Tuple[str, str]] = []

        for psg in psg_files:
            psg_basename = os.path.basename(psg).split("-")[0]
            subject_key = psg_basename[:-2]

            matching_hyp = [h for h in hyp_files if subject_key in os.path.basename(h)]

            if matching_hyp:
                pairs.append((psg, matching_hyp[0]))
            else:
                print(f"[Warning] No hypnogram found for {psg_basename} (Key: {subject_key})")
        return pairs

    def process_subject(
            self, psg_path: str, hyp_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            # 1. Read Raw Data
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
            annot = mne.read_annotations(hyp_path)
            annot.crop(annot[1]["onset"] - 30 * 60, annot[-2]["onset"] + 30 * 60)
            raw.set_annotations(annot)

            # 2. Select Channel & Filter
            if self.select_channel:
                raw.pick_channels([self.select_channel])

            raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)

            # 3. Epoch
            events, _ = mne.events_from_annotations(
                raw,
                event_id=self.annot_map,
                chunk_duration=30.0,
                verbose=False,
            )

            tmax = 30.0 - 1.0 / raw.info["sfreq"]
            epochs = mne.Epochs(
                raw,
                events,
                event_id=self.annot_map,
                tmin=0.0,
                tmax=tmax,
                baseline=None,
                verbose=False,
            )

            # 4. Extract Data (uV scale)
            x: np.ndarray = epochs.get_data() * 1e6
            y: np.ndarray = epochs.events[:, 2]

            # 5. Subject-wise Normalization
            if self.apply_normalization:
                x = self._normalize_subject(x)

            return x.astype(np.float32), y.astype(np.int64)

        except Exception as e:
            filename = os.path.basename(psg_path)
            print(f"[Error] Failed to process {filename}: {e}")
            return None, None

    def run(self) -> None:
        pairs = self.get_subject_files()
        print(f"Found {len(pairs)} subject pairs. Normalization: {self.apply_normalization}")

        for psg, hyp in tqdm(pairs, desc="Processing"):
            x, y = self.process_subject(psg, hyp)

            if x is not None and y is not None:
                basename = os.path.basename(psg).split("-")[0]
                save_path = os.path.join(self.output_dir, f"{basename}.npz")
                x, y = x.squeeze(), y.squeeze()
                np.savez_compressed(save_path, x=x, y=y)

        print(f"Done! Data saved in: {self.output_dir}")


if __name__ == "__main__":
    augments = get_args()
    DATA_DIR = augments.data_dir
    OUTPUT_DIR = augments.output_dir

    data_parser = SleepEDFxParser(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        select_channel="EEG Fpz-Cz",
        apply_normalization=True  # 정규화 활성화
    )

    if os.path.exists(DATA_DIR):
        data_parser.run()
    else:
        print(f"Directory not found: {DATA_DIR}")
