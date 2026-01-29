# RL-BioAug

**Label-Efficient Reinforcement Learning for Self-Supervised EEG Representation Learning**

> Under Review

![Main Results](main_figure.png)

## Abstract

Contrastive learning has emerged as a promising paradigm in EEG analysis by reducing reliance on labeled data. In general, the effectiveness of contrastive learning fundamentally depends on the quality of data augmentation due to non-stationarity of EEG signals with statistical properties changing over time. Static or random augmentation strategies can corrupt intrinsic information, therefore leading to degraded performance in representation learning.

To address this, we propose **RL-BioAug**, a reinforcement learning (RL)-based autonomous augmentation framework for self-supervised EEG representation learning. The proposed RL agent analyzes the features of input signals to determine an optimal augmentation policy tailored to each instance.

**Key Findings:**
- Superior performance on Sleep-EDFX and CHB-MIT datasets
- Adaptive strategy selection: *Time Masking* for sleep stage classification, *Crop Resize* for seizure detection
- Potential to replace conventional heuristic-based augmentations

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Augmentation Strategies](#augmentation-strategies)
- [License](#license)

---

## Installation

### Requirements

```bash
# Core dependencies
pip install torch torchvision
pip install numpy
pip install mne
pip install pyyaml
pip install tqdm
pip install tensorboard
```

### Clone Repository

```bash
git clone https://github.com/your-username/RL-BioAug.git
cd RL-BioAug
```

---

## Project Structure

```
RL-BioAug/
├── main.py                    # Main training orchestrator (2-phase pipeline)
├── linear_probing.py          # Evaluation script for linear probing
├── configs/                   # Configuration files (YAML)
│   ├── sleep_edf_config_topk_*.yaml
│   ├── sleep_edf_linear_probing.yaml
│   ├── ptb_xl_config.yaml
│   └── sleep_edf_comparison.yaml
└── src/
    ├── main_search.py         # Phase 1: RL agent training
    ├── main_retrain.py        # Phase 2: SSL retraining with fixed agent
    ├── loss.py                # NT-Xent loss implementation
    ├── data/
    │   ├── dataset.py         # Dataset classes (SleepEDF, CHB-MIT)
    │   ├── sleep_edfx_parser.py
    │   └── chb_mit_parser.py
    ├── models/
    │   ├── resnet.py          # ResNet1D18 encoder
    │   └── policy.py          # ContextAgent (Transformer-based policy)
    └── transforms/
        └── augmentation.py    # Augmentation strategies
```

---

## Quick Start

### 1. Prepare Data

```bash
# Sleep-EDFX dataset
python src/data/sleep_edfx_parser.py \
    --data_dir /path/to/sleep-edfx-raw \
    --output_dir ./data/sleep-edfx/

# CHB-MIT dataset
python src/data/chb_mit_parser.py \
    --data_dir /path/to/chb-mit-raw \
    --output_dir ./data/chb-mit/
```

### 2. Train

```bash
python main.py --config configs/sleep_edf_config_topk_1.yaml --mode all
```

### 3. Evaluate

```bash
python linear_probing.py \
    --config configs/sleep_edf_linear_probing.yaml \
    --checkpoint_path checkpoints/experiment_id/final_encoder_retrained.pth
```

---

## Data Preparation

### Sleep-EDFX Dataset

Download from [PhysioNet Sleep-EDFX](https://physionet.org/content/sleep-edfx/1.0.0/)

```bash
python src/data/sleep_edfx_parser.py \
    --data_dir /path/to/sleep-edfx-raw \
    --output_dir ./data/sleep-edfx/
```

**Details:**
- 5 sleep stages: W, N1, N2, N3, REM
- 30-second EEG epochs
- EEG Fpz-Cz channel, bandpass filtered (0.5-40 Hz)

### CHB-MIT Seizure Dataset

Download from [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/1.0.0/)

```bash
python src/data/chb_mit_parser.py \
    --data_dir /path/to/chb-mit-raw \
    --output_dir ./data/chb-mit/ \
    --chunk_duration 4.0
```

**Details:**
- Binary classification: seizure / background
- 4-second epochs
- Configurable channel selection

---

## Training

RL-BioAug uses a **two-phase training pipeline**:

### Phase 1: RL Agent Search

The RL agent learns optimal augmentation policies using KNN reward signals.

```bash
python main.py --config configs/sleep_edf_config_topk_1.yaml --mode search
```

### Phase 2: SSL Retraining

The encoder is retrained with the fixed (frozen) agent using NT-Xent loss.

```bash
python main.py --config configs/sleep_edf_config_topk_1.yaml --mode retrain
```

### Full Pipeline

Run both phases sequentially:

```bash
python main.py --config configs/sleep_edf_config_topk_1.yaml --mode all
```

### Training Modes

| Mode | Description |
|------|-------------|
| `search` | Phase 1 only - RL agent discovery |
| `retrain` | Phase 2 only - SSL retraining with fixed agent |
| `all` | Both phases sequentially |

---

## Evaluation

### Linear Probing

Evaluate the learned representations using linear probing:

```bash
python linear_probing.py \
    --config configs/sleep_edf_linear_probing.yaml \
    --checkpoint_path checkpoints/experiment_name/final_encoder_retrained.pth
```

### Outputs

- **Checkpoints:** `checkpoints/experiment_name/`
  - `best_agent.pth` - Best RL agent (Phase 1)
  - `final_encoder_retrained.pth` - Final encoder weights (Phase 2)

- **Logs:** `runs/experiment_name/`
  - TensorBoard logs for metrics and action frequencies

---

## Augmentation Strategies

RL-BioAug supports **5 learnable augmentation strategies**:

| Strategy | Description |
|----------|-------------|
| **Time Masking** | Zero-out random temporal segments (1/4 of signal) |
| **Time Permutation** | Shuffle 5 temporal segments |
| **Crop Resize** | Crop (30%-90%) and interpolate back to original length |
| **Time Flip** | Reverse the signal in time |
| **Time Warp** | Dynamically warp 5 temporal segments with random ratios |

### Weak Augmentation (Applied to all samples)

- **Scaling:** Uniform random multiplier (1 ± 2%)
- **Gaussian Noise:** σ = 0.01

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lee2025rlbioaug,
  title={RL-BioAug: Label-Efficient Reinforcement Learning for Self-Supervised EEG Representation Learning},
  author={Lee, Choel-Hui and others},
  journal={Under Review},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Choel-Hui Lee
```

---

## Acknowledgements

- [PhysioNet](https://physionet.org/) for providing EEG datasets
- Sleep-EDFX and CHB-MIT dataset contributors
