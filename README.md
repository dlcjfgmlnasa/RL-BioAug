# RL-BioAug: Self-Adaptive Data Augmentation for Self-Supervised EEG Representation Learning

## Abstract
Contrastive learning has emerged as a promising paradigm in EEG analysis by reducing reliance on labeled data. In general, the effectiveness of contrastive learning fundamentally depends on the quality of data augmentation due to non-stationarity of EEG signals with statistical properties changing over time. Static or random augmentation strategies can corrupt intrinsic information, therefore leading to degraded performance in representation learning. To address this, we propose RL-BioAug, a reinforcement learning (RL)-based autonomous augmentation framework for self-supervised EEG representation learning. The proposed RL agent analyzes the features of input signals to determine an optimal augmentation policy tailored to each instance. Experimental results demonstrated that our method achieves superior performance compared to existing methods on the Sleep-EDFX and CHB-MIT datasets. Notably, this agent adaptively chose optimal strategies---for example, 'Time Masking' for sleep stage classification and 'Crop Resize' for seizure detection. Our framework suggests its potential to replace conventional heuristic-based augmentations and establish a new autonomous paradigm for data augmentation.

## 🚀 Key Features

* **RL-Driven Adaptive Augmentation:** A Transformer-based agent observes the latent state of the EEG signal and selects the optimal augmentation technique from a candidate pool.
* **Dual Augmentation Mechanism:**
    * **Strong Augmentation (Agent Action):** Applies intensive transformations (e.g., Time Masking, Permutation, Crop & Resize, Time Flip, Time Warp) to create challenging views for the encoder.
    * **Weak Augmentation (Anchor View):** Applies minimal perturbations (Jittering + Scaling) to serve as a stable **semantic anchor**, preserving the original signal patterns.
* **Soft-KNN Consistency Reward:** A quantitative metric that evaluates the quality of the learned representation by measuring the density of same-class neighbors in the embedding space.
* **Cycle-Consistent Training:** The framework alternates between the **SSL Step** (Encoder update via contrastive loss) and the **RL Step** (Agent update via policy gradient) to achieve mutual improvement.

---

## 🛠️ Framework Architecture

<div align="center">
  <img src="./assets/framework.png" alt="RL-BioAug Framework" width="90%">
</div>

The training process is modeled as a **Markov Decision Process (MDP)**:

1.  **State ($s_t$):** The agent observes the latent vector extracted from the frozen encoder.
2.  **Action ($a_t$):** The agent selects a specific strong augmentation technique.
3.  **Reward ($r_t$):** The agent receives a reward based on the **Soft-KNN consistency score**, computed using a reference set.
4.  **Optimization:** The encoder minimizes contrastive loss, while the agent maximizes the expected reward.

---

## 📦 Installation

This project requires **Python 3.8+** and **PyTorch**.

```bash
# 1. Clone the repository
git clone [https://github.com/dlcjfgmlnasa/RL-BioAug.git](https://github.com/dlcjfgmlnasa/RL-BioAug.git)
cd RL-BioAug

# 2. Install dependencies
pip install -r requirements.txt
