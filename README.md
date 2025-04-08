# Continual Learning Framework

This repository contains a modular framework for Continual Learning (CL) research. It is structured to enable easy experimentation with various CL approaches, datasets, network architectures, and logging strategies.

## 📁 Directory Overview

Each directory in the `src/` folder is designed with a specific responsibility in the CL pipeline:

### `approaches/`
Implements different continual learning strategies and algorithms.

- Contains methods such as:
  - **EWC** (Elastic Weight Consolidation)
  - **LwF** (Learning without Forgetting)
- Each approach typically includes logic for loss calculation, knowledge retention, and backward transfer control.

---

### `datasets/`
Contains dataset loaders and utilities tailored for continual learning scenarios.

- Supports various CL benchmarks like:
  - Split MNIST
  - Split FMNIST
  - Split CIFAR10
  - Split CIFAR-100
  - Permuted MNIST
- Handles incremental task splitting, normalization, and data loading across tasks.

---

### `layers/`
Contains custom layer implementations, which are especially useful in probabilistic or Bayesian continual learning.

- Includes:
  - Bayesian linear layers
  - Variational inference layers
  - Custom dropout or uncertainty-aware modules
- Useful for modeling uncertainty and improving robustness over time.

---

### `loggers/`
Logging utilities for tracking training progress and evaluation results.

- Supports:
  - TensorBoard logging
  - CSV or JSON export for metrics
  - Custom in-memory or live plotting loggers
- Helpful for monitoring continual performance, forgetting, and generalization.

---

### `networks/`
Implements the backbone neural network architectures used in continual learning.

- Includes:
  - **Bayesian ResNet-18**
  - **Bayesian ResNet-32**
  - **Bae=yesian LeNet**
- These models are usually adapted to support dynamic output heads and continual task expansion.

---

## 🔧 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```
### 2. Run Experiments
``` bash

python src/main_incremental.py --approach lwf --nepochs 200 --batch-size 128 --num-workers 4 --datasets fmnist --num-tasks 5 --nc-first-task 2 --lr 0.05 --weight-decay 5e-4 --clipping 1  --network bbbresnet18  --momentum 0.9 --exp-name exp1 --seed 0
```
Alternatively,
```bash
sh run.sh
```

> ### 🙏 Acknowledgements
> This repository is **heavily inspired by and builds upon** the following open-source projects:
>
> - 🔥 [**FACIL**](https://github.com/mmasana/FACIL) by Marc Masana et al.  
>   A strong baseline framework for continual learning research. We thank the authors for their contribution to the CL community.
>
> - 🍀 [**PyTorch-BayesianCNN**](https://github.com/kumar-shridhar/PyTorch-BayesianCNN) by Shridhar Kumar  
>   The Bayesian layer implementations in this repo are adapted and extended from this excellent resource.

