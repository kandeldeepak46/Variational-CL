# Exempler-Free Variational Continual Learning

This repository contains a modular framework for Continual Learning (CL) research. It is structured to enable easy experimentation with various CL approaches, datasets, network architectures, and logging strategies.

## 📁 Directory Overview

Each directory in the `src/` folder is designed with a specific responsibility in the CL pipeline:

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
### 📦 Datasets

To experiment with datasets like TinyImageNet, you can download it from the following source and put in in ../data directory:

- [TinyImageNet-200 on Kaggle](https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200)

> ### 🙏 Acknowledgements
> This repository is **heavily inspired by and builds upon** the following open-source projects:
>
> - 🔥 [**FACIL**](https://github.com/mmasana/FACIL) by Marc Masana et al.  
>   A strong baseline framework for continual learning research. We thank the authors for their contribution to the CL community.
>
> - 🍀 [**PyTorch-BayesianCNN**](https://github.com/kumar-shridhar/PyTorch-BayesianCNN) by Shridhar Kumar  
>   The Bayesian layer implementations in this repo are adapted and extended from this excellent resource.

