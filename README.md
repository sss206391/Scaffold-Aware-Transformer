# Scaffold-Aware Transformer

[![DOI](https://img.shields.io/badge/DOI-10.1186%2Fs13321--026--01221--6-blue)](https://doi.org/10.1186/s13321-026-01221-6)

Title : **Novel Molecular Design via a Scaffold-Aware Transformer with Multi-Scale Attention Mechanisms**

Authors : ***Junyoung Park, Sunyong Yoo***

## Description

We present a scaffold-aware generative framework for de novo molecular design that integrates a transformer-based generator with a graph attention network (GAT)-based predictor. Our approach enables explicit control over molecular scaffolds, the core structural frameworks that define topology and guide key substituent vectors in drug discovery, while simultaneously optimizing for high binding affinity through a cyclic learning mechanism.

## Datasets
- Original GuacaMol benchmark dataset: Available at https://github.com/BenevolentAI/guacamol
- Original bioactivity datasets (KOR and PIK3CA): Available at https://github.com/larngroup/DiverseDRL

## Requirements
This package requires:

- Python : 3.7.16
- moses : 0.10.0
- scikit-learn : 0.24.2 (for QSAR scoring function)
- networkx : 2.6.3
- rdkit : 2023.9.2
- torch : 1.12.1
- numpy
- pandas
- scipy
- tqdm (for training Prior)


## Usage
```
# Pre-train generator
python train/1_pretraining_generator.py

# Pre-train predictor for KOR
python train/1_pretraining_predictor.py --project_name predictor_kor

# Evaluate pre-trained models
python train/2_evaluate_generator.py
python train/2_evaluate_predictor.py --project_name predictor_kor

# Fine-tune generator with KOR predictor
python train/3_fine_tuning.py --project_name kor

# Evaluate fine-tuned generator
python train/4_evaluate_fine_tuning.py --project_name kor
```

## Contacts
If you have any questions or suggestions, please contact us by e-mail.

Junyoung Park, sss206391@gmail.com

Sunyong Yoo, syyoo@jnu.ac.kr

## Citation
If you use this code or data in your research, please cite our paper:

> Park, J., & Yoo, S. (2026). Novel molecular design via a scaffold-aware transformer with multi-scale attention mechanisms. Journal of Cheminformatics.

**DOI:** [10.1186/s13321-026-01221-6](https://doi.org/10.1186/s13321-026-01221-6)

**Paper:** [Read the full article here](https://doi.org/10.1186/s13321-026-01221-6)
