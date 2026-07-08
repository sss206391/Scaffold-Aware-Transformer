<<<<<<< HEAD
# Few_Shot CoT 기반 임상시험 프로토콜 생성

## Requirements

- Python : 3.9.25
- torch>=2.0.0                
- transformers>=4.40.0
- faiss-cpu>=1.7.4
- nltk>=3.8.1
- rouge-score>=0.1.2
- numpy>=1.24.0
- tqdm>=4.65.0

## NLTK 데이터 다운로드
```
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

## HuggingFace 모델
- Embedding:
  - Qwen/Qwen3-Embedding-4B
- LLM :
  - meta-llama/Llama-3.1-8B-Instruct (HF 토큰 필요)
  - google/gemma-2-9b (HF 토큰 필요)
  - Qwen/Qwen2.5-7B-Instruct
=======
# Scaffold-Aware Transformer

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
>>>>>>> 40443653f5d0a2fe102e7185b9ac4e4d2056eac2
