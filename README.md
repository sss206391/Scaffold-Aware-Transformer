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
