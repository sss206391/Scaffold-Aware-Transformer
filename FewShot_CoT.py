"""
RAG 기반 임상시험 프로토콜 생성 연구
1. FAISS에서 유사 임상시험 검색
2. 검색된 실제 데이터를 Few-shot 예시로 활용
3. LLaMA 3.1-instruct으로 프로토콜 생성
"""

import os
import json
import time
import pickle
import random
import numpy as np
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline, set_seed
import faiss

# 1. 설정
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 경로 설정
INDEX_PATH = "faiss_index"

# 모델 설정
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
# EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# GPU 설정
EMBEDDING_DEVICE = "cuda:0"  # 임베딩 모델용 GPU
LLM_DEVICE = "cuda:1"        # LLM용 GPU

# 검색할 유사 임상시험 개수 설정
TOP_K = 3  

# 2. 시드고정
SEED = 42
def set_all_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # Hugging Face transformers seed
    
    # For CUDA deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Seed 고정 [{seed}]")

# 3. 벡터DB 및 모델 로딩
class ClinicalTrialRAG:
    def __init__(self):
        self.index = None
        self.nct_ids = None
        self.study_data = None
        self.embed_tokenizer = None
        self.embed_model = None
        self.llm_pipe = None
        self.llm_tokenizer = None
        self.llm_model = None
    
    def load_faiss_index(self):
        # FAISS 인덱스 및 메타데이터 로딩
        start = time.time()
        print("**FAISS 인덱스 로딩 진행**")
        
        self.index = faiss.read_index(os.path.join(INDEX_PATH, "clinical_trials.index"))
        
        with open(os.path.join(INDEX_PATH, "nct_ids.pkl"), 'rb') as f:
            self.nct_ids = pickle.load(f)
        
        with open(os.path.join(INDEX_PATH, "study_data.pkl"), 'rb') as f:
            self.study_data = pickle.load(f)
        
        print(f"[FAISS 인덱스 로딩 완료] : {self.index.ntotal:,}개 ({time.time() - start:.1f}초)")
    
    def load_embedding_model(self):
        # 임베딩 모델 로딩
        start = time.time()
        print("**임베딩 모델 로딩 진행**")
        
        self.embed_tokenizer = AutoTokenizer.from_pretrained(
            EMBEDDING_MODEL, 
            trust_remote_code=True
        )
        self.embed_model = AutoModel.from_pretrained(
            EMBEDDING_MODEL,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(EMBEDDING_DEVICE)
        self.embed_model.eval()
        
        print(f"[임베딩 모델 로딩 완료] : ({time.time() - start:.1f}초)")
    
    def load_llm(self):
        # LLM 로딩
        start = time.time()
        print("**LLM 로딩 진행**")
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        # 수정: int로 변환
        gpu_id = int(LLM_DEVICE.split(":")[-1])
        
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map={"": gpu_id},  # int로 전달
            trust_remote_code=True
        )
        
        self.llm_pipe = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            torch_dtype=torch.float16,
        )
        
        print(f"[LLM 로딩 완료] : ({time.time() - start:.1f}초)")
    
    def load_all(self):
        # FAISS 인덱스, 임베딩 모델, LLM 모델 로딩
        self.load_faiss_index()
        self.load_embedding_model()
        self.load_llm()
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        # 사용자 입력 쿼리를 벡터로 변환
        with torch.no_grad():
            inputs = self.embed_tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(EMBEDDING_DEVICE)
            
            outputs = self.embed_model(**inputs)
            
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().astype('float32')
    
    def search_similar(self, query: str, top_k: int = TOP_K) -> list:
        # 유사한 임상시험 검색 함수
        start = time.time()
        
        # 쿼리 임베딩 가져오기
        query_embedding = self.get_query_embedding(query)
        
        # FAISS 검색
        scores, indices = self.index.search(query_embedding, top_k)
        
        # 결과 구성
        results = []
        for i, idx in enumerate(indices[0]):
            nct_id = self.nct_ids[idx]
            study = self.study_data[nct_id]
            results.append({
                "nct_id": nct_id,
                "score": float(scores[0][i]),
                "data": study
            })
        
        print(f"[유사 임상시험 검색 완료] : ({time.time() - start:.2f}초)")
        for r in results:
            title = r["data"]["Study Overview"].get("Brief Title", "N/A")[:50]
            print(f"   - {r['nct_id']}: {title}... (유사도: {r['score']:.3f})")
        
        return results
    
    def build_rag_prompt(self, user_input: str, similar_studies: list) -> str:
        # "RAG 기반 Few-Shot CoT 프롬프트

        system_prompt = """You are an expert clinical trial protocol designer with extensive experience in regulatory compliance and study design.
    
    ## Your Role:
    - Synthesize information from similar clinical trials and user requirements to generate a complete, scientifically rigorous clinical trial protocol
    - Apply your expertise in clinical research methodology, biostatistics, and regulatory guidelines (ICH-GCP, FDA, EMA)
    
    ## Task Instructions:
    1. **Analyze Similar Studies**: Review the provided reference clinical trials to understand established patterns and best practices for the given condition/intervention
    2. **Synthesize User Requirements**: Combine user input with insights from similar studies to design an appropriate protocol
    3. **Step-by-Step Reasoning**: Explain your reasoning process for each design decision, including:
       - Why specific study design elements were chosen
       - How the protocol addresses potential biases
       - Justification for eligibility criteria
    4. **Generate Complete Protocol**: Produce a comprehensive protocol with three main sections:
       - **Study Overview**: Title, summary, conditions, study type, phases, sponsor, interventions
       - **Participation Criteria**: Inclusion/exclusion criteria, age, sex, volunteer status
       - **Study Plan**: Design information, arm groups, primary/secondary outcomes
    
    ## Output Requirements:
    - Reasoning process: Detailed explanation in the user's input language
    - Final JSON output: ALL field values MUST be in English
    - Follow the exact JSON structure shown in the examples
    - Ensure scientific accuracy and regulatory compliance"""
    
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
               
        # Few-shot 예시 

        for i, study in enumerate(similar_studies, 1):
            nct_id = study["nct_id"]
            data = study["data"]
            overview = data.get("Study Overview", {})
            
            example_input = f"""
    Reference Clinical Trial #{i} ({nct_id}):
    - Title: {overview.get('Brief Title', 'N/A')}
    - Conditions: {', '.join(overview.get('Conditions', []))}
    - Study Type: {overview.get('Study Type', 'N/A')}
    - Interventions: {', '.join([inv.get('Name', '') for inv in overview.get('Interventions', [])])}
    """
            
            example_output = json.dumps(data, ensure_ascii=False, indent=2)
            reasoning = self._generate_reasoning(data)
            
            prompt += f"""
    <|start_header_id|>user<|end_header_id|>
    
    {example_input}<|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    
    {reasoning}
    
    Final Output:
    ````json
    {example_output}
    ```<|eot_id|>
    """
        
        # 유저 프롬프트: 구체적인 작업 요청
        
        user_prompt = f"""
    ## User Request:
    {user_input}
    
    ## Instructions for This Request:
    1. Analyze the above research requirements and the {len(similar_studies)} reference clinical trials provided
    2. Identify key design elements: study type, intervention type, target population, primary endpoints
    3. Explain your reasoning process step-by-step for each protocol section
    4. Generate a complete clinical trial protocol in JSON format
    
    Please proceed with your analysis and protocol generation."""
    
        prompt += f"""
    <|start_header_id|>user<|end_header_id|>
    
    {user_prompt}<|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    
    ## 추론 과정 (Reasoning Process):
    
    ### 1. 연구 개요 분석 (Study Overview Analysis):
    """
        
        return prompt

    def _generate_reasoning(self, data: dict) -> str:
        # 추론과정 출력
        overview = data.get("Study Overview", {})
        criteria = data.get("Participation Criteria", {})
        plan = data.get("Study Plan", {})
        
        study_type = overview.get("Study Type", "N/A")
        phases = overview.get("Phases", [])
        conditions = overview.get("Conditions", [])
        interventions = overview.get("Interventions", [])
        allocation = plan.get("Design Information", {}).get("Allocation", "N/A")
        masking = plan.get("Design Information", {}).get("Masking", "N/A")
        purpose = plan.get("Design Information", {}).get("Primary Purpose", "N/A")
        
        intervention_types = [inv.get("Type", "") for inv in interventions]
        intervention_names = [inv.get("Name", "") for inv in interventions]
        
        reasoning = f"""## 추론 과정 (Reasoning Process):
    
    ### 1. 연구 개요 분석 (Study Overview Analysis):
    - 대상 질환: "{', '.join(conditions)}"
    - 중재 방법: "{', '.join(intervention_names)}" ({', '.join(intervention_types)} 유형)
    - 연구 유형 결정: 참가자에게 적극적인 중재가 수행되므로 {study_type}이 적절합니다.
    - 임상시험 단계: {"약물 연구이므로 Phase가 적용됩니다." if "DRUG" in intervention_types else "약물 연구가 아니므로 Phase는 NA입니다."}
    - **결론**: Study Type = {study_type}, Phases = {', '.join(phases) if phases else 'NA'}
    
    ### 2. 참여 기준 분석 (Participation Criteria Analysis):
    - 대상 집단: {', '.join(conditions)} 환자
    - 건강한 자원자: {criteria.get('Healthy Volunteers', 'N/A')} - {"해당 질환 환자만 참여 가능" if criteria.get('Healthy Volunteers') == 'No' else "건강한 자원자 참여 가능"}
    - 연령 범위: {criteria.get('Minimum Age', 'N/A')} ~ {criteria.get('Maximum Age', 'N/A')}
    - 성별 제한: {criteria.get('Sex', 'N/A')} - {"성별 제한 없음" if criteria.get('Sex') == 'ALL' else "특정 성별 기준 적용"}
    
    ### 3. 연구 설계 분석 (Study Plan Analysis):
    - 주요 목적: {purpose} - {"진단 도구 효과 평가" if purpose == "DIAGNOSTIC" else "치료 효과 평가" if purpose == "TREATMENT" else "특정 연구 목적"}
    - 배정 방법: {allocation} - {"무작위 배정으로 선택 편향 최소화" if allocation == "RANDOMIZED" else "비무작위 배정"}
    - 눈가림: {masking} - {"공개 라벨 연구 (눈가림 없음)" if masking == "NONE" else f"{masking} 수준 눈가림 적용"}
    - 연구 모델: {plan.get("Design Information", {}).get("Intervention Model", "N/A")}
    
    ### 4. 종합 결론:
    위 분석을 바탕으로, 본 프로토콜은 {study_type} 설계, {allocation} 배정, {masking} 눈가림, {purpose}를 주요 목적으로 구성합니다.
    
    Final Output (in English):"""
        
        return reasoning
        
    # def _generate_reasoning(self, data: dict) -> str:
    #     """Generate English CoT reasoning from data"""
    #     overview = data.get("Study Overview", {})
    #     criteria = data.get("Participation Criteria", {})
    #     plan = data.get("Study Plan", {})
        
    #     study_type = overview.get("Study Type", "N/A")
    #     phases = overview.get("Phases", [])
    #     conditions = overview.get("Conditions", [])
    #     interventions = overview.get("Interventions", [])
    #     allocation = plan.get("Design Information", {}).get("Allocation", "N/A")
    #     masking = plan.get("Design Information", {}).get("Masking", "N/A")
    #     purpose = plan.get("Design Information", {}).get("Primary Purpose", "N/A")
        
    #     intervention_types = [inv.get("Type", "") for inv in interventions]
    #     intervention_names = [inv.get("Name", "") for inv in interventions]
        
    #     reasoning = f"""Reasoning Process:
    # 1. Study Overview Analysis:
    #    - Study Type: {study_type}
    #    - Phases: {', '.join(phases) if phases else 'N/A'}
    #    - Conditions: {', '.join(overview.get('Conditions', []))}
    
    # 2. Participation Criteria Analysis:
    #    - Healthy Volunteers: {criteria.get('Healthy Volunteers', 'N/A')}
    #    - Sex: {criteria.get('Sex', 'N/A')}
    #    - Age: {criteria.get('Minimum Age', 'N/A')} ~ {criteria.get('Maximum Age', 'N/A')}
    
    # 3. Study Plan Analysis:
    #    - Allocation: {allocation}
    #    - Masking: {masking}
    #    - Primary Purpose: {purpose}
    
    # Final Output:"""
        
    #     return reasoning
    
    def generate_protocol(self, user_input: str, max_new_tokens: int = 2048) -> dict:
        # 생성함수

        set_all_seeds(SEED)
        
        # 1. 유사 임상시험 검색
        print("\n 1. 유사 임상시험 검색")
        similar_studies = self.search_similar(user_input, TOP_K)
        
        # 2. RAG 프롬프트 구성
        print("\n 2. 프롬프트 구성")
        prompt = self.build_rag_prompt(user_input, similar_studies)
        
        # 3. LLM 생성
        print("\n 3. 프로토콜 생성 중...")
        start = time.time()
        
        outputs = self.llm_pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False
        )
        
        generated_text = outputs[0]["generated_text"]
        exe_time=(time.time() - start)
        
        if exe_time>=60:
            print(f"[생성 완료] : ({exe_time/60}분, {(exe_time%60)})")
        else:
            print(f"[생성 완료] : ({exe_time}초)")
        
        try:
            # Extract reasoning (before JSON block)
            json_start = generated_text.find("```json")
            if json_start != -1:
                reasoning_text = generated_text[:json_start].strip()
                json_end = generated_text.find("```", json_start + 7)
                if json_end != -1:
                    json_str = generated_text[json_start + 7:json_end].strip()
                    result = json.loads(json_str)
            else:
                # Try to find JSON without code block
                json_start = generated_text.find("{")
                if json_start != -1:
                    reasoning_text = generated_text[:json_start].strip()
                    json_end = generated_text.rfind("}") + 1
                    json_str = generated_text[json_start:json_end]
                    result = json.loads(json_str)
                
        except json.JSONDecodeError:
            print("⚠️ JSON parsing failed")
            result = {"raw_output": generated_text}
        
        return result, generated_text, similar_studies, reasoning_text
    
    
    def release_gpu(self):
        # GPU 해제
        if self.embed_model is not None:
            self.embed_model.cpu()
            del self.embed_model
        if self.embed_tokenizer is not None:
            del self.embed_tokenizer
        if self.llm_model is not None:
            self.llm_model.cpu()
            del self.llm_model
        if self.llm_pipe is not None:
            del self.llm_pipe
        if self.llm_tokenizer is not None:
            del self.llm_tokenizer
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU 메모리 해제 완료")

# 4. 메인 실행
#---------------------------------------

if __name__ == "__main__":
    rag = ClinicalTrialRAG()
    
    try:
        # 모델 로딩
        rag.load_all()
        
        # 사용자 입력
        user_input = """
사용자 입력 정보:
- 연구 주제: 폐색전증 의심 환자에서 초음파 검사의 진단적 유용성
- 연구 목적: CT 또는 폐환기관류스캔 의뢰 감소
- 대상: 응급실 내원 성인 환자
- 중재: Point-of-care 초음파 검사
- 기간: 24시간 내 결과 확인
"""
        print("main 시작")
        print("\n" + "="*70)
        print("RAG 기반 임상시험 프로토콜 생성")
        print("="*70)
        print(f"\n입력:\n{user_input}")
        
        # 프로토콜 생성
        result, raw_output, similar, reasoning = rag.generate_protocol(user_input)
         
        # 추론과정 출력
        print("\n" + "="*70)
        print(" 추론 과정 출력 (CoT)")
        print("="*70)
        print(reasoning if reasoning else "No explicit reasoning found in output")
        
        # 결과 생성
        print("\n" + "="*70)
        print(" 생성된 프로토콜 출력 ")
        print("="*70)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        
        # 결과 저장
        os.makedirs("Results", exist_ok=True)
        
        output_data = {
            "input": user_input,
            "similar_studies": [{"nct_id": s["nct_id"], "score": s["score"]} for s in similar],
            "reasoning": reasoning,
            "generated_protocol": result,
            "raw_output": raw_output
        }
        
        with open("Results/rag_generated_protocol.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print("생성 완료")
        
    finally:
        rag.release_gpu()  